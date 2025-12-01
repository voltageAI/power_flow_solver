defmodule PowerFlowSolver.MatpowerParser do
  @moduledoc """
  Parser for MATPOWER case files (.m format).

  Parses MATPOWER files into the solver-compatible format with buses and lines.
  All values are converted to per-unit on the system MVA base.

  ## Example

      {:ok, system} = PowerFlowSolver.MatpowerParser.parse_file("case118.m")
      {:ok, solution, iterations} = PowerFlowSolver.NewtonRaphson.solve(system)
  """

  @doc """
  Parses a MATPOWER .m file and returns a solver-compatible system map.

  Returns `{:ok, system}` or `{:error, reason}` where system contains:
  - `:buses` - List of bus maps with solver-compatible fields
  - `:lines` - List of line maps with solver-compatible fields
  - `:base_mva` - System MVA base (typically 100)
  """
  def parse_file(filepath) do
    with {:ok, content} <- File.read(filepath),
         {:ok, data} <- parse_content(content) do
      {:ok, data}
    end
  end

  @doc """
  Parses MATPOWER content string.
  """
  def parse_content(content) do
    with {:ok, bus_data} <- extract_section(content, "mpc.bus"),
         {:ok, gen_data} <- extract_section(content, "mpc.gen"),
         {:ok, branch_data} <- extract_section(content, "mpc.branch") do
      base_mva = extract_base_mva(content)
      {buses, id_mapping} = parse_buses(bus_data, gen_data, base_mva)
      lines = parse_branches(branch_data, id_mapping, base_mva)

      {:ok, %{buses: buses, lines: lines, base_mva: base_mva}}
    end
  end

  # Extract baseMVA value
  defp extract_base_mva(content) do
    case Regex.run(~r/mpc\.baseMVA\s*=\s*(\d+)/, content, capture: :all_but_first) do
      [value] -> String.to_integer(value)
      _ -> 100
    end
  end

  # Extract a data section from MATPOWER file
  defp extract_section(content, section_name) do
    case Regex.run(~r/#{section_name}\s*=\s*\[/, content) do
      nil ->
        {:error, "Section #{section_name} not found"}

      _ ->
        regex = ~r/#{section_name}\s*=\s*\[(.*?)\];/s

        case Regex.run(regex, content, capture: :all_but_first) do
          [data_str] ->
            {:ok, data_str}

          _ ->
            {:error, "Could not parse section #{section_name}"}
        end
    end
  end

  # Parse bus data section
  defp parse_buses(bus_str, gen_str, base_mva) do
    gen_buses = parse_generator_buses(gen_str, base_mva)

    parsed_buses =
      bus_str
      |> String.split("\n")
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 == "" or String.starts_with?(&1, "%")))
      |> Enum.map(&parse_bus_line(&1, gen_buses, base_mva))
      |> Enum.reject(&is_nil/1)

    # Create mapping from original bus IDs to new contiguous IDs (0-indexed)
    original_ids = Enum.map(parsed_buses, & &1.original_id)
    id_mapping = original_ids |> Enum.with_index() |> Map.new()

    # Renumber buses to be contiguous
    buses =
      parsed_buses
      |> Enum.with_index()
      |> Enum.map(fn {bus, new_id} ->
        bus
        |> Map.put(:id, new_id)
        |> Map.delete(:original_id)
      end)

    {buses, id_mapping}
  end

  # Parse generator buses to get PV bus info
  defp parse_generator_buses(gen_str, base_mva) do
    gen_str
    |> String.split("\n")
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == "" or String.starts_with?(&1, "%")))
    |> Enum.map(&parse_generator_line(&1, base_mva))
    |> Enum.reject(&is_nil/1)
    |> Enum.group_by(fn {bus_id, _} -> bus_id end)
    |> Enum.map(fn {bus_id, gens} ->
      # Sum up generation from multiple generators at same bus
      total_pg = Enum.reduce(gens, 0.0, fn {_, g}, acc -> acc + g.pg end)
      total_qg = Enum.reduce(gens, 0.0, fn {_, g}, acc -> acc + g.qg end)
      {_, first_gen} = List.first(gens)

      # Sum Q limits from all generators at this bus
      total_qmin = Enum.reduce(gens, 0.0, fn {_, g}, acc -> acc + (g.qmin || 0.0) end)
      total_qmax = Enum.reduce(gens, 0.0, fn {_, g}, acc -> acc + (g.qmax || 0.0) end)

      {bus_id,
       %{pg: total_pg, qg: total_qg, vg: first_gen.vg, qmin: total_qmin, qmax: total_qmax}}
    end)
    |> Map.new()
  end

  # Parse a single generator line
  # Format: bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin ...
  defp parse_generator_line(line, base_mva) do
    parts = parse_data_line(line)

    if length(parts) >= 6 do
      bus_id = trunc(Enum.at(parts, 0))
      pg = Enum.at(parts, 1) / base_mva
      qg = Enum.at(parts, 2) / base_mva
      qmax = if length(parts) >= 4, do: Enum.at(parts, 3) / base_mva, else: nil
      qmin = if length(parts) >= 5, do: Enum.at(parts, 4) / base_mva, else: nil
      vg = Enum.at(parts, 5)
      {bus_id, %{pg: pg, qg: qg, vg: vg, qmin: qmin, qmax: qmax}}
    else
      nil
    end
  end

  # Parse a single bus line
  # Format: bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
  defp parse_bus_line(line, gen_buses, base_mva) do
    parts = parse_data_line(line)

    if length(parts) >= 13 do
      bus_id = trunc(Enum.at(parts, 0))
      bus_type = trunc(Enum.at(parts, 1))
      pd = Enum.at(parts, 2) / base_mva
      qd = Enum.at(parts, 3) / base_mva
      _gs = Enum.at(parts, 4)
      _bs = Enum.at(parts, 5)
      vm = Enum.at(parts, 7)
      va = Enum.at(parts, 8)
      base_kv = Enum.at(parts, 9)

      type =
        case bus_type do
          3 -> :slack
          2 -> :pv
          1 -> if Map.has_key?(gen_buses, bus_id), do: :pv, else: :pq
          _ -> :pq
        end

      gen = Map.get(gen_buses, bus_id, %{pg: 0.0, qg: 0.0, vg: vm, qmin: nil, qmax: nil})

      %{
        id: bus_id - 1,
        original_id: bus_id,
        type: type,
        p_load: pd,
        q_load: qd,
        p_gen: gen.pg,
        q_gen: gen.qg,
        v_magnitude: if(type == :pq, do: vm, else: gen.vg),
        v_angle: va * :math.pi() / 180.0,
        base_kv: base_kv,
        q_min: gen.qmin,
        q_max: gen.qmax
      }
    else
      nil
    end
  end

  # Parse branch data section
  defp parse_branches(branch_str, id_mapping, _base_mva) do
    branch_str
    |> String.split("\n")
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == "" or String.starts_with?(&1, "%")))
    |> Enum.with_index()
    |> Enum.map(&parse_branch_line(&1, id_mapping))
    |> Enum.reject(&is_nil/1)
  end

  # Parse a single branch line
  # Format: fbus tbus r x b rateA rateB rateC ratio angle status angmin angmax
  defp parse_branch_line({line, idx}, id_mapping) do
    parts = parse_data_line(line)

    if length(parts) >= 11 do
      from_bus_orig = trunc(Enum.at(parts, 0))
      to_bus_orig = trunc(Enum.at(parts, 1))
      r = Enum.at(parts, 2)
      x = Enum.at(parts, 3)
      b = Enum.at(parts, 4)
      rate_a = if length(parts) >= 6, do: Enum.at(parts, 5), else: 0.0
      tap_ratio = if length(parts) >= 9, do: Enum.at(parts, 8), else: 0.0
      phase_shift = if length(parts) >= 10, do: Enum.at(parts, 9), else: 0.0
      status = trunc(Enum.at(parts, 10))

      from_bus = Map.get(id_mapping, from_bus_orig)
      to_bus = Map.get(id_mapping, to_bus_orig)

      valid_impedance = x > 0 or (x == 0 and r > 0)

      if status == 1 and from_bus != nil and to_bus != nil and valid_impedance do
        tap = if tap_ratio == 0.0, do: 1.0, else: tap_ratio
        shift_rad = phase_shift * :math.pi() / 180.0
        rating = if rate_a > 0, do: rate_a, else: nil

        %{
          id: idx,
          from: from_bus,
          to: to_bus,
          r: r,
          x: x,
          b: b,
          tap: tap,
          shift: shift_rad,
          rating_mva: rating
        }
      else
        nil
      end
    else
      nil
    end
  end

  defp parse_data_line(line) do
    line
    |> String.replace(";", "")
    |> String.split(~r/\s+/)
    |> Enum.reject(&(&1 == ""))
    |> Enum.map(&parse_number/1)
  end

  defp parse_number(str) do
    case Float.parse(str) do
      {num, _} -> num
      :error -> 0.0
    end
  end
end
