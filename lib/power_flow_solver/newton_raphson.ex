defmodule PowerFlowSolver.NewtonRaphson do
  @moduledoc """
  Clean Rust implementation of Newton-Raphson power flow solver.

  This module provides a wrapper around the complete Rust NIF implementation
  that runs the entire iteration loop in Rust, eliminating boundary crossings.

  ## Features

  - **Complete Rust Implementation**: Entire iteration loop runs in Rust
  - **Zero Boundary Crossings**: Only 2 calls (setup + result) vs 200+ in Elixir
  - **Clean Architecture**: Well-factored, maintainable Rust code
  - **Performance**: Expected 1.5-2.0s improvement on large systems

  ## Phase 3A Implementation

  This is Phase 3A of the Rust optimization plan. It implements:
  - ✅ Core Newton-Raphson iteration loop
  - ✅ Jacobian building in parallel
  - ✅ Power injection calculation
  - ✅ Convergence checking
  - ⏳ Q-limit enforcement (Phase 3B)
  - ⏳ Line search (Phase 3C)

  ## Usage

      system = %{
        buses: [...],
        y_bus: %{row_ptrs: ..., col_indices: ..., values: ...}
      }

      {:ok, result, iterations} = NewtonRaphsonRust.solve(system, tolerance: 1.0e-2)
  """

  alias PowerFlowSolver.SparseLinearAlgebra

  @doc """
  Solve power flow using complete Rust Newton-Raphson implementation.

  ## Arguments

  - `system` - Map containing:
    - `:buses` - List of Bus structs
    - `:y_bus` - Admittance matrix in CSR format
  - `opts` - Options:
    - `:max_iterations` - Maximum iterations (default: 100)
    - `:tolerance` - Convergence tolerance (default: 1.0e-2)
    - `:initial_voltage` - Provide custom initial voltage vector (default: flat start)
    - `:enforce_q_limits` - Enforce generator Q limits (default: false)
    - `:q_tolerance` - Q-limit tolerance (default: 1.0e-4)

  ## Returns

  - `{:ok, solution, iterations}` - Converged solution
  - `{:error, reason}` - Error description

  ## Example

      {:ok, solution, 5} = NewtonRaphsonRust.solve(system, tolerance: 1.0e-2)
  """
  def solve(system, opts \\ []) do
    max_iterations = Keyword.get(opts, :max_iterations, 100)
    tolerance = Keyword.get(opts, :tolerance, 1.0e-2)
    custom_init = Keyword.get(opts, :initial_voltage)
    enforce_q_limits = Keyword.get(opts, :enforce_q_limits, false)
    q_tolerance = Keyword.get(opts, :q_tolerance, 1.0e-4)

    # CRITICAL: Sort buses by ID to ensure correct alignment with Y-bus matrix
    # The Y-bus uses bus IDs as row/column indices, so buses must be in order
    sorted_buses = Enum.sort_by(system.buses, & &1.id)
    system = %{system | buses: sorted_buses}

    # Build Y-bus if not provided
    y_bus = build_y_bus(system)

    # Initialize voltage vector
    # If custom initial voltage not provided, use flat start (1.0 p.u. at 0 degrees)
    voltage =
      if custom_init != nil do
        custom_init
      else
        # Use flat start
        initialize_voltage(system.buses)
      end

    # Prepare bus data for Rust
    bus_data = prepare_bus_data(system.buses)

    # Prepare Y-bus data
    y_bus_data = prepare_y_bus_data(y_bus)

    # Convert voltage to list of {magnitude, angle} tuples
    voltage_list = voltage_to_list(voltage)

    # Call Rust NIF
    case SparseLinearAlgebra.solve_power_flow_rust(
           bus_data,
           y_bus_data,
           voltage_list,
           max_iterations,
           tolerance,
           enforce_q_limits,
           q_tolerance
         ) do
      {:ok, final_voltage, iterations, true, _final_mismatch} ->
        # Convert voltage back to map format for compatibility
        voltage_map = list_to_voltage_map(final_voltage, system.buses)
        {:ok, voltage_map, iterations}

      {:ok, _final_voltage, iterations, false, final_mismatch} ->
        {:error, "Failed to converge in #{iterations} iterations (mismatch: #{final_mismatch})"}

      {:error, [], 0, false, mismatch} when mismatch == 0.0 or mismatch == -0.0 ->
        {:error, "Rust solver internal error (check stderr for details)"}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # Initialize voltage using bus data (magnitude and angle from input)
  # For solved cases, this allows immediate convergence
  defp initialize_voltage(buses) do
    buses
    |> Enum.with_index()
    |> Enum.map(fn {bus, i} ->
      v_mag =
        if is_number(bus.v_magnitude),
          do: bus.v_magnitude,
          else: Decimal.to_float(bus.v_magnitude)

      # Get angle (already in radians from parser)
      v_angle_rad =
        if Map.has_key?(bus, :v_angle) and bus.v_angle != nil do
          if is_number(bus.v_angle),
            do: bus.v_angle,
            else: Decimal.to_float(bus.v_angle)
        else
          0.0
        end

      # Solver format uses :type (not :bus_type)
      bus_type = if is_map(bus) and Map.has_key?(bus, :type), do: bus.type, else: bus.bus_type

      case bus_type do
        :slack ->
          # Slack bus: use specified voltage and angle
          {i, {v_mag, v_angle_rad}}

        :pv ->
          # PV bus: use specified magnitude and angle
          {i, {v_mag, v_angle_rad}}

        :pq ->
          # PQ bus: use bus's voltage from input (for warm start/solved cases)
          # If no angle provided, defaults to 0.0
          {i, {v_mag, v_angle_rad}}
      end
    end)
    |> Map.new()
  end

  # Prepare bus data for Rust NIF
  # Returns list of {bus_type, p_scheduled, q_scheduled, v_scheduled, q_min, q_max, q_load}
  defp prepare_bus_data(buses) do
    Enum.map(buses, fn bus ->
      # Solver format uses :type (not :bus_type)
      type = if is_map(bus) and Map.has_key?(bus, :type), do: bus.type, else: bus.bus_type

      bus_type =
        case type do
          :slack -> 0
          :pv -> 1
          :pq -> 2
        end

      # P and Q are net injections: generation - load
      # Handle both float and Decimal types, and missing fields
      p_gen = Map.get(bus, :p_gen, 0.0)
      p_load = Map.get(bus, :p_load, 0.0)
      q_gen = Map.get(bus, :q_gen, 0.0)
      q_load = Map.get(bus, :q_load, 0.0)

      p_scheduled =
        if is_number(p_gen) and is_number(p_load) do
          p_gen - p_load
        else
          Decimal.to_float(Decimal.sub(p_gen, p_load))
        end

      q_scheduled =
        if is_number(q_gen) and is_number(q_load) do
          q_gen - q_load
        else
          Decimal.to_float(Decimal.sub(q_gen, q_load))
        end

      v_mag = Map.get(bus, :v_magnitude, 1.0)
      v_scheduled = if is_number(v_mag), do: v_mag, else: Decimal.to_float(v_mag)

      # Q-limits (convert to float if Decimal) - these are optional
      q_min_raw = Map.get(bus, :q_min)

      q_min =
        cond do
          is_nil(q_min_raw) -> nil
          is_number(q_min_raw) -> q_min_raw
          true -> Decimal.to_float(q_min_raw)
        end

      q_max_raw = Map.get(bus, :q_max)

      q_max =
        cond do
          is_nil(q_max_raw) -> nil
          is_number(q_max_raw) -> q_max_raw
          true -> Decimal.to_float(q_max_raw)
        end

      # Q_load needed for converting Q injection to Q generation
      q_load_float = if is_number(q_load), do: q_load, else: Decimal.to_float(q_load)

      {bus_type, p_scheduled, q_scheduled, v_scheduled, q_min, q_max, q_load_float}
    end)
  end

  # Prepare Y-bus data for Rust NIF
  defp prepare_y_bus_data(y_bus) do
    row_ptrs = tuple_or_list_to_list(y_bus.row_ptrs)
    col_indices = tuple_or_list_to_list(y_bus.col_indices)
    values = tuple_or_list_to_list(y_bus.values)

    {row_ptrs, col_indices, values}
  end

  # Convert voltage map/list to list of {magnitude, angle} tuples
  defp voltage_to_list(voltage) when is_map(voltage) do
    Enum.map(0..(map_size(voltage) - 1), fn i ->
      Map.get(voltage, i, {1.0, 0.0})
    end)
  end

  defp voltage_to_list(voltage) when is_list(voltage), do: voltage
  defp voltage_to_list(voltage) when is_tuple(voltage), do: Tuple.to_list(voltage)

  # Convert list of {magnitude, angle} tuples back to voltage map
  # Uses the actual bus IDs from the system, not array indices
  defp list_to_voltage_map(voltage_list, buses) do
    voltage_list
    |> Enum.zip(buses)
    |> Enum.map(fn {voltage, bus} ->
      bus_id = Map.get(bus, :id)
      {bus_id, voltage}
    end)
    |> Map.new()
  end

  # Helper to convert tuple or list to list
  defp tuple_or_list_to_list(data) when is_tuple(data), do: Tuple.to_list(data)
  defp tuple_or_list_to_list(data) when is_list(data), do: data

  # Build Y-bus if not already built, or return existing
  # This function is public to allow testing and external Y-bus construction
  def build_y_bus(%{y_bus_tuple: y_bus_tuple}) when not is_nil(y_bus_tuple),
    do: y_bus_tuple

  def build_y_bus(%{y_bus: y_bus}) when not is_nil(y_bus) and is_map(y_bus),
    do: y_bus

  def build_y_bus(system) do
    # Build Y-bus from branches and transformers with float conversion
    num_buses = length(system.buses)

    # Combine branches and transformers
    # Support both separate branches/transformers and combined lines format
    all_lines =
      cond do
        Map.has_key?(system, :lines) and not is_nil(system.lines) ->
          system.lines
        true ->
          (system.branches || []) ++ (system.transformers || [])
      end

    # Initialize sparse matrix storage
    entries =
      Enum.flat_map(all_lines, fn branch ->
        # Convert to float if needed (PowerSystemConverter already converts to floats)
        r = if is_number(branch.r), do: branch.r, else: Decimal.to_float(branch.r)
        x = if is_number(branch.x), do: branch.x, else: Decimal.to_float(branch.x)
        # Transformers don't have b field, branches do
        b_charging =
          cond do
            Map.has_key?(branch, :b) and is_number(branch.b) -> branch.b
            Map.has_key?(branch, :b) and branch.b != nil -> Decimal.to_float(branch.b)
            true -> 0.0
          end

        # Transformers use tap_ratio field, branches use tap
        tap =
          cond do
            Map.has_key?(branch, :tap_ratio) and branch.tap_ratio != nil ->
              Decimal.to_float(branch.tap_ratio)

            Map.has_key?(branch, :tap) and is_number(branch.tap) ->
              branch.tap

            Map.has_key?(branch, :tap) and branch.tap != nil ->
              Decimal.to_float(branch.tap)

            true ->
              1.0
          end

        # Transformers use phase_shift field
        shift =
          cond do
            Map.has_key?(branch, :phase_shift) and branch.phase_shift != nil ->
              Decimal.to_float(branch.phase_shift)

            Map.has_key?(branch, :shift) and is_number(branch.shift) ->
              branch.shift

            Map.has_key?(branch, :shift) and branch.shift != nil ->
              Decimal.to_float(branch.shift)

            true ->
              0.0
          end

        from_bus = if Map.has_key?(branch, :from), do: branch.from, else: branch.from_bus_number
        to_bus = if Map.has_key?(branch, :to), do: branch.to, else: branch.to_bus_number

        # Calculate series admittance: y = 1/(r + jx) = (r - jx)/(r² + x²)
        z_mag_sq = r * r + x * x
        g = r / z_mag_sq
        b_series = -x / z_mag_sq

        # Shunt admittance (split between both ends)
        b_shunt = b_charging / 2.0

        # Check if this is a transformer (tap != 1.0 or shift != 0)
        if tap != 1.0 or shift != 0.0 do
          # TRANSFORMER MODEL
          t_sq = tap * tap

          # Y_ii: y_series/t² + y_shunt
          y_ii = {g / t_sq, b_series / t_sq + b_shunt}

          # Y_jj: y_series + y_shunt
          y_jj = {g, b_series + b_shunt}

          # Y_ij: -y_series/(t * e^(j*shift))
          # = -y_series * e^(-j*shift) / t
          cos_shift = :math.cos(shift)
          sin_shift = :math.sin(shift)

          # Multiply -y_series by e^(-j*shift) = (cos_shift - j*sin_shift)
          # (-g - j*b) * (cos_shift - j*sin_shift)
          # = -g*cos_shift + j*g*sin_shift - j*b*cos_shift - b*sin_shift
          # = (-g*cos_shift - b*sin_shift) + j*(g*sin_shift - b*cos_shift)
          y_ij_re = (-g * cos_shift - b_series * sin_shift) / tap
          y_ij_im = (g * sin_shift - b_series * cos_shift) / tap
          y_ij = {y_ij_re, y_ij_im}

          # Y_ji: -y_series/(t * e^(-j*shift))
          # = -y_series * e^(j*shift) / t
          # Multiply -y_series by e^(j*shift) = (cos_shift + j*sin_shift)
          # (-g - j*b) * (cos_shift + j*sin_shift)
          # = -g*cos_shift - j*g*sin_shift - j*b*cos_shift + b*sin_shift
          # = (-g*cos_shift + b*sin_shift) + j*(-g*sin_shift - b*cos_shift)
          y_ji_re = (-g * cos_shift + b_series * sin_shift) / tap
          y_ji_im = (-g * sin_shift - b_series * cos_shift) / tap
          y_ji = {y_ji_re, y_ji_im}

          [
            {{from_bus, to_bus}, y_ij},
            {{to_bus, from_bus}, y_ji},
            {{from_bus, from_bus}, y_ii},
            {{to_bus, to_bus}, y_jj}
          ]
        else
          # REGULAR LINE MODEL (no transformer)
          # Off-diagonal: -y_series
          y_off = {-g, -b_series}

          # Diagonal: y_series + y_shunt
          y_diag = {g, b_series + b_shunt}

          [
            {{from_bus, to_bus}, y_off},
            {{to_bus, from_bus}, y_off},
            {{from_bus, from_bus}, y_diag},
            {{to_bus, to_bus}, y_diag}
          ]
        end
      end)
      |> Enum.reduce(%{}, fn {{i, j}, {re, im}}, acc ->
        Map.update(acc, {i, j}, {re, im}, fn {existing_re, existing_im} ->
          {existing_re + re, existing_im + im}
        end)
      end)

    # Convert to CSR format
    # First, group entries by row and sort by column within each row
    entries_by_row =
      entries
      |> Enum.group_by(fn {{i, _j}, _} -> i end, fn {{_i, j}, val} -> {j, val} end)
      |> Enum.map(fn {row, cols} -> {row, Enum.sort_by(cols, fn {j, _} -> j end)} end)
      |> Enum.into(%{})

    # Build CSR arrays
    {row_ptrs, col_indices, values} =
      Enum.reduce(0..(num_buses - 1), {[0], [], []}, fn row, {rp, ci, v} ->
        row_entries = Map.get(entries_by_row, row, [])

        # Extract column indices and values
        row_cols = Enum.map(row_entries, fn {j, _val} -> j end)
        row_vals = Enum.map(row_entries, fn {_j, val} -> val end)

        # Append to arrays
        new_ci = ci ++ row_cols
        new_v = v ++ row_vals
        # row_ptrs contains cumulative counts: next pointer is current length
        new_rp = rp ++ [length(new_ci)]

        {new_rp, new_ci, new_v}
      end)

    %{
      row_ptrs: row_ptrs,
      col_indices: col_indices,
      values: values
    }
  end
end
