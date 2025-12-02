defmodule PowerFlowSolver.SCR do
  @moduledoc """
  Short Circuit Ratio (SCR) calculation module.

  This module provides high-level functions for calculating Short Circuit Ratio,
  a key metric for assessing grid strength at renewable energy interconnection points.

  ## Background

  The Short Circuit Ratio is defined as:

      SCR = S_sc / P_rated

  Where:
  - `S_sc` is the short circuit capacity at the point of interconnection (MVA)
  - `P_rated` is the rated power of the plant (MW)

  The short circuit capacity is calculated from the Thevenin equivalent impedance:

      S_sc = V_base² / |Z_th| = S_base / |Z_th|  (in per-unit system)

  Where `Z_th` is the Thevenin impedance seen looking into the grid from the bus.

  ## Z-bus Matrix

  The Z-bus (impedance matrix) is the inverse of the Y-bus (admittance matrix).
  The diagonal elements `Z_ii` represent the Thevenin impedance at each bus:

      Z_bus = Y_bus⁻¹
      Z_th(bus_i) = Z_ii (diagonal element)

  ## SCR Interpretation

  | SCR Range | Grid Strength | Implications |
  |-----------|--------------|--------------|
  | < 3       | Weak         | Stability concerns, may need grid-forming inverters |
  | 3 - 5     | Moderate     | Some constraints, careful design needed |
  | > 5       | Strong       | Standard grid-following inverters typically OK |
  | > 10      | Very Strong  | Minimal concerns for inverter-based resources |

  ## Usage

      # Calculate SCR for specific plants
      {:ok, results} = PowerFlowSolver.SCR.calculate(system, [
        %{bus_id: 5, p_rated_mw: 100.0},
        %{bus_id: 12, p_rated_mw: 50.0}
      ])

      # Get grid strength at all buses
      {:ok, impedances} = PowerFlowSolver.SCR.get_all_thevenin_impedances(system)

      # Find weakest points in the grid
      weak_buses = PowerFlowSolver.SCR.find_weak_buses(system, threshold_mva: 500.0)
  """

  alias PowerFlowSolver.SparseLinearAlgebra
  alias PowerFlowSolver.NewtonRaphson

  # ============================================================================
  # Type Definitions
  # ============================================================================

  @typedoc """
  Plant/generator data for SCR calculation.

  ## Fields

  - `:bus_id` - Bus ID (as used in the system) where the plant is connected
  - `:p_rated_mw` - Rated power of the plant in MW
  - `:xdpp` - Optional generator subtransient reactance (per-unit on machine base)
  - `:mva_base` - Optional machine MVA base for X''d conversion (required if xdpp provided)
  - `:name` - Optional plant name for reporting
  """
  @type plant :: %{
          required(:bus_id) => non_neg_integer(),
          required(:p_rated_mw) => float(),
          optional(:xdpp) => float() | nil,
          optional(:mva_base) => float() | nil,
          optional(:name) => String.t() | nil
        }

  @typedoc """
  Result of SCR calculation for a single plant.

  ## Fields

  - `:bus_id` - Bus ID where the plant is connected
  - `:z_thevenin_pu` - Thevenin impedance magnitude (per-unit)
  - `:z_thevenin_angle` - Thevenin impedance angle (radians)
  - `:short_circuit_mva` - Short circuit capacity (MVA)
  - `:p_rated_mw` - Plant rated power (MW)
  - `:scr` - Short Circuit Ratio (dimensionless)
  - `:grid_strength` - Qualitative assessment: `:weak`, `:moderate`, `:strong`, or `:very_strong`
  """
  @type scr_result :: %{
          bus_id: non_neg_integer(),
          z_thevenin_pu: float(),
          z_thevenin_angle: float(),
          short_circuit_mva: float(),
          p_rated_mw: float(),
          scr: float(),
          grid_strength: :weak | :moderate | :strong | :very_strong
        }

  @typedoc """
  Thevenin impedance at a bus.

  ## Fields

  - `:bus_id` - Bus ID
  - `:z_real` - Real part of Thevenin impedance (per-unit)
  - `:z_imag` - Imaginary part of Thevenin impedance (per-unit)
  - `:z_magnitude` - Impedance magnitude (per-unit)
  - `:z_angle` - Impedance angle (radians)
  - `:short_circuit_mva` - Short circuit capacity (MVA)
  """
  @type thevenin_impedance :: %{
          bus_id: non_neg_integer(),
          z_real: float(),
          z_imag: float(),
          z_magnitude: float(),
          z_angle: float(),
          short_circuit_mva: float()
        }

  @typedoc """
  Configuration options for SCR calculation.

  ## Fields

  - `:system_mva_base` - System MVA base (default: 100.0)
  - `:include_generator_reactance` - Include generator X''d in model (default: false)
  """
  @type config :: %{
          optional(:system_mva_base) => float(),
          optional(:include_generator_reactance) => boolean()
        }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Calculate SCR for one or more plants.

  This is the main entry point for SCR calculations. It:
  1. Builds the Y-bus from the system if not already built
  2. Inverts Y-bus to get Z-bus
  3. Extracts Thevenin impedances
  4. Calculates SCR for each plant

  ## Arguments

  - `system` - Power system map with `:buses`, `:branches`, etc.
  - `plants` - List of plant maps (see `t:plant/0` type)
  - `opts` - Optional configuration (see `t:config/0` type)

  ## Returns

  - `{:ok, results}` - List of SCR results (see `t:scr_result/0` type)
  - `{:error, reason}` - Error message

  ## Examples

      # Simple usage
      {:ok, results} = PowerFlowSolver.SCR.calculate(system, [
        %{bus_id: 5, p_rated_mw: 100.0}
      ])

      # With generator reactance modeling
      {:ok, results} = PowerFlowSolver.SCR.calculate(system, [
        %{bus_id: 5, p_rated_mw: 100.0, xdpp: 0.2, mva_base: 120.0}
      ], include_generator_reactance: true)

      # Process results
      Enum.each(results, fn result ->
        IO.puts("Bus \#{result.bus_id}: SCR = \#{Float.round(result.scr, 2)} (\#{result.grid_strength})")
      end)
  """
  @spec calculate(map(), [plant()], keyword()) :: {:ok, [scr_result()]} | {:error, String.t()}
  def calculate(system, plants, opts \\ []) do
    system_mva_base = Keyword.get(opts, :system_mva_base, 100.0)
    include_gen_reactance = Keyword.get(opts, :include_generator_reactance, false)

    with {:ok, y_bus_data, bus_id_to_index} <- prepare_y_bus(system),
         {:ok, plants_indexed} <- convert_plants_to_indexed(plants, bus_id_to_index) do
      # Call Rust NIF
      case SparseLinearAlgebra.calculate_scr_batch_rust(
             y_bus_data,
             plants_indexed,
             system_mva_base,
             include_gen_reactance
           ) do
        {:ok, raw_results} ->
          # Convert raw results back to structs with original bus IDs
          index_to_bus_id = Map.new(bus_id_to_index, fn {k, v} -> {v, k} end)

          results =
            raw_results
            |> Enum.map(fn {bus_idx, z_th_pu, z_th_angle, s_sc, p_rated, scr} ->
              %{
                bus_id: Map.get(index_to_bus_id, bus_idx, bus_idx),
                z_thevenin_pu: z_th_pu,
                z_thevenin_angle: z_th_angle,
                short_circuit_mva: s_sc,
                p_rated_mw: p_rated,
                scr: scr,
                grid_strength: classify_grid_strength(scr)
              }
            end)

          {:ok, results}

        {:error, _} ->
          {:error, "SCR calculation failed - check Y-bus matrix validity"}
      end
    end
  end

  @doc """
  Get Thevenin impedances at all buses in the system.

  Useful for understanding grid strength across the entire system before
  specifying plant locations.

  ## Arguments

  - `system` - Power system map
  - `opts` - Optional configuration

  ## Returns

  - `{:ok, impedances}` - List of Thevenin impedances (see `t:thevenin_impedance/0`)
  - `{:error, reason}` - Error message

  ## Example

      {:ok, impedances} = PowerFlowSolver.SCR.get_all_thevenin_impedances(system)

      # Sort by short circuit capacity (weakest first)
      weakest_first = Enum.sort_by(impedances, & &1.short_circuit_mva)

      # Get top 10 weakest buses
      weak_buses = Enum.take(weakest_first, 10)
  """
  @spec get_all_thevenin_impedances(map(), keyword()) ::
          {:ok, [thevenin_impedance()]} | {:error, String.t()}
  def get_all_thevenin_impedances(system, opts \\ []) do
    system_mva_base = Keyword.get(opts, :system_mva_base, 100.0)

    with {:ok, y_bus_data, bus_id_to_index} <- prepare_y_bus(system) do
      case SparseLinearAlgebra.get_thevenin_impedances_rust(y_bus_data, system_mva_base) do
        {:ok, raw_results} ->
          index_to_bus_id = Map.new(bus_id_to_index, fn {k, v} -> {v, k} end)

          impedances =
            raw_results
            |> Enum.map(fn {bus_idx, z_real, z_imag, s_sc} ->
              z_mag = :math.sqrt(z_real * z_real + z_imag * z_imag)
              z_angle = :math.atan2(z_imag, z_real)

              %{
                bus_id: Map.get(index_to_bus_id, bus_idx, bus_idx),
                z_real: z_real,
                z_imag: z_imag,
                z_magnitude: z_mag,
                z_angle: z_angle,
                short_circuit_mva: s_sc
              }
            end)

          {:ok, impedances}

        {:error, _} ->
          {:error, "Failed to calculate Thevenin impedances"}
      end
    end
  end

  @doc """
  Find buses with weak grid strength (low short circuit capacity).

  ## Arguments

  - `system` - Power system map
  - `opts` - Options:
    - `:threshold_mva` - Short circuit capacity threshold (default: 500.0 MVA)
    - `:system_mva_base` - System MVA base (default: 100.0)

  ## Returns

  - `{:ok, weak_buses}` - List of buses below threshold, sorted weakest first
  - `{:error, reason}` - Error message

  ## Example

      {:ok, weak_buses} = PowerFlowSolver.SCR.find_weak_buses(system, threshold_mva: 1000.0)

      IO.puts("Found \#{length(weak_buses)} weak buses:")
      Enum.each(weak_buses, fn bus ->
        IO.puts("  Bus \#{bus.bus_id}: S_sc = \#{Float.round(bus.short_circuit_mva, 1)} MVA")
      end)
  """
  @spec find_weak_buses(map(), keyword()) :: {:ok, [thevenin_impedance()]} | {:error, String.t()}
  def find_weak_buses(system, opts \\ []) do
    threshold_mva = Keyword.get(opts, :threshold_mva, 500.0)

    case get_all_thevenin_impedances(system, opts) do
      {:ok, impedances} ->
        weak_buses =
          impedances
          |> Enum.filter(fn imp -> imp.short_circuit_mva < threshold_mva end)
          |> Enum.sort_by(& &1.short_circuit_mva)

        {:ok, weak_buses}

      error ->
        error
    end
  end

  @doc """
  Calculate SCR at a single bus for a hypothetical plant size.

  Convenience function for quick what-if analysis.

  ## Arguments

  - `system` - Power system map
  - `bus_id` - Bus ID to analyze
  - `p_rated_mw` - Hypothetical plant size in MW
  - `opts` - Optional configuration

  ## Returns

  - `{:ok, result}` - SCR result for the bus
  - `{:error, reason}` - Error message

  ## Example

      {:ok, result} = PowerFlowSolver.SCR.calculate_at_bus(system, 5, 100.0)
      IO.puts("SCR at bus 5 for 100 MW plant: \#{Float.round(result.scr, 2)}")
  """
  @spec calculate_at_bus(map(), non_neg_integer(), float(), keyword()) ::
          {:ok, scr_result()} | {:error, String.t()}
  def calculate_at_bus(system, bus_id, p_rated_mw, opts \\ []) do
    case calculate(system, [%{bus_id: bus_id, p_rated_mw: p_rated_mw}], opts) do
      {:ok, [result]} -> {:ok, result}
      {:ok, []} -> {:error, "Bus #{bus_id} not found in system"}
      error -> error
    end
  end

  @doc """
  Get the full Z-bus (impedance) matrix.

  Returns the complete dense Z-bus matrix. For large systems, this can be
  memory-intensive. Consider using `get_all_thevenin_impedances/2` if you
  only need diagonal elements.

  ## Arguments

  - `system` - Power system map

  ## Returns

  - `{:ok, z_bus}` - Dense Z-bus as map with `:matrix` and `:bus_ids` keys
  - `{:error, reason}` - Error message

  ## Example

      {:ok, z_bus} = PowerFlowSolver.SCR.get_z_bus(system)

      # Access element Z_ij
      row = Enum.at(z_bus.matrix, i)
      {z_real, z_imag} = Enum.at(row, j)
  """
  @spec get_z_bus(map()) :: {:ok, map()} | {:error, String.t()}
  def get_z_bus(system) do
    with {:ok, y_bus_data, bus_id_to_index} <- prepare_y_bus(system) do
      case SparseLinearAlgebra.invert_y_bus_rust(y_bus_data) do
        {:ok, z_matrix} ->
          # Create ordered list of bus IDs
          bus_ids =
            bus_id_to_index
            |> Enum.sort_by(fn {_id, idx} -> idx end)
            |> Enum.map(fn {id, _idx} -> id end)

          {:ok, %{matrix: z_matrix, bus_ids: bus_ids, n: length(bus_ids)}}

        {:error, _} ->
          {:error, "Failed to invert Y-bus matrix"}
      end
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  # Prepare Y-bus data for Rust NIF
  # Returns {y_bus_tuple, bus_id_to_index_map}
  defp prepare_y_bus(system) do
    # Sort buses by ID to create consistent indexing
    sorted_buses = Enum.sort_by(system.buses, & &1.id)

    # Create bus ID to index mapping
    bus_id_to_index =
      sorted_buses
      |> Enum.with_index()
      |> Enum.map(fn {bus, idx} -> {bus.id, idx} end)
      |> Map.new()

    # Build Y-bus using the existing NewtonRaphson function
    system_with_sorted = %{system | buses: sorted_buses}
    y_bus = NewtonRaphson.build_y_bus(system_with_sorted)

    # Convert to tuple format expected by NIF
    y_bus_data = {
      y_bus.row_ptrs,
      y_bus.col_indices,
      y_bus.values
    }

    {:ok, y_bus_data, bus_id_to_index}
  end

  # Convert plant maps to indexed tuples for Rust NIF
  defp convert_plants_to_indexed(plants, bus_id_to_index) do
    plants_indexed =
      plants
      |> Enum.filter(fn plant -> Map.has_key?(bus_id_to_index, plant.bus_id) end)
      |> Enum.map(fn plant ->
        bus_idx = Map.fetch!(bus_id_to_index, plant.bus_id)
        p_rated = Map.fetch!(plant, :p_rated_mw)
        xdpp = Map.get(plant, :xdpp)
        mva_base = Map.get(plant, :mva_base)
        {bus_idx, p_rated, xdpp, mva_base}
      end)

    if Enum.empty?(plants_indexed) and not Enum.empty?(plants) do
      {:error, "No valid plant bus IDs found in system"}
    else
      {:ok, plants_indexed}
    end
  end

  # Classify grid strength based on SCR value
  defp classify_grid_strength(scr) when scr < 3.0, do: :weak
  defp classify_grid_strength(scr) when scr < 5.0, do: :moderate
  defp classify_grid_strength(scr) when scr < 10.0, do: :strong
  defp classify_grid_strength(_scr), do: :very_strong
end
