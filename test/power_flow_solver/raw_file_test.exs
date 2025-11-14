defmodule PowerFlowSolver.RawFileTest do
  use ExUnit.Case
  require Logger

  @test_data_dir Path.join(__DIR__, "../test_data")
  @raw_file_path Path.join(@test_data_dir, "0_6bus_start_simple_v33.raw")

  describe "PSS/E RAW file parsing" do
    test "reads and parses 6-bus test case" do
      # Parse the RAW file using power_system_parsers
      assert File.exists?(@raw_file_path), "Test file not found: #{@raw_file_path}"

      {:ok, {buses, lines, shunts, _metadata}} =
        PowerSystemParsers.PsseRawParser.parse_file(@raw_file_path)

      # Log the parsed data structure
      Logger.info("Parsed RAW file data:")
      Logger.info("  Buses: #{length(buses)}")
      Logger.info("  Lines: #{length(lines)}")
      Logger.info("  Fixed Shunts: #{length(shunts.fixed)}")
      Logger.info("  Switched Shunts: #{length(shunts.switched)}")

      # Log raw bus data to see what the parser returned
      Logger.info("\nRaw parsed bus data:")
      Enum.each(buses, fn bus ->
        Logger.info("  Bus #{bus.id}: type=#{bus.type}, p_load=#{bus.p_load}, q_load=#{bus.q_load}, " <>
          "p_gen=#{bus.p_gen}, q_gen=#{bus.q_gen}, v_mag=#{bus.v_magnitude}")
      end)

      # Log raw line data
      Logger.info("\nRaw parsed line data:")
      Enum.take(lines, 3) |> Enum.each(fn line ->
        Logger.info("  Line #{line.from} -> #{line.to}: r=#{line.r}, x=#{line.x}, " <>
          "tap_ratio=#{Map.get(line, :tap_ratio, "N/A")}")
      end)

      # Use parser output directly - no conversion needed
      # Parser returns buses with all required fields and lines (combined branches + transformers)
      system = %{buses: buses, lines: lines}

      # Log the system structure that will be passed to solve
      Logger.info("\nSystem structure for solver:")
      Logger.info("Number of buses: #{length(system.buses)}")

      Logger.info("\nBuses:")
      Enum.each(system.buses, fn bus ->
        Logger.info("  Bus #{bus.id}: type=#{bus.type}, " <>
          "p_load=#{bus.p_load}, q_load=#{bus.q_load}, " <>
          "p_gen=#{bus.p_gen}, q_gen=#{bus.q_gen}, " <>
          "v_mag=#{bus.v_magnitude}, v_angle=#{bus.v_angle}")
      end)

      Logger.info("\nLines (branches + transformers): #{length(system.lines)}")
      Enum.each(system.lines, fn line ->
        tap_info = if Map.has_key?(line, :tap_ratio) and line.tap_ratio != 1.0 do
          " (transformer, tap=#{line.tap_ratio})"
        else
          ""
        end
        Logger.info("  Line #{line.from} -> #{line.to}: " <>
          "r=#{line.r}, x=#{line.x}#{tap_info}")
      end)

      Logger.info("\nY-bus structure:")
      if Map.has_key?(system, :y_bus) do
        y_bus = system.y_bus
        Logger.info("  Format: CSR (Compressed Sparse Row)")
        Logger.info("  Dimension: #{length(y_bus.row_ptrs) - 1} x #{length(y_bus.row_ptrs) - 1}")
        Logger.info("  Non-zero entries: #{length(y_bus.values)}")
        Logger.info("  Row pointers: #{inspect(y_bus.row_ptrs)}")
        Logger.info("  Column indices: #{inspect(y_bus.col_indices)}")
        Logger.info("  Values (first 10): #{inspect(Enum.take(y_bus.values, 10))}")
      else
        Logger.info("  Y-bus not pre-built (will be constructed from branches/transformers)")

        # Build y_bus to see what it looks like
        Logger.info("\nBuilding Y-bus from branches/transformers...")
        Logger.info("Running power flow solver to convergence...")

        # Call the solver which will build the y_bus internally
        Logger.info("\n--- Test 1: Without Q-limit enforcement ---")
        case PowerFlowSolver.NewtonRaphson.solve(system, max_iterations: 100, tolerance: 1.0e-6, enforce_q_limits: false) do
          {:ok, solution, iterations} ->
            Logger.info("\n=== POWER FLOW CONVERGED ===")
            Logger.info("Iterations: #{iterations}")
            Logger.info("\nFinal voltage solution (no Q-limits):")
            Enum.each(system.buses, fn bus ->
              {v_mag, v_ang} = Map.get(solution, bus.id)
              v_ang_deg = v_ang * 180.0 / :math.pi()
              Logger.info("  Bus #{bus.id}: #{Float.round(v_mag, 6)} pu ∠ #{Float.round(v_ang_deg, 4)}°")
            end)
          {:error, reason} ->
            Logger.info("\n=== POWER FLOW FAILED ===")
            Logger.info("  Error: #{reason}")
        end

        # Now try with Q-limits enforced
        Logger.info("\n--- Test 2: With Q-limit enforcement ---")
        case PowerFlowSolver.NewtonRaphson.solve(system, max_iterations: 100, tolerance: 1.0e-6, enforce_q_limits: true) do
          {:ok, solution, iterations} ->
            Logger.info("\n=== POWER FLOW CONVERGED (Q-limits enforced) ===")
            Logger.info("Iterations: #{iterations}")
            Logger.info("\nFinal voltage solution (with Q-limits):")
            Enum.each(system.buses, fn bus ->
              {v_mag, v_ang} = Map.get(solution, bus.id)
              v_ang_deg = v_ang * 180.0 / :math.pi()
              Logger.info("  Bus #{bus.id}: #{Float.round(v_mag, 6)} pu ∠ #{Float.round(v_ang_deg, 4)}°")
            end)
          {:error, reason} ->
            Logger.info("\n=== POWER FLOW FAILED (Q-limits) ===")
            Logger.info("  Error: #{reason}")
        end

        # Test 3: Convert Bus 1 (PV) to PQ and re-solve
        Logger.info("\n--- Test 3: Convert Bus 1 from PV to PQ ---")
        system_pq = %{system | buses: Enum.map(system.buses, fn bus ->
          if bus.id == 1 do
            # Convert Bus 1 from PV to PQ
            Logger.info("Converting Bus #{bus.id} from #{bus.type} to :pq")
            %{bus | type: :pq}
          else
            bus
          end
        end)}

        Logger.info("Modified bus types:")
        Enum.each(system_pq.buses, fn bus ->
          Logger.info("  Bus #{bus.id}: type=#{bus.type}")
        end)

        case PowerFlowSolver.NewtonRaphson.solve(system_pq, max_iterations: 100, tolerance: 1.0e-6, enforce_q_limits: false) do
          {:ok, solution, iterations} ->
            Logger.info("\n=== POWER FLOW CONVERGED (Bus 1 as PQ) ===")
            Logger.info("Iterations: #{iterations}")
            Logger.info("\nFinal voltage solution (Bus 1 as PQ):")
            Enum.each(system_pq.buses, fn bus ->
              {v_mag, v_ang} = Map.get(solution, bus.id)
              v_ang_deg = v_ang * 180.0 / :math.pi()
              Logger.info("  Bus #{bus.id}: #{Float.round(v_mag, 6)} pu ∠ #{Float.round(v_ang_deg, 4)}°")
            end)

            Logger.info("\n=== COMPARISON WITH POWERWORLD ===")
            Logger.info("PowerWorld results:")
            Logger.info("  Bus 1 (SLACK): 1.000000 pu ∠ 0.0306°")
            Logger.info("  Bus 2 (PQ):    0.986581 pu ∠ 3.3589°")
            Logger.info("  Bus 3 (PQ):    0.998545 pu ∠ -1.6962°")
            Logger.info("  Bus 4 (PQ):    0.979093 pu ∠ -1.5003°")
            Logger.info("  Bus 5 (PQ):    1.005778 pu ∠ -7.1437°")
            Logger.info("  Bus 6 (PQ):    0.994052 pu ∠ 1.4542°")
          {:error, reason} ->
            Logger.info("\n=== POWER FLOW FAILED (Bus 1 as PQ) ===")
            Logger.info("  Error: #{reason}")
        end

        # Build Y-bus using the public function from NewtonRaphson module
        y_bus = PowerFlowSolver.NewtonRaphson.build_y_bus(system)

        Logger.info("\nBuilt Y-bus structure:")
        Logger.info("  Format: CSR (Compressed Sparse Row)")
        Logger.info("  Dimension: #{length(y_bus.row_ptrs) - 1} x #{length(y_bus.row_ptrs) - 1}")
        Logger.info("  Non-zero entries: #{length(y_bus.values)}")
        Logger.info("  Row pointers: #{inspect(y_bus.row_ptrs)}")
        Logger.info("  Column indices: #{inspect(y_bus.col_indices)}")
        Logger.info("  Values: #{inspect(y_bus.values)}")

        # Print as dense matrix for visualization
        Logger.info("\nY-bus as Dense Matrix:")
        print_dense_y_bus(y_bus)
      end
    end
  end

  # Convert CSR sparse matrix to dense format and print
  defp print_dense_y_bus(y_bus) do
    n = length(y_bus.row_ptrs) - 1

    # Convert CSR to dense matrix
    dense =
      for i <- 0..(n-1) do
        for j <- 0..(n-1) do
          # Find value at position (i,j)
          row_start = Enum.at(y_bus.row_ptrs, i)
          row_end = Enum.at(y_bus.row_ptrs, i + 1)

          # Search for column j in this row's entries
          col_indices_slice = Enum.slice(y_bus.col_indices, row_start..(row_end - 1))
          values_slice = Enum.slice(y_bus.values, row_start..(row_end - 1))

          case Enum.find_index(col_indices_slice, &(&1 == j)) do
            nil -> {0.0, 0.0}  # Not found, entry is zero
            idx -> Enum.at(values_slice, idx)
          end
        end
      end

    # Print header
    Logger.info("  Bus |" <> Enum.map_join(0..(n-1), " | ", fn j -> " Bus #{j} " end) <> " |")
    Logger.info("  " <> String.duplicate("-", 11 + n * 25))

    # Print each row
    Enum.with_index(dense) |> Enum.each(fn {row, i} ->
      row_str = Enum.map_join(row, " | ", fn {re, im} ->
        format_complex(re, im)
      end)
      Logger.info("  #{i}   | #{row_str} |")
    end)

    # Print magnitude view
    Logger.info("\nY-bus Magnitudes:")
    Logger.info("  Bus |" <> Enum.map_join(0..(n-1), " | ", fn j -> " Bus #{j} " end) <> " |")
    Logger.info("  " <> String.duplicate("-", 11 + n * 12))

    Enum.with_index(dense) |> Enum.each(fn {row, i} ->
      row_str = Enum.map_join(row, " | ", fn {re, im} ->
        mag = :math.sqrt(re * re + im * im)
        format_float(mag, 8)
      end)
      Logger.info("  #{i}   | #{row_str} |")
    end)
  end

  defp format_complex(re, im) do
    re_str = format_float(re, 6)

    if im >= 0 do
      im_str = format_float(im, 6)
      "#{re_str}+j#{im_str}"
    else
      im_str = format_float(-im, 6)
      "#{re_str}-j#{im_str}"
    end
  end

  defp format_float(val, width) do
    str = :io_lib.format("~#{width}.2f", [val]) |> to_string()
    String.pad_leading(str, width + 3)
  end
end
