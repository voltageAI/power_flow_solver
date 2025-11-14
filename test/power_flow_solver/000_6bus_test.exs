defmodule PowerFlowSolver.SixBus000Test do
  use ExUnit.Case
  require Logger

  @test_data_dir Path.join(__DIR__, "../test_data")
  @raw_file_path Path.join(@test_data_dir, "000_6bus_v33.RAW")

  describe "PSS/E RAW 000_6bus solved case" do
    test "solves 6-bus test case with detailed logging" do
      # Parse the RAW file using power_system_parsers
      assert File.exists?(@raw_file_path), "Test file not found: #{@raw_file_path}"

      {:ok, {buses, lines, shunts, metadata}} =
        PowerSystemParsers.PsseRawParser.parse_file(@raw_file_path)

      Logger.info("\n" <> String.duplicate("=", 80))
      Logger.info("PSS/E RAW FILE: 000_6bus_v33.RAW")
      Logger.info(String.duplicate("=", 80))

      # Log case metadata
      Logger.info("\n--- CASE METADATA ---")
      Logger.info("Description: Test 000 6bus with 5 PQ buses 1 Generator PV slack")

      # Log metadata structure (areas, zones, etc.)
      if Map.has_key?(metadata, :areas) and length(metadata.areas) > 0 do
        Logger.info("Areas: #{length(metadata.areas)}")
      end
      if Map.has_key?(metadata, :zones) and length(metadata.zones) > 0 do
        Logger.info("Zones: #{length(metadata.zones)}")
      end

      # Log parsed data summary
      Logger.info("\n--- PARSED DATA SUMMARY ---")
      Logger.info("Buses: #{length(buses)}")
      Logger.info("Lines: #{length(lines)}")
      Logger.info("Fixed Shunts: #{length(shunts.fixed)}")
      Logger.info("Switched Shunts: #{length(shunts.switched)}")

      # Log detailed bus data
      Logger.info("\n--- BUS DATA (from RAW file) ---")

      Enum.each(buses, fn bus ->
        Logger.info(
          "Bus #{bus.id}: " <>
            "type=#{bus.type}, " <>
            "V_initial=#{Float.round(bus.v_magnitude, 5)} pu ∠ #{Float.round(bus.v_angle, 4)}°, " <>
            "P_load=#{bus.p_load} MW, Q_load=#{bus.q_load} Mvar, " <>
            "P_gen=#{bus.p_gen} MW, Q_gen=#{bus.q_gen} Mvar"
        )
      end)

      # Log line data
      Logger.info("\n--- LINE DATA (branches + transformers) ---")

      Enum.each(lines, fn line ->
        tap_info =
          if Map.has_key?(line, :tap_ratio) and line.tap_ratio != 1.0 do
            " [XFMR: tap=#{line.tap_ratio}]"
          else
            " [LINE]"
          end

        Logger.info(
          "Line #{line.id}: #{line.from} → #{line.to}: " <>
            "r=#{line.r} pu, x=#{line.x} pu" <> tap_info
        )
      end)

      # Create system structure for solver
      system = %{buses: buses, lines: lines}

      # Build and log Y-bus
      Logger.info("\n" <> String.duplicate("=", 80))
      Logger.info("Y-BUS CONSTRUCTION")
      Logger.info(String.duplicate("=", 80))

      y_bus = PowerFlowSolver.NewtonRaphson.build_y_bus(system)

      Logger.info("\n--- Y-BUS STRUCTURE ---")
      Logger.info("Format: CSR (Compressed Sparse Row)")
      Logger.info("Dimension: #{length(y_bus.row_ptrs) - 1} x #{length(y_bus.row_ptrs) - 1}")
      Logger.info("Non-zero entries: #{length(y_bus.values)}")

      # Print Y-bus as dense matrix
      Logger.info("\n--- Y-BUS MATRIX (DENSE FORMAT) ---")
      print_dense_y_bus(y_bus)

      # Print Y-bus magnitudes
      Logger.info("\n--- Y-BUS MAGNITUDES ---")
      print_y_bus_magnitudes(y_bus)

      # Solve power flow
      Logger.info("\n" <> String.duplicate("=", 80))
      Logger.info("POWER FLOW SOLUTION")
      Logger.info(String.duplicate("=", 80))

      case PowerFlowSolver.NewtonRaphson.solve(system,
             max_iterations: 100,
             tolerance: 1.0e-6,
             enforce_q_limits: false
           ) do
        {:ok, solution, iterations} ->
          Logger.info("\n✓ POWER FLOW CONVERGED")
          Logger.info("Iterations: #{iterations}")

          Logger.info("\n--- FINAL VOLTAGE SOLUTION ---")

          Enum.each(system.buses, fn bus ->
            {v_mag, v_ang} = Map.get(solution, bus.id)
            v_ang_deg = v_ang * 180.0 / :math.pi()

            # Calculate differences from initial values
            v_diff = v_mag - bus.v_magnitude
            ang_diff = v_ang_deg - bus.v_angle

            v_diff_str = if v_diff >= 0, do: "+#{:io_lib.format("~9.6f", [v_diff])}", else: "#{:io_lib.format("~9.6f", [v_diff])}"
            ang_diff_str = if ang_diff >= 0, do: "+#{:io_lib.format("~8.4f", [ang_diff])}", else: "#{:io_lib.format("~8.4f", [ang_diff])}"

            Logger.info(
              "Bus #{bus.id} (#{bus.type}): " <>
                "V=#{:io_lib.format("~8.6f", [v_mag])} pu ∠ #{:io_lib.format("~8.4f", [v_ang_deg])}° " <>
                "(ΔV=#{v_diff_str}, Δθ=#{ang_diff_str}°)"
            )
          end)

          # Test assertions
          Logger.info("\n--- SOLUTION VALIDATION ---")
          assert iterations <= 10, "Solution should converge in 10 iterations or less"
          Logger.info("✓ Converged in #{iterations} iterations (≤ 10)")

          # Check that voltage magnitudes are within reasonable bounds
          Enum.each(system.buses, fn bus ->
            {v_mag, _v_ang} = Map.get(solution, bus.id)
            assert v_mag >= 0.9 and v_mag <= 1.1, "Bus #{bus.id} voltage out of bounds"
          end)

          Logger.info("✓ All bus voltages within bounds (0.9 - 1.1 pu)")

          Logger.info("\n" <> String.duplicate("=", 80))
          Logger.info("TEST COMPLETE")
          Logger.info(String.duplicate("=", 80))

        {:error, reason} ->
          Logger.error("\n✗ POWER FLOW FAILED")
          Logger.error("Reason: #{reason}")
          flunk("Power flow failed to converge: #{reason}")
      end
    end
  end

  # Helper function to print Y-bus as dense matrix
  defp print_dense_y_bus(y_bus) do
    n = length(y_bus.row_ptrs) - 1

    # Convert CSR to dense matrix
    dense =
      for i <- 0..(n - 1) do
        for j <- 0..(n - 1) do
          # Find value at position (i,j)
          row_start = Enum.at(y_bus.row_ptrs, i)
          row_end = Enum.at(y_bus.row_ptrs, i + 1)

          # Search for column j in this row's entries
          col_indices_slice = Enum.slice(y_bus.col_indices, row_start..(row_end - 1))
          values_slice = Enum.slice(y_bus.values, row_start..(row_end - 1))

          case Enum.find_index(col_indices_slice, &(&1 == j)) do
            # Not found, entry is zero
            nil -> {0.0, 0.0}
            idx -> Enum.at(values_slice, idx)
          end
        end
      end

    # Print header
    header = "  Bus |" <> Enum.map_join(0..(n - 1), " | ", fn j -> " Bus #{j} " end) <> " |"
    Logger.info(header)
    Logger.info("  " <> String.duplicate("-", String.length(header) - 2))

    # Print each row
    Enum.with_index(dense)
    |> Enum.each(fn {row, i} ->
      row_str =
        Enum.map_join(row, " | ", fn {re, im} ->
          format_complex(re, im)
        end)

      Logger.info("  #{i}   | #{row_str} |")
    end)
  end

  # Helper function to print Y-bus magnitudes
  defp print_y_bus_magnitudes(y_bus) do
    n = length(y_bus.row_ptrs) - 1

    # Convert CSR to dense matrix
    dense =
      for i <- 0..(n - 1) do
        for j <- 0..(n - 1) do
          row_start = Enum.at(y_bus.row_ptrs, i)
          row_end = Enum.at(y_bus.row_ptrs, i + 1)

          col_indices_slice = Enum.slice(y_bus.col_indices, row_start..(row_end - 1))
          values_slice = Enum.slice(y_bus.values, row_start..(row_end - 1))

          case Enum.find_index(col_indices_slice, &(&1 == j)) do
            nil -> {0.0, 0.0}
            idx -> Enum.at(values_slice, idx)
          end
        end
      end

    # Print header
    header = "  Bus |" <> Enum.map_join(0..(n - 1), " | ", fn j -> " Bus #{j} " end) <> " |"
    Logger.info(header)
    Logger.info("  " <> String.duplicate("-", String.length(header) - 2))

    # Print magnitudes
    Enum.with_index(dense)
    |> Enum.each(fn {row, i} ->
      row_str =
        Enum.map_join(row, " | ", fn {re, im} ->
          mag = :math.sqrt(re * re + im * im)
          :io_lib.format("~8.2f", [mag]) |> to_string() |> String.pad_leading(8)
        end)

      Logger.info("  #{i}   | #{row_str} |")
    end)
  end

  # Format complex number for display
  defp format_complex(re, im) do
    re_str = :io_lib.format("~6.2f", [re]) |> to_string() |> String.pad_leading(6)

    if im >= 0 do
      im_str = :io_lib.format("~6.2f", [im]) |> to_string() |> String.pad_leading(6)
      "#{re_str}+j#{im_str}"
    else
      im_str = :io_lib.format("~6.2f", [-im]) |> to_string() |> String.pad_leading(6)
      "#{re_str}-j#{im_str}"
    end
  end
end
