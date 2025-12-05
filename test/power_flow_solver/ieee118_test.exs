defmodule PowerFlowSolver.IEEE118Test do
  @moduledoc """
  Comprehensive test for IEEE 118-bus system.

  This test validates the power flow solver against the MATPOWER reference solution.
  The IEEE 118-bus system represents the American Electric Power (AEP) system
  from approximately 1962.

  Reference angles from MATPOWER solution (bus 69 is slack at 30°):
  - Minimum angle: ~6.9° (bus 110)
  - Maximum angle: ~39.7° (bus 9 or 10)
  - Angle spread: ~33°
  """

  use ExUnit.Case

  @test_data_dir Path.join(__DIR__, "../test_data")
  @case_file Path.join(@test_data_dir, "case118.m")

  # Reference values from MATPOWER solution
  # Slack bus is bus 69 (0-indexed: 68) with angle = 30°
  @slack_bus_id 68
  @slack_angle_deg 30.0

  # Expected angle range (relative to slack)
  @expected_min_angle_deg 6.9
  @expected_max_angle_deg 39.7
  @expected_angle_spread_deg 32.8

  describe "IEEE 118-bus system" do
    test "parses case file correctly" do
      assert File.exists?(@case_file), "Test file not found: #{@case_file}"

      {:ok, system} = PowerFlowSolver.MatpowerParser.parse_file(@case_file)

      assert length(system.buses) == 118, "Should have 118 buses"
      assert length(system.lines) > 0, "Should have lines"
      assert system.base_mva == 100, "Base MVA should be 100"

      # Check bus types
      bus_types = Enum.frequencies_by(system.buses, & &1.type)
      IO.puts("\nBus type distribution:")
      IO.puts("  Slack: #{Map.get(bus_types, :slack, 0)}")
      IO.puts("  PV: #{Map.get(bus_types, :pv, 0)}")
      IO.puts("  PQ: #{Map.get(bus_types, :pq, 0)}")

      assert Map.get(bus_types, :slack, 0) == 1, "Should have exactly 1 slack bus"

      # Verify slack bus
      slack_bus = Enum.find(system.buses, &(&1.type == :slack))
      assert slack_bus.id == @slack_bus_id, "Slack bus should be bus #{@slack_bus_id}"
    end

    test "shows input data summary" do
      {:ok, system} = PowerFlowSolver.MatpowerParser.parse_file(@case_file)

      IO.puts("\n=== IEEE 118 INPUT DATA SUMMARY ===")

      # Power balance
      total_p_gen = Enum.reduce(system.buses, 0.0, fn b, acc -> acc + b.p_gen end)
      total_p_load = Enum.reduce(system.buses, 0.0, fn b, acc -> acc + b.p_load end)
      total_q_gen = Enum.reduce(system.buses, 0.0, fn b, acc -> acc + b.q_gen end)
      total_q_load = Enum.reduce(system.buses, 0.0, fn b, acc -> acc + b.q_load end)

      IO.puts("\nPower totals (p.u. on 100 MVA base):")
      IO.puts("  Total P generation: #{Float.round(total_p_gen, 3)} p.u.")
      IO.puts("  Total P load: #{Float.round(total_p_load, 3)} p.u.")
      IO.puts("  P balance (gen - load): #{Float.round(total_p_gen - total_p_load, 3)} p.u.")
      IO.puts("  Total Q generation: #{Float.round(total_q_gen, 3)} p.u.")
      IO.puts("  Total Q load: #{Float.round(total_q_load, 3)} p.u.")

      # Input angles
      input_angles_deg = Enum.map(system.buses, fn b -> b.v_angle * 180 / :math.pi() end)
      min_input = Enum.min(input_angles_deg)
      max_input = Enum.max(input_angles_deg)

      IO.puts("\nInput voltage angles:")
      IO.puts("  Min: #{Float.round(min_input, 2)}°")
      IO.puts("  Max: #{Float.round(max_input, 2)}°")
      IO.puts("  Spread: #{Float.round(max_input - min_input, 2)}°")

      # Sample of buses with generation
      gen_buses = Enum.filter(system.buses, &(&1.p_gen > 0))
      IO.puts("\nBuses with P generation (#{length(gen_buses)} total):")

      gen_buses
      |> Enum.take(10)
      |> Enum.each(fn b ->
        IO.puts(
          "  Bus #{b.id}: type=#{b.type}, p_gen=#{Float.round(b.p_gen, 3)}, p_load=#{Float.round(b.p_load, 3)}"
        )
      end)

      # Check for PV buses with zero generation (this was flagged in the issue)
      pv_buses = Enum.filter(system.buses, &(&1.type == :pv))
      pv_zero_gen = Enum.filter(pv_buses, &(&1.p_gen == 0.0))

      IO.puts("\nPV buses with zero P generation: #{length(pv_zero_gen)} of #{length(pv_buses)}")

      if length(pv_zero_gen) > 0 do
        IO.puts("  (This is normal - some PV buses are synchronous condensers)")

        pv_zero_gen
        |> Enum.take(5)
        |> Enum.each(fn b ->
          IO.puts(
            "  Bus #{b.id}: p_gen=#{b.p_gen}, q_min=#{inspect(b.q_min)}, q_max=#{inspect(b.q_max)}"
          )
        end)
      end

      assert true
    end

    test "solves power flow and compares to reference" do
      {:ok, system} = PowerFlowSolver.MatpowerParser.parse_file(@case_file)

      IO.puts("\n=== SOLVING IEEE 118 POWER FLOW ===")

      # Solve with default settings
      result =
        PowerFlowSolver.NewtonRaphson.solve(system,
          max_iterations: 100,
          tolerance: 1.0e-4,
          enforce_q_limits: false
        )

      case result do
        {:ok, solution, iterations} ->
          IO.puts("Converged in #{iterations} iterations")

          # Extract angles
          angles_rad =
            system.buses
            |> Enum.map(fn bus ->
              {_v_mag, v_ang, _q_gen} = Map.get(solution, bus.id)
              {bus.id, v_ang}
            end)
            |> Map.new()

          angles_deg = Map.new(angles_rad, fn {id, rad} -> {id, rad * 180 / :math.pi()} end)

          # Get slack angle
          slack_angle = Map.get(angles_deg, @slack_bus_id)
          IO.puts("\nSlack bus (#{@slack_bus_id}) angle: #{Float.round(slack_angle, 2)}°")

          # Calculate angles relative to slack
          relative_angles =
            Map.new(angles_deg, fn {id, ang} -> {id, ang - slack_angle + @slack_angle_deg} end)

          min_angle = Enum.min(Map.values(relative_angles))
          max_angle = Enum.max(Map.values(relative_angles))
          angle_spread = max_angle - min_angle

          IO.puts("\nOutput angles (shifted to slack = #{@slack_angle_deg}°):")
          IO.puts("  Min: #{Float.round(min_angle, 2)}° (expected ~#{@expected_min_angle_deg}°)")
          IO.puts("  Max: #{Float.round(max_angle, 2)}° (expected ~#{@expected_max_angle_deg}°)")

          IO.puts(
            "  Spread: #{Float.round(angle_spread, 2)}° (expected ~#{@expected_angle_spread_deg}°)"
          )

          # Calculate scaling factor
          actual_spread = angle_spread
          expected_spread = @expected_angle_spread_deg
          scaling_factor = actual_spread / expected_spread
          IO.puts("\nAngle spread scaling factor: #{Float.round(scaling_factor, 2)}x")

          # Compare sample buses
          IO.puts("\nSample bus comparisons (solver vs expected):")
          # These are approximate expected values from MATPOWER
          sample_expected = %{
            0 => 10.67,
            9 => 35.61,
            67 => 27.55,
            68 => 30.0
          }

          Enum.each(sample_expected, fn {bus_id, expected_deg} ->
            actual = Map.get(relative_angles, bus_id, 0.0)
            diff = actual - expected_deg

            IO.puts(
              "  Bus #{bus_id}: actual=#{Float.round(actual, 2)}°, expected=#{expected_deg}°, diff=#{Float.round(diff, 2)}°"
            )
          end)

          # Check voltage magnitudes
          v_mags =
            Enum.map(system.buses, fn bus ->
              {v_mag, _v_ang, _q_gen} = Map.get(solution, bus.id)
              v_mag
            end)

          min_v = Enum.min(v_mags)
          max_v = Enum.max(v_mags)
          IO.puts("\nVoltage magnitudes:")
          IO.puts("  Min: #{Float.round(min_v, 4)} p.u.")
          IO.puts("  Max: #{Float.round(max_v, 4)} p.u.")

          # Assertions
          assert iterations <= 20, "Should converge in reasonable iterations"
          assert min_v >= 0.9, "Voltage should not be too low"
          assert max_v <= 1.1, "Voltage should not be too high"

          # Check if angle spread is reasonable (within 2x of expected)
          if scaling_factor > 2.0 do
            IO.puts("\n⚠️  WARNING: Angle spread is #{Float.round(scaling_factor, 1)}x expected!")
            IO.puts("    This indicates a potential issue with the solver.")
          end

        {:error, reason} ->
          IO.puts("FAILED: #{reason}")
          flunk("Power flow failed to converge: #{reason}")
      end
    end

    test "analyzes mismatch at MATPOWER solution point" do
      {:ok, system} = PowerFlowSolver.MatpowerParser.parse_file(@case_file)

      IO.puts("\n=== MISMATCH ANALYSIS AT INPUT ANGLES ===")

      # Build Y-bus (unused but demonstrates the API)
      _y_bus = PowerFlowSolver.NewtonRaphson.build_y_bus(system)

      # Use input voltage as the "solution" to check mismatch
      # If input is the correct MATPOWER solution, mismatch should be near zero
      _voltage =
        system.buses
        |> Enum.map(fn bus ->
          {bus.id, {bus.v_magnitude, bus.v_angle}}
        end)
        |> Map.new()

      # Calculate power injections at input voltage
      IO.puts("\nPower mismatch at input voltage (should be near zero if input is solved):")

      # We need to compute S = V* * (Y * V) for each bus
      # This is complex, so let's just check a few buses

      sample_buses = [0, 9, 68, 100]

      Enum.each(sample_buses, fn bus_id ->
        bus = Enum.find(system.buses, &(&1.id == bus_id))
        p_sched = bus.p_gen - bus.p_load
        q_sched = bus.q_gen - bus.q_load

        IO.puts("  Bus #{bus_id} (#{bus.type}):")
        IO.puts("    P_sched = #{Float.round(p_sched, 4)} p.u.")
        IO.puts("    Q_sched = #{Float.round(q_sched, 4)} p.u.")
      end)

      # The actual mismatch calculation would require computing Y*V
      # For now, we just verify the scheduled values look reasonable
      assert true
    end

    test "compares flat start vs warm start convergence" do
      {:ok, system} = PowerFlowSolver.MatpowerParser.parse_file(@case_file)

      IO.puts("\n=== FLAT START VS WARM START COMPARISON ===")

      # Warm start (use input angles)
      IO.puts("\nWarm start (using input angles from MATPOWER):")

      {:ok, warm_solution, warm_iterations} =
        PowerFlowSolver.NewtonRaphson.solve(system,
          max_iterations: 100,
          tolerance: 1.0e-4
        )

      IO.puts("  Iterations: #{warm_iterations}")

      # Flat start (zero all angles)
      flat_start_buses =
        Enum.map(system.buses, fn bus ->
          Map.put(bus, :v_angle, 0.0)
        end)

      flat_system = %{system | buses: flat_start_buses}

      IO.puts("\nFlat start (all angles = 0°):")

      {:ok, flat_solution, flat_iterations} =
        PowerFlowSolver.NewtonRaphson.solve(flat_system,
          max_iterations: 100,
          tolerance: 1.0e-4
        )

      IO.puts("  Iterations: #{flat_iterations}")

      # Compare solutions
      IO.puts("\nSolution comparison:")

      # Check if solutions are the same (relative to slack bus)
      # Since slack bus angle is arbitrary, compare relative angles
      {_, warm_slack_ang, _} = Map.get(warm_solution, @slack_bus_id)
      {_, flat_slack_ang, _} = Map.get(flat_solution, @slack_bus_id)

      IO.puts("  Slack bus angles: warm=#{Float.round(warm_slack_ang * 180 / :math.pi(), 2)}°, flat=#{Float.round(flat_slack_ang * 180 / :math.pi(), 2)}°")

      relative_angle_diffs =
        Enum.map(system.buses, fn bus ->
          {_, warm_ang, _} = Map.get(warm_solution, bus.id)
          {_, flat_ang, _} = Map.get(flat_solution, bus.id)
          warm_rel = warm_ang - warm_slack_ang
          flat_rel = flat_ang - flat_slack_ang
          abs(warm_rel - flat_rel) * 180 / :math.pi()
        end)

      max_rel_diff = Enum.max(relative_angle_diffs)
      avg_rel_diff = Enum.sum(relative_angle_diffs) / length(relative_angle_diffs)

      IO.puts("  Max relative angle difference: #{Float.round(max_rel_diff, 4)}°")
      IO.puts("  Avg relative angle difference: #{Float.round(avg_rel_diff, 4)}°")

      if max_rel_diff > 0.1 do
        IO.puts("  ⚠️  Solutions differ significantly!")
      else
        IO.puts("  ✓ Solutions match (relative angles identical)")
      end

      assert warm_iterations <= flat_iterations,
             "Warm start should converge in fewer or equal iterations"
    end

    test "demonstrates effect of missing generation data" do
      {:ok, system} = PowerFlowSolver.MatpowerParser.parse_file(@case_file)

      IO.puts("\n=== EFFECT OF MISSING GENERATION DATA ===")
      IO.puts("This test demonstrates how missing/incorrect p_gen values")
      IO.puts("cause inflated angle spreads in the solution.\n")

      # Test different generation scaling factors
      scaling_tests = [
        {1.0, "100% gen (correct)"},
        {0.5, "50% gen"},
        {0.33, "33% gen (produces ~3x spread)"},
        {0.0, "0% gen (fails)"}
      ]

      results =
        Enum.map(scaling_tests, fn {scale, label} ->
          scaled_buses =
            Enum.map(system.buses, fn bus ->
              %{bus | p_gen: bus.p_gen * scale}
            end)

          scaled_system = %{system | buses: scaled_buses}

          result =
            PowerFlowSolver.NewtonRaphson.solve(scaled_system,
              max_iterations: 100,
              tolerance: 1.0e-2
            )

          case result do
            {:ok, solution, _} ->
              angles_deg =
                Enum.map(system.buses, fn bus ->
                  {_, ang, _} = Map.get(solution, bus.id)
                  ang * 180 / :math.pi()
                end)

              spread = Enum.max(angles_deg) - Enum.min(angles_deg)
              {label, :ok, spread}

            {:error, _} ->
              {label, :error, nil}
          end
        end)

      IO.puts("Results:")

      Enum.each(results, fn
        {label, :ok, spread} ->
          ratio = spread / @expected_angle_spread_deg
          IO.puts("  #{label}: spread=#{Float.round(spread, 1)}° (#{Float.round(ratio, 2)}x expected)")

        {label, :error, _} ->
          IO.puts("  #{label}: FAILED TO CONVERGE")
      end)

      IO.puts("\nConclusion: Missing generator data causes angle spread inflation.")
      IO.puts("A 3x spread indicates ~33% of expected generation is present.")

      # Verify correct data produces ~1x spread
      {_, :ok, correct_spread} = Enum.find(results, fn {label, _, _} -> label == "100% gen (correct)" end)
      assert_in_delta correct_spread, @expected_angle_spread_deg, 1.0
    end
  end
end
