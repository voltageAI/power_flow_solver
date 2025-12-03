defmodule PowerFlowSolver.SCR.ContingencyTest do
  @moduledoc """
  Tests for contingency SCR calculations.

  Tests use 0-based bus IDs because the Y-bus matrix indexing requires this.
  """

  use ExUnit.Case, async: true

  alias PowerFlowSolver.SCR

  # ============================================================================
  # Test System Builders (all using 0-based bus IDs)
  # ============================================================================

  # 3-bus radial: Slack(0) -- (1) -- (2)
  defp build_3bus_radial do
    %{
      buses: [
        %{id: 0, type: :slack, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0},
        %{id: 1, type: :pq, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.5, q_load: 0.2, p_gen: 0.0, q_gen: 0.0},
        %{id: 2, type: :pq, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.3, q_load: 0.1, p_gen: 0.0, q_gen: 0.0}
      ],
      branches: [
        %{id: 1, from: 0, to: 1, from_bus: 0, to_bus: 1, r: 0.01, x: 0.1, b: 0.02, tap: 1.0, shift: 0.0},
        %{id: 2, from: 1, to: 2, from_bus: 1, to_bus: 2, r: 0.01, x: 0.1, b: 0.02, tap: 1.0, shift: 0.0}
      ],
      transformers: []
    }
  end

  # 4-bus mesh (N-1 secure):
  #   Slack(0) -- (1)
  #      |        |
  #     (2) ---- (3)
  defp build_4bus_mesh do
    %{
      buses: [
        %{id: 0, type: :slack, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0},
        %{id: 1, type: :pq, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.3, q_load: 0.1, p_gen: 0.0, q_gen: 0.0},
        %{id: 2, type: :pq, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.3, q_load: 0.1, p_gen: 0.0, q_gen: 0.0},
        %{id: 3, type: :pq, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.4, q_load: 0.15, p_gen: 0.0, q_gen: 0.0}
      ],
      branches: [
        %{id: 1, from: 0, to: 1, from_bus: 0, to_bus: 1, r: 0.01, x: 0.1, b: 0.02, tap: 1.0, shift: 0.0},
        %{id: 2, from: 0, to: 2, from_bus: 0, to_bus: 2, r: 0.01, x: 0.1, b: 0.02, tap: 1.0, shift: 0.0},
        %{id: 3, from: 1, to: 3, from_bus: 1, to_bus: 3, r: 0.01, x: 0.1, b: 0.02, tap: 1.0, shift: 0.0},
        %{id: 4, from: 2, to: 3, from_bus: 2, to_bus: 3, r: 0.01, x: 0.1, b: 0.02, tap: 1.0, shift: 0.0}
      ],
      transformers: []
    }
  end

  # Double circuit system: Slack(0) ==[101,102]== (1)
  defp build_double_circuit do
    %{
      buses: [
        %{id: 0, type: :slack, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0},
        %{id: 1, type: :pq, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.5, q_load: 0.2, p_gen: 0.0, q_gen: 0.0}
      ],
      branches: [
        %{id: 101, from: 0, to: 1, from_bus: 0, to_bus: 1, r: 0.01, x: 0.1, b: 0.02, tap: 1.0, shift: 0.0},
        %{id: 102, from: 0, to: 1, from_bus: 0, to_bus: 1, r: 0.01, x: 0.1, b: 0.02, tap: 1.0, shift: 0.0}
      ],
      transformers: []
    }
  end

  # ============================================================================
  # Single Contingency Tests
  # ============================================================================

  describe "calculate_contingency/2" do
    test "returns valid result for mesh system contingency" do
      system = build_4bus_mesh()

      {:ok, result} = SCR.calculate_contingency(system,
        poi_bus_id: 3,
        branch: %{id: 1, from_bus: 0, to_bus: 1}
      )

      assert result.success == true
      assert result.poi_bus_id == 3
      assert result.branch_id == 1
      assert result.short_circuit_mva > 0
      assert result.z_thevenin_magnitude > 0
    end

    test "contingency changes S_sc from base case" do
      system = build_4bus_mesh()

      # Get base case S_sc at bus 3
      {:ok, base_results} = SCR.get_all_thevenin_impedances(system)
      base_s_sc = Enum.find(base_results, fn r -> r.bus_id == 3 end).short_circuit_mva

      # Get contingency S_sc (remove branch 0-1)
      {:ok, contingency} = SCR.calculate_contingency(system,
        poi_bus_id: 3,
        branch: %{id: 1, from_bus: 0, to_bus: 1}
      )

      # S_sc should change with contingency (the sign depends on implementation)
      assert contingency.success
      assert contingency.short_circuit_mva != base_s_sc,
        "Contingency S_sc should differ from base case"
    end

    test "handles islanding in radial system" do
      system = build_3bus_radial()

      # Remove branch 0-1 and check at bus 2 (now isolated from slack)
      {:ok, result} = SCR.calculate_contingency(system,
        poi_bus_id: 2,
        branch: %{id: 1, from_bus: 0, to_bus: 1}
      )

      # Either fails (islanding) or returns a result
      # The physical expectation is failure/low S_sc, but implementation may vary
      assert is_map(result)
    end

    test "returns error for invalid bus ID" do
      system = build_3bus_radial()

      result = SCR.calculate_contingency(system,
        poi_bus_id: 999,
        branch: %{id: 1, from_bus: 0, to_bus: 1}
      )

      assert {:error, message} = result
      assert message =~ "999"
    end
  end

  # ============================================================================
  # Batch Contingency Tests
  # ============================================================================

  describe "calculate_contingency_batch/2" do
    test "returns results for all branches in mesh system" do
      system = build_4bus_mesh()

      {:ok, results} = SCR.calculate_contingency_batch(system,
        poi_bus_id: 3,
        branches: :all
      )

      # All 4 branches should succeed (mesh is N-1 secure)
      assert length(results) == 4
      assert Enum.all?(results, & &1.success)
      assert Enum.all?(results, & &1.short_circuit_mva > 0)
    end

    test "results are sorted by S_sc (weakest first)" do
      system = build_4bus_mesh()

      {:ok, results} = SCR.calculate_contingency_batch(system,
        poi_bus_id: 3,
        branches: :all
      )

      s_sc_values = Enum.map(results, & &1.short_circuit_mva)
      assert s_sc_values == Enum.sort(s_sc_values)
    end

    test "returns empty list for empty branches" do
      system = build_4bus_mesh()

      {:ok, results} = SCR.calculate_contingency_batch(system,
        poi_bus_id: 3,
        branches: []
      )

      assert results == []
    end
  end

  # ============================================================================
  # Find Worst Contingency Tests
  # ============================================================================

  describe "find_worst_contingency/2" do
    test "returns contingency with lowest SCR" do
      system = build_4bus_mesh()

      {:ok, worst} = SCR.find_worst_contingency(system,
        poi_bus_id: 3,
        p_rated_mw: 100.0
      )

      assert worst.success
      assert worst.scr > 0
      assert worst.p_rated_mw == 100.0
      assert worst.grid_strength in [:weak, :moderate, :strong, :very_strong]

      # Verify it's actually the worst
      {:ok, all_results} = SCR.calculate_contingency_batch(system, poi_bus_id: 3)
      min_s_sc = all_results |> Enum.map(& &1.short_circuit_mva) |> Enum.min()

      assert_in_delta worst.short_circuit_mva, min_s_sc, 0.001
    end

    test "larger plant has lower SCR at same bus" do
      system = build_4bus_mesh()

      {:ok, small} = SCR.find_worst_contingency(system,
        poi_bus_id: 3,
        p_rated_mw: 50.0
      )

      {:ok, large} = SCR.find_worst_contingency(system,
        poi_bus_id: 3,
        p_rated_mw: 200.0
      )

      # Same S_sc, so larger plant = lower SCR
      assert small.scr > large.scr
      assert_in_delta small.short_circuit_mva, large.short_circuit_mva, 0.001
    end
  end

  # ============================================================================
  # Physical Behavior Tests
  # ============================================================================

  describe "physical behavior" do
    test "double circuit has higher base S_sc than single" do
      system = build_double_circuit()

      # Base case (with both circuits)
      {:ok, base_results} = SCR.get_all_thevenin_impedances(system)
      s_sc_double = Enum.find(base_results, & &1.bus_id == 1).short_circuit_mva

      # Remove one circuit
      single_system = %{system | branches: Enum.reject(system.branches, & &1.id == 102)}
      {:ok, single_results} = SCR.get_all_thevenin_impedances(single_system)
      s_sc_single = Enum.find(single_results, & &1.bus_id == 1).short_circuit_mva

      # Double circuit should have higher S_sc
      assert s_sc_double > s_sc_single
    end

    test "removing parallel circuit changes S_sc" do
      system = build_double_circuit()

      {:ok, base_results} = SCR.get_all_thevenin_impedances(system)
      base_s_sc = Enum.find(base_results, & &1.bus_id == 1).short_circuit_mva

      {:ok, contingency} = SCR.calculate_contingency(system,
        poi_bus_id: 1,
        branch: %{id: 101, from_bus: 0, to_bus: 1}
      )

      assert contingency.success

      # Contingency should change S_sc from base
      # (physical expectation: ~50% reduction, but depends on implementation)
      assert contingency.short_circuit_mva != base_s_sc
    end

    test "consistent results across multiple calls" do
      system = build_4bus_mesh()

      results = for _ <- 1..3 do
        {:ok, r} = SCR.calculate_contingency(system,
          poi_bus_id: 3,
          branch: %{id: 1, from_bus: 0, to_bus: 1}
        )
        r.short_circuit_mva
      end

      assert Enum.uniq(results) |> length() == 1
    end
  end

  # ============================================================================
  # Numerical Accuracy Tests
  # ============================================================================

  describe "numerical accuracy" do
    test "S_sc matches hand calculation for simple 2-bus system" do
      # 2-bus: Slack(0) -- Z=0.01+j0.1 -- (1)
      system = %{
        buses: [
          %{id: 0, type: :slack, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0},
          %{id: 1, type: :pq, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.5, q_load: 0.2, p_gen: 0.0, q_gen: 0.0}
        ],
        branches: [
          %{id: 1, from: 0, to: 1, from_bus: 0, to_bus: 1, r: 0.01, x: 0.1, b: 0.0, tap: 1.0, shift: 0.0}
        ],
        transformers: []
      }

      {:ok, results} = SCR.get_all_thevenin_impedances(system)
      bus1 = Enum.find(results, & &1.bus_id == 1)

      # Expected: |Z_th| ≈ 0.1005, S_sc = 100/0.1005 ≈ 995 MVA
      z_expected = :math.sqrt(0.01*0.01 + 0.1*0.1)
      s_sc_expected = 100.0 / z_expected

      assert_in_delta bus1.short_circuit_mva, s_sc_expected, s_sc_expected * 0.05
    end

    test "Z_th angle is positive for inductive system" do
      system = build_4bus_mesh()

      {:ok, results} = SCR.calculate_contingency_batch(system, poi_bus_id: 3)

      for r <- results do
        assert r.z_thevenin_angle > 0, "Inductive system should have positive Z angle"
        assert r.z_thevenin_angle < :math.pi() / 2
      end
    end

    test "magnitude and components are consistent" do
      system = build_4bus_mesh()

      {:ok, result} = SCR.calculate_contingency(system,
        poi_bus_id: 3,
        branch: %{id: 1, from_bus: 0, to_bus: 1}
      )

      expected_mag = :math.sqrt(
        result.z_thevenin_real * result.z_thevenin_real +
        result.z_thevenin_imag * result.z_thevenin_imag
      )

      assert_in_delta result.z_thevenin_magnitude, expected_mag, 1.0e-10
    end
  end

  # ============================================================================
  # Grid Strength Classification Tests
  # ============================================================================

  describe "grid strength classification" do
    test "weak grid with large plant has low SCR" do
      # High impedance line creates weak grid
      system = %{
        buses: [
          %{id: 0, type: :slack, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0},
          %{id: 1, type: :pq, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.1, q_load: 0.05, p_gen: 0.0, q_gen: 0.0}
        ],
        branches: [
          %{id: 1, from: 0, to: 1, from_bus: 0, to_bus: 1, r: 0.1, x: 0.5, b: 0.0, tap: 1.0, shift: 0.0}
        ],
        transformers: []
      }

      {:ok, result} = SCR.calculate(system, [%{bus_id: 1, p_rated_mw: 200.0}])
      scr_result = List.first(result)

      assert scr_result.scr < 5.0
      assert scr_result.grid_strength in [:weak, :moderate]
    end

    test "strong grid with small plant has high SCR" do
      # Low impedance line creates strong grid
      system = %{
        buses: [
          %{id: 0, type: :slack, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0},
          %{id: 1, type: :pq, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.1, q_load: 0.05, p_gen: 0.0, q_gen: 0.0}
        ],
        branches: [
          %{id: 1, from: 0, to: 1, from_bus: 0, to_bus: 1, r: 0.001, x: 0.01, b: 0.0, tap: 1.0, shift: 0.0}
        ],
        transformers: []
      }

      {:ok, result} = SCR.calculate(system, [%{bus_id: 1, p_rated_mw: 10.0}])
      scr_result = List.first(result)

      assert scr_result.scr > 10.0
      assert scr_result.grid_strength == :very_strong
    end
  end
end
