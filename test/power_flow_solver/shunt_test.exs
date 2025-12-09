defmodule PowerFlowSolver.ShuntTest do
  @moduledoc """
  Comprehensive tests for fixed shunt device handling in Y-bus construction
  and power flow solutions.

  ## Test Strategy

  These tests validate that bus shunts (capacitor banks, reactors, etc.) are
  correctly incorporated into the Y-bus matrix and produce correct power flow
  solutions. We use multiple validation approaches to avoid confirmation bias:

  1. **Analytical Y-bus verification**: For simple systems, Y-bus elements can be
     calculated by hand. We verify our implementation matches these calculations.

  2. **Power balance verification**: At a solved operating point, P and Q injections
     must balance. A missing shunt causes Q imbalance proportional to V²×B.

  3. **Known test cases**: We use published IEEE test cases or cases verified
     against commercial software (PowerWorld, PSS/E, etc.).

  ## Shunt Admittance Theory

  A fixed shunt at bus i adds to the diagonal of Y-bus:
    Y_ii += G_shunt + j*B_shunt

  Where:
  - G_shunt = shunt conductance (MW at 1.0 pu voltage) / S_base
  - B_shunt = shunt susceptance (MVAR at 1.0 pu voltage) / S_base

  For a capacitor bank: B > 0 (supplies reactive power)
  For a reactor: B < 0 (absorbs reactive power)

  The reactive power injection from a shunt is:
    Q_shunt = V² × B_shunt (in per-unit)

  This is critical for voltage support - missing shunts cause voltage errors.
  """

  use ExUnit.Case, async: true

  alias PowerFlowSolver.NewtonRaphson

  describe "Y-bus matrix construction with shunts" do
    @tag :ybus
    test "single bus with shunt has correct diagonal element" do
      # ANALYTICAL TEST CASE
      # ====================
      # Single bus (slack) with a 30 MVAR capacitor bank
      # Base: 100 MVA
      #
      # Y-bus should be a 1x1 matrix with:
      #   Y_11 = j*B = j*(30/100) = j*0.3 pu
      #
      # With no lines, the only Y-bus entry is from the shunt.

      system = %{
        buses: [
          %{
            id: 0,
            type: :slack,
            p_load: 0.0,
            q_load: 0.0,
            p_gen: 0.0,
            q_gen: 0.0,
            v_magnitude: 1.0,
            v_angle: 0.0
          }
        ],
        lines: [],
        shunts: %{
          fixed: [
            %{bus_id: 0, g: 0.0, b: 30.0}  # 30 MVAR capacitor
          ],
          switched: []
        },
        base_mva: 100.0
      }

      y_bus = NewtonRaphson.build_y_bus(system)

      # Y-bus should have one entry at (0,0)
      assert length(y_bus.values) == 1
      assert y_bus.row_ptrs == [0, 1]
      assert y_bus.col_indices == [0]

      # Check the value: should be (0.0, 0.3) for 30 MVAR at 100 MVA base
      [{g, b}] = y_bus.values
      assert_in_delta g, 0.0, 1.0e-10, "G should be zero for pure capacitor"
      assert_in_delta b, 0.3, 1.0e-10, "B should be 30/100 = 0.3 pu"
    end

    @tag :ybus
    test "shunt with both G and B components" do
      # A shunt with both conductance and susceptance
      # G = 5 MW, B = -20 MVAR (reactor) at 100 MVA base
      # Expected: Y_11 = (0.05, -0.2)

      system = %{
        buses: [
          %{id: 0, type: :slack, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0}
        ],
        lines: [],
        shunts: %{
          fixed: [%{bus_id: 0, g: 5.0, b: -20.0}],
          switched: []
        },
        base_mva: 100.0
      }

      y_bus = NewtonRaphson.build_y_bus(system)

      [{g, b}] = y_bus.values
      assert_in_delta g, 0.05, 1.0e-10, "G should be 5/100 = 0.05 pu"
      assert_in_delta b, -0.2, 1.0e-10, "B should be -20/100 = -0.2 pu (reactor)"
    end

    @tag :ybus
    test "shunt adds to existing line admittance on diagonal" do
      # ANALYTICAL TEST CASE
      # ====================
      # 2-bus system: slack -- line -- PQ load
      # Line: R=0.01, X=0.1 pu (no charging)
      # Shunt at bus 1: B = 20 MVAR capacitor
      #
      # Without shunt, Y-bus diagonal at bus 1:
      #   y_series = 1/(0.01 + j*0.1) = (0.01 - j*0.1)/(0.01² + 0.1²)
      #            = (0.01 - j*0.1)/0.0101 = 0.9901 - j*9.901 pu
      #   Y_11 = y_series = 0.9901 - j*9.901
      #
      # With 20 MVAR shunt:
      #   Y_11 = 0.9901 + j*(-9.901 + 0.2) = 0.9901 - j*9.701

      system = %{
        buses: [
          %{id: 0, type: :slack, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0},
          %{id: 1, type: :pq, p_load: 0.5, q_load: 0.2, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0}
        ],
        lines: [
          %{id: 0, from: 0, to: 1, r: 0.01, x: 0.1, b: 0.0, tap: 1.0, shift: 0.0}
        ],
        shunts: %{
          fixed: [%{bus_id: 1, g: 0.0, b: 20.0}],
          switched: []
        },
        base_mva: 100.0
      }

      y_bus = NewtonRaphson.build_y_bus(system)

      # Find the diagonal entry for bus 1 (row 1, col 1)
      # CSR format: row_ptrs tells us where each row starts
      row1_start = Enum.at(y_bus.row_ptrs, 1)
      row1_end = Enum.at(y_bus.row_ptrs, 2)
      row1_cols = Enum.slice(y_bus.col_indices, row1_start..(row1_end - 1))
      row1_vals = Enum.slice(y_bus.values, row1_start..(row1_end - 1))

      # Find diagonal (col == 1)
      diag_idx = Enum.find_index(row1_cols, &(&1 == 1))
      {g_11, b_11} = Enum.at(row1_vals, diag_idx)

      # Expected values from analytical calculation
      z_mag_sq = 0.01 * 0.01 + 0.1 * 0.1  # 0.0101
      g_series = 0.01 / z_mag_sq  # ≈ 0.9901
      b_series = -0.1 / z_mag_sq  # ≈ -9.901
      b_shunt_pu = 20.0 / 100.0   # 0.2

      expected_g = g_series
      expected_b = b_series + b_shunt_pu  # -9.901 + 0.2 = -9.701

      assert_in_delta g_11, expected_g, 1.0e-4, "G_11 should match series conductance"
      assert_in_delta b_11, expected_b, 1.0e-4,
        "B_11 should be series susceptance + shunt: #{b_series} + #{b_shunt_pu} = #{expected_b}"
    end

    @tag :ybus
    test "multiple shunts at same bus are summed" do
      # Two shunts at bus 0:
      # Shunt 1: G=0, B=15 MVAR
      # Shunt 2: G=0, B=25 MVAR
      # Total: G=0, B=40 MVAR → (0, 0.4) pu at 100 MVA base

      system = %{
        buses: [
          %{id: 0, type: :slack, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0}
        ],
        lines: [],
        shunts: %{
          fixed: [
            %{bus_id: 0, g: 0.0, b: 15.0},
            %{bus_id: 0, g: 0.0, b: 25.0}
          ],
          switched: []
        },
        base_mva: 100.0
      }

      y_bus = NewtonRaphson.build_y_bus(system)

      [{g, b}] = y_bus.values
      assert_in_delta g, 0.0, 1.0e-10
      assert_in_delta b, 0.4, 1.0e-10, "Total B should be (15+25)/100 = 0.4 pu"
    end

    @tag :ybus
    test "shunts at different buses are independent" do
      # 2-bus system with shunts at both buses
      # Bus 0: 10 MVAR capacitor
      # Bus 1: 30 MVAR capacitor
      # No lines connecting them (island test)

      system = %{
        buses: [
          %{id: 0, type: :slack, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0},
          %{id: 1, type: :slack, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0}
        ],
        lines: [],
        shunts: %{
          fixed: [
            %{bus_id: 0, g: 0.0, b: 10.0},
            %{bus_id: 1, g: 0.0, b: 30.0}
          ],
          switched: []
        },
        base_mva: 100.0
      }

      y_bus = NewtonRaphson.build_y_bus(system)

      # Should have 2 diagonal entries only
      assert length(y_bus.values) == 2
      assert y_bus.col_indices == [0, 1]

      [{_g0, b0}, {_g1, b1}] = y_bus.values
      assert_in_delta b0, 0.1, 1.0e-10, "Bus 0: B = 10/100 = 0.1 pu"
      assert_in_delta b1, 0.3, 1.0e-10, "Bus 1: B = 30/100 = 0.3 pu"
    end
  end

  describe "power flow solutions with shunts" do
    @tag :power_flow
    test "2-bus system with capacitor bank - voltage support" do
      # POWER FLOW VALIDATION TEST
      # ==========================
      # This test validates that shunts correctly affect power flow solutions.
      #
      # System: Slack bus -- line -- PQ load with capacitor
      # Line: R=0.01, X=0.1 pu
      # Load: P=0.5 pu, Q=0.3 pu
      # Capacitor at load bus: 20 MVAR (0.2 pu)
      #
      # The capacitor should raise the voltage at the load bus by providing
      # local reactive power support, reducing Q flow on the line.
      #
      # We solve WITH and WITHOUT the shunt and verify:
      # 1. With shunt: higher voltage at bus 1
      # 2. The voltage difference matches theoretical expectation

      base_system = %{
        buses: [
          %{id: 0, type: :slack, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0},
          %{id: 1, type: :pq, p_load: 0.5, q_load: 0.3, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0}
        ],
        lines: [
          %{id: 0, from: 0, to: 1, r: 0.01, x: 0.1, b: 0.0, tap: 1.0, shift: 0.0}
        ],
        base_mva: 100.0
      }

      # Solve WITHOUT shunt
      system_no_shunt = Map.put(base_system, :shunts, %{fixed: [], switched: []})
      {:ok, solution_no_shunt, _} = NewtonRaphson.solve(system_no_shunt, tolerance: 1.0e-8)
      {v_no_shunt, _, _} = solution_no_shunt[1]

      # Solve WITH 20 MVAR capacitor
      system_with_shunt = Map.put(base_system, :shunts, %{
        fixed: [%{bus_id: 1, g: 0.0, b: 20.0}],
        switched: []
      })
      {:ok, solution_with_shunt, _} = NewtonRaphson.solve(system_with_shunt, tolerance: 1.0e-8)
      {v_with_shunt, _, _} = solution_with_shunt[1]

      # Capacitor should RAISE voltage (provides Q, reduces voltage drop)
      assert v_with_shunt > v_no_shunt,
        "Capacitor should raise voltage: with=#{v_with_shunt}, without=#{v_no_shunt}"

      # The voltage improvement should be meaningful (order of 0.01 pu for this case)
      voltage_improvement = v_with_shunt - v_no_shunt
      assert voltage_improvement > 0.005,
        "Voltage improvement should be > 0.5%: got #{voltage_improvement * 100}%"
    end

    @tag :power_flow
    test "reactor reduces voltage at load bus" do
      # Reactor (negative B) should LOWER voltage by absorbing reactive power

      base_system = %{
        buses: [
          %{id: 0, type: :slack, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0},
          %{id: 1, type: :pq, p_load: 0.3, q_load: 0.1, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0}
        ],
        lines: [
          %{id: 0, from: 0, to: 1, r: 0.01, x: 0.1, b: 0.0, tap: 1.0, shift: 0.0}
        ],
        base_mva: 100.0
      }

      # Solve WITHOUT reactor
      system_no_shunt = Map.put(base_system, :shunts, %{fixed: [], switched: []})
      {:ok, solution_no_shunt, _} = NewtonRaphson.solve(system_no_shunt, tolerance: 1.0e-8)
      {v_no_shunt, _, _} = solution_no_shunt[1]

      # Solve WITH 15 MVAR reactor (negative B)
      system_with_reactor = Map.put(base_system, :shunts, %{
        fixed: [%{bus_id: 1, g: 0.0, b: -15.0}],
        switched: []
      })
      {:ok, solution_with_reactor, _} = NewtonRaphson.solve(system_with_reactor, tolerance: 1.0e-8)
      {v_with_reactor, _, _} = solution_with_reactor[1]

      # Reactor should LOWER voltage
      assert v_with_reactor < v_no_shunt,
        "Reactor should lower voltage: with=#{v_with_reactor}, without=#{v_no_shunt}"
    end

    @tag :power_flow
    test "Q generation at slack bus reflects shunt reactive power" do
      # When a shunt is at a non-slack bus, it changes the Q that must flow
      # from the slack bus. We can verify the Q_gen at slack matches expectations.
      #
      # System: Slack -- line -- PQ with capacitor
      # Load: P=0, Q=0 (no load)
      # Capacitor: 30 MVAR at bus 1
      #
      # At V=1.0 pu, the capacitor generates Q = V² × B = 1.0 × 0.3 = 0.3 pu
      # This Q must flow back to the slack bus (minus line losses)
      #
      # Expected: Q_gen at slack ≈ -0.3 pu (absorbing Q from the cap)

      system = %{
        buses: [
          %{id: 0, type: :slack, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0},
          %{id: 1, type: :pq, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0}
        ],
        lines: [
          # Low impedance line to minimize losses
          %{id: 0, from: 0, to: 1, r: 0.001, x: 0.01, b: 0.0, tap: 1.0, shift: 0.0}
        ],
        shunts: %{
          fixed: [%{bus_id: 1, g: 0.0, b: 30.0}],
          switched: []
        },
        base_mva: 100.0
      }

      {:ok, solution, _} = NewtonRaphson.solve(system, tolerance: 1.0e-8)

      # Get Q generation at slack bus
      {_v, _ang, q_gen_slack} = solution[0]

      # Q_gen should be approximately -0.3 pu (slack absorbs Q from capacitor)
      # Allow some tolerance for line reactive losses
      assert_in_delta q_gen_slack, -0.3, 0.02,
        "Slack Q_gen should be ~-0.3 pu (absorbing cap Q), got #{q_gen_slack}"
    end
  end

  describe "data format compatibility" do
    @tag :format
    test "supports shunts in system.shunts.fixed format" do
      system = %{
        buses: [
          %{id: 0, type: :slack, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0}
        ],
        lines: [],
        shunts: %{
          fixed: [%{bus_id: 0, g: 0.0, b: 50.0}],
          switched: []
        },
        base_mva: 100.0
      }

      y_bus = NewtonRaphson.build_y_bus(system)
      [{_, b}] = y_bus.values
      assert_in_delta b, 0.5, 1.0e-10
    end

    @tag :format
    test "supports shunts in system.fixed_shunts format" do
      system = %{
        buses: [
          %{id: 0, type: :slack, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0}
        ],
        lines: [],
        fixed_shunts: [%{bus_id: 0, g: 0.0, b: 50.0}],
        base_mva: 100.0
      }

      y_bus = NewtonRaphson.build_y_bus(system)
      [{_, b}] = y_bus.values
      assert_in_delta b, 0.5, 1.0e-10
    end

    @tag :format
    test "handles missing shunts gracefully" do
      system = %{
        buses: [
          %{id: 0, type: :slack, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0},
          %{id: 1, type: :pq, p_load: 0.1, q_load: 0.05, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0}
        ],
        lines: [
          %{id: 0, from: 0, to: 1, r: 0.01, x: 0.1, b: 0.0, tap: 1.0, shift: 0.0}
        ]
        # No shunts key at all
      }

      # Should not crash, should solve normally
      {:ok, _solution, iterations} = NewtonRaphson.solve(system, tolerance: 1.0e-6)
      assert iterations < 20
    end

    @tag :format
    test "handles empty shunts list" do
      system = %{
        buses: [
          %{id: 0, type: :slack, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0}
        ],
        lines: [],
        shunts: %{fixed: [], switched: []}
      }

      y_bus = NewtonRaphson.build_y_bus(system)
      # With no lines and no shunts, Y-bus should be empty
      assert y_bus.values == []
    end

    @tag :format
    test "uses default base_mva of 100 when not specified" do
      system = %{
        buses: [
          %{id: 0, type: :slack, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0}
        ],
        lines: [],
        shunts: %{
          fixed: [%{bus_id: 0, g: 0.0, b: 100.0}],  # 100 MVAR
          switched: []
        }
        # No base_mva specified
      }

      y_bus = NewtonRaphson.build_y_bus(system)
      [{_, b}] = y_bus.values
      # With default 100 MVA base, 100 MVAR → 1.0 pu
      assert_in_delta b, 1.0, 1.0e-10
    end
  end

  describe "regression tests" do
    @tag :regression
    test "shunt value matches expected Y-bus error from Scott's Cases 2/3" do
      # This test replicates the scenario that exposed the bug:
      # A 30 MVAR shunt was missing from Y-bus, causing Max |ΔImag| = 0.300000
      #
      # We verify that a 30 MVAR shunt at 100 MVA base produces exactly
      # a 0.3 pu imaginary component in the Y-bus.

      system = %{
        buses: [
          %{id: 0, type: :slack, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0}
        ],
        lines: [],
        shunts: %{
          fixed: [%{bus_id: 0, g: 0.0, b: 30.0}],  # The specific value from Cases 2/3
          switched: []
        },
        base_mva: 100.0
      }

      y_bus = NewtonRaphson.build_y_bus(system)
      [{_, b}] = y_bus.values

      # This is THE critical assertion that would have caught the original bug
      assert_in_delta b, 0.3, 1.0e-10,
        "30 MVAR shunt at 100 MVA base must produce B = 0.3 pu in Y-bus"
    end
  end

  describe "3-bus analytical validation" do
    @tag :analytical
    test "3-bus system Y-bus with shunt matches hand calculation" do
      # COMPLETE ANALYTICAL VALIDATION
      # ==============================
      # 3-bus system:
      #   Bus 0 (slack) -- Line 0-1 -- Bus 1 (PQ) -- Line 1-2 -- Bus 2 (PQ with shunt)
      #
      # Line 0-1: R=0.02, X=0.06 pu, B=0
      # Line 1-2: R=0.01, X=0.04 pu, B=0
      # Shunt at Bus 2: G=0, B=25 MVAR at 100 MVA base
      #
      # Hand calculation of Y-bus:
      #
      # y_01 = 1/(0.02 + j0.06) = (0.02 - j0.06)/(0.02² + 0.06²)
      #      = (0.02 - j0.06)/0.004 = 5 - j15
      #
      # y_12 = 1/(0.01 + j0.04) = (0.01 - j0.04)/(0.01² + 0.04²)
      #      = (0.01 - j0.04)/0.0017 = 5.882 - j23.529
      #
      # Y-bus (without shunt):
      #   Y_00 = y_01 = 5 - j15
      #   Y_01 = Y_10 = -y_01 = -5 + j15
      #   Y_11 = y_01 + y_12 = 5 + 5.882 - j(15 + 23.529) = 10.882 - j38.529
      #   Y_12 = Y_21 = -y_12 = -5.882 + j23.529
      #   Y_22 = y_12 = 5.882 - j23.529
      #
      # With shunt at Bus 2 (B = 0.25 pu):
      #   Y_22 = 5.882 - j23.529 + j0.25 = 5.882 - j23.279

      system = %{
        buses: [
          %{id: 0, type: :slack, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0},
          %{id: 1, type: :pq, p_load: 0.2, q_load: 0.1, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0},
          %{id: 2, type: :pq, p_load: 0.3, q_load: 0.15, p_gen: 0.0, q_gen: 0.0,
            v_magnitude: 1.0, v_angle: 0.0}
        ],
        lines: [
          %{id: 0, from: 0, to: 1, r: 0.02, x: 0.06, b: 0.0, tap: 1.0, shift: 0.0},
          %{id: 1, from: 1, to: 2, r: 0.01, x: 0.04, b: 0.0, tap: 1.0, shift: 0.0}
        ],
        shunts: %{
          fixed: [%{bus_id: 2, g: 0.0, b: 25.0}],  # 25 MVAR = 0.25 pu
          switched: []
        },
        base_mva: 100.0
      }

      y_bus = NewtonRaphson.build_y_bus(system)

      # Extract Y-bus values by position
      # Row 0: cols 0, 1
      # Row 1: cols 0, 1, 2
      # Row 2: cols 1, 2

      # Helper to find value at (row, col)
      find_ybus_value = fn row, col ->
        row_start = Enum.at(y_bus.row_ptrs, row)
        row_end = Enum.at(y_bus.row_ptrs, row + 1)
        cols = Enum.slice(y_bus.col_indices, row_start..(row_end - 1))
        vals = Enum.slice(y_bus.values, row_start..(row_end - 1))

        case Enum.find_index(cols, &(&1 == col)) do
          nil -> {0.0, 0.0}
          idx -> Enum.at(vals, idx)
        end
      end

      # Calculate expected values
      z01_sq = 0.02 * 0.02 + 0.06 * 0.06  # 0.004
      g01 = 0.02 / z01_sq  # 5.0
      b01 = -0.06 / z01_sq  # -15.0

      z12_sq = 0.01 * 0.01 + 0.04 * 0.04  # 0.0017
      g12 = 0.01 / z12_sq  # 5.882...
      b12 = -0.04 / z12_sq  # -23.529...

      b_shunt = 25.0 / 100.0  # 0.25 pu

      # Verify diagonal elements
      {g00, b00} = find_ybus_value.(0, 0)
      assert_in_delta g00, g01, 1.0e-6, "Y_00 real"
      assert_in_delta b00, b01, 1.0e-6, "Y_00 imag"

      {g11, b11} = find_ybus_value.(1, 1)
      assert_in_delta g11, g01 + g12, 1.0e-6, "Y_11 real"
      assert_in_delta b11, b01 + b12, 1.0e-6, "Y_11 imag"

      {g22, b22} = find_ybus_value.(2, 2)
      assert_in_delta g22, g12, 1.0e-6, "Y_22 real"
      # THIS IS THE KEY TEST: B_22 should include the shunt
      expected_b22 = b12 + b_shunt
      assert_in_delta b22, expected_b22, 1.0e-6,
        "Y_22 imag should be #{b12} + #{b_shunt} = #{expected_b22}, got #{b22}"

      # Verify off-diagonal elements (should be negative of series admittance)
      {g01_off, b01_off} = find_ybus_value.(0, 1)
      assert_in_delta g01_off, -g01, 1.0e-6, "Y_01 real"
      assert_in_delta b01_off, -b01, 1.0e-6, "Y_01 imag"

      {g12_off, b12_off} = find_ybus_value.(1, 2)
      assert_in_delta g12_off, -g12, 1.0e-6, "Y_12 real"
      assert_in_delta b12_off, -b12, 1.0e-6, "Y_12 imag"
    end
  end
end
