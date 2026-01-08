#!/usr/bin/env elixir
#
# Simple 2-Bus Power Flow Example
#
# System:
#   Bus 0 (Slack): 1.0 pu, generating 1.5 pu, load 0.5 pu
#   Bus 1 (PQ):    Load 1.0 pu active, 0.5 pu reactive
#   Line 0→1:      r=0.02 pu, x=0.06 pu
#

# Define the system
system = %{
  buses: [
    # Slack bus
    %{
      id: 0,
      type: :slack,
      p_gen: 1.5,
      p_load: 0.5,
      q_gen: 0.0,
      q_load: 0.0,
      v_magnitude: 1.0,
      v_angle: 0.0
    },
    # Load bus
    %{
      id: 1,
      type: :pq,
      p_gen: 0.0,
      p_load: 1.0,
      q_gen: 0.0,
      q_load: 0.5,
      v_magnitude: 1.0,
      v_angle: 0.0
    }
  ],
  lines: [
    %{
      from: 0,
      to: 1,
      r: 0.02,
      x: 0.06,
      b: 0.0
    }
  ]
}

IO.puts("\n=== 2-Bus Power Flow Example ===\n")

IO.puts("System:")
IO.puts("  Bus 0 (Slack): P_net = #{system.buses |> Enum.at(0) |> then(&(&1.p_gen - &1.p_load))} pu")
IO.puts("  Bus 1 (PQ):    P_load = 1.0 pu, Q_load = 0.5 pu")
IO.puts("  Line 0→1:      R = 0.02 pu, X = 0.06 pu\n")

# Solve
case PowerFlowSolver.NewtonRaphson.solve(system, tolerance: 1.0e-6) do
  {:ok, solution, iterations} ->
    IO.puts("✓ Converged in #{iterations} iterations\n")

    IO.puts("Results:")
    Enum.each(system.buses, fn bus ->
      {v_mag, v_angle} = Map.get(solution, bus.id)
      v_angle_deg = v_angle * 180.0 / :math.pi()

      IO.puts("  Bus #{bus.id}: V = #{Float.round(v_mag, 6)} pu ∠ #{Float.round(v_angle_deg, 4)}°")
    end)

  {:error, reason} ->
    IO.puts("✗ Failed: #{reason}")
end
