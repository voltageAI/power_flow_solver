# Comparison of PowerFlowSolver vs PowerWorld results
#
# Run with: elixir compare_results.exs

# Your solver results (0-based indexing)
solver_results = [
  {0, 1.000000, 0.000, "SLACK"},
  {1, 1.000000, 3.152, "PV"},
  {2, 1.005973, -1.750, "PQ"},
  {3, 0.989799, -1.576, "PQ"},
  {4, 1.016985, -7.098, "PQ"},
  {5, 1.005162, 1.298, "PQ"}
]

# PowerWorld results (1-based indexing)
powerworld_results = [
  {1, 1.000000, 0.031, "SLACK"},
  {2, 0.986581, 3.359, "PQ"},
  {3, 0.998545, -1.696, "PQ"},
  {4, 0.979093, -1.500, "PQ"},
  {5, 1.005778, -7.144, "PQ"},
  {6, 0.994052, 1.454, "PQ"}
]

IO.puts("\n=== VOLTAGE COMPARISON ===\n")
IO.puts("Bus | Type  | Solver V  | PW V      | ΔV (%)   | Solver θ | PW θ     | Δθ (°)")
IO.puts(String.duplicate("-", 85))

Enum.zip(solver_results, powerworld_results)
|> Enum.each(fn {{bus_0, v_solver, theta_solver, type_solver},
                 {_bus_1, v_pw, theta_pw, _type_pw}} ->
  delta_v = v_solver - v_pw
  delta_v_pct = (delta_v / v_pw) * 100.0
  delta_theta = theta_solver - theta_pw

  type_str = String.pad_trailing(type_solver, 5)

  v_solver_str = :io_lib.format("~9.6f", [v_solver]) |> to_string()
  v_pw_str = :io_lib.format("~9.6f", [v_pw]) |> to_string()
  delta_v_str = :io_lib.format("~8.4f", [delta_v_pct]) |> to_string() |> String.pad_leading(9)
  theta_solver_str = :io_lib.format("~8.3f", [theta_solver]) |> to_string() |> String.pad_leading(9)
  theta_pw_str = :io_lib.format("~8.3f", [theta_pw]) |> to_string() |> String.pad_leading(9)
  delta_theta_str = :io_lib.format("~7.3f", [delta_theta]) |> to_string() |> String.pad_leading(8)

  IO.puts(
    "#{String.pad_leading(to_string(bus_0), 2)}  | #{type_str} | " <>
    "#{v_solver_str} | #{v_pw_str} | #{delta_v_str} | " <>
    "#{theta_solver_str} | #{theta_pw_str} | #{delta_theta_str}"
  )
end)

IO.puts("\n=== ANALYSIS ===\n")
IO.puts("Key observations:")
IO.puts("1. Bus 1 (PV): PowerWorld shows 0.9866 pu, Solver shows 1.0 pu")
IO.puts("   - This is a ~1.35% difference in voltage magnitude")
IO.puts("   - PV buses should hold voltage magnitude constant at specified setpoint")
IO.puts("   - Need to check what voltage setpoint PowerWorld is using")
IO.puts("\n2. All voltage magnitudes differ by ~1%")
IO.puts("   - Suggests possible per-unit base mismatch or modeling difference")
IO.puts("\n3. Angles are very close (within 0.2°)")
IO.puts("   - This suggests the network model is correct")
IO.puts("   - The issue is likely in voltage magnitude control")
