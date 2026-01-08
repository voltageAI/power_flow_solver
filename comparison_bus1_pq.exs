# Comparison when Bus 1 is converted from PV to PQ

# Your solver results with Bus 1 as PQ (0-based indexing)
solver_pq_results = [
  {0, 1.000000, 0.000, "SLACK"},
  {1, 0.986279, 3.360, "PQ"},
  {2, 0.998662, -1.698, "PQ"},
  {3, 0.978991, -1.502, "PQ"},
  {4, 1.005668, -7.149, "PQ"},
  {5, 0.993908, 1.455, "PQ"}
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

IO.puts("\n=== VOLTAGE COMPARISON (Bus 1 as PQ) ===\n")
IO.puts("Bus | Type  | Solver V  | PW V      | ΔV (%)   | Solver θ | PW θ     | Δθ (°)")
IO.puts(String.duplicate("-", 85))

Enum.zip(solver_pq_results, powerworld_results)
|> Enum.each(fn {{bus_0, v_solver, theta_solver, type_solver},
                 {_bus_1, v_pw, theta_pw, _type_pw}} ->
  delta_v = v_solver - v_pw
  delta_v_pct = abs(delta_v / v_pw) * 100.0
  delta_theta = abs(theta_solver - theta_pw)

  v_solver_str = :io_lib.format("~9.6f", [v_solver]) |> to_string()
  v_pw_str = :io_lib.format("~9.6f", [v_pw]) |> to_string()
  delta_v_str = :io_lib.format("~8.4f", [delta_v_pct]) |> to_string() |> String.pad_leading(9)
  theta_solver_str = :io_lib.format("~8.3f", [theta_solver]) |> to_string() |> String.pad_leading(9)
  theta_pw_str = :io_lib.format("~8.3f", [theta_pw]) |> to_string() |> String.pad_leading(9)
  delta_theta_str = :io_lib.format("~7.3f", [delta_theta]) |> to_string() |> String.pad_leading(8)

  type_str = String.pad_trailing(type_solver, 5)

  IO.puts(
    "#{String.pad_leading(to_string(bus_0), 2)}  | #{type_str} | " <>
    "#{v_solver_str} | #{v_pw_str} | #{delta_v_str} | " <>
    "#{theta_solver_str} | #{theta_pw_str} | #{delta_theta_str}"
  )
end)

IO.puts("\n=== ANALYSIS ===\n")
IO.puts("✓ EXCELLENT MATCH!")
IO.puts("")
IO.puts("When Bus 1 is treated as PQ instead of PV:")
IO.puts("  • Bus 1: 0.986279 pu vs PowerWorld 0.986581 pu = 0.03% error")
IO.puts("  • Bus 1 angle: 3.360° vs PowerWorld 3.359° = 0.001° error")
IO.puts("  • All other buses match within 0.02-0.1%")
IO.puts("")
IO.puts("CONCLUSION:")
IO.puts("  PowerWorld is treating Bus 2 (your Bus 1) as a PQ bus, NOT a PV bus.")
IO.puts("  This could be because:")
IO.puts("    1. The generator at Bus 2 hit its reactive power limit")
IO.puts("    2. The generator was manually disabled in PowerWorld")
IO.puts("    3. PowerWorld case has different bus type settings than the RAW file")
