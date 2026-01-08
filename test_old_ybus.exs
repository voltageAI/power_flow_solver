#!/usr/bin/env elixir

# Test if the RAW file voltages are the solution for the OLD Y-bus
# (with physical transformer model, tap = windv1*nomv2 / windv2*nomv1)

require Logger

test_data_dir = Path.join(__DIR__, "test/test_data")
raw_file_path = Path.join(test_data_dir, "000_6bus_v33.RAW")

{:ok, {buses, lines, _shunts, _metadata}} =
  PowerSystemParsers.PsseRawParser.parse_file(raw_file_path)

Logger.info("\n" <> String.duplicate("=", 80))
Logger.info("TESTING: Are RAW voltages solved for OLD Y-bus?")
Logger.info(String.duplicate("=", 80))

Logger.info("\nCurrent parser tap ratios:")
Enum.each(lines, fn line ->
  tap = Map.get(line, :tap, 1.0)
  if tap != 1.0 do
    Logger.info("  #{line.from} → #{line.to}: tap = #{tap}")
  end
end)

# Manually override tap ratios to OLD formula
Logger.info("\nManually calculating OLD tap ratios (windv1*nomv2 / windv2*nomv1):")

# From RAW file we know:
# Bus 2→4: windv1=1.0, nomv1=345, windv2=1.0, nomv2=138
# Bus 2→6: windv1=1.0, nomv1=138, windv2=1.0, nomv2=138
# Bus 4→3: windv1=1.0, nomv1=138, windv2=1.0, nomv2=345
# Bus 4→5: windv1=0.96875, nomv1=138, windv2=1.0, nomv2=34.5

old_taps = %{
  {1, 3} => (1.0 * 138.0) / (1.0 * 345.0),      # Bus 2→4: 0.4
  {1, 5} => (1.0 * 138.0) / (1.0 * 138.0),      # Bus 2→6: 1.0
  {3, 2} => (1.0 * 345.0) / (1.0 * 138.0),      # Bus 4→3: 2.5
  {3, 4} => (0.96875 * 34.5) / (1.0 * 138.0)    # Bus 4→5: 0.2421875
}

Enum.each(old_taps, fn {{from, to}, tap} ->
  Logger.info("  #{from} → #{to}: tap = #{tap}")
end)

# Create modified lines with OLD taps
lines_with_old_taps = Enum.map(lines, fn line ->
  from = line.from
  to = line.to
  old_tap = Map.get(old_taps, {from, to})

  if old_tap != nil do
    Map.put(line, :tap, old_tap)
  else
    line
  end
end)

Logger.info("\n--- Testing with NEW Y-bus (current parser) ---")
system_new = %{buses: buses, lines: lines}

case PowerFlowSolver.NewtonRaphson.solve(system_new, max_iterations: 100, tolerance: 1.0e-6) do
  {:ok, _solution, iterations} ->
    Logger.info("Converged in #{iterations} iterations")
  {:error, reason} ->
    Logger.info("Failed: #{reason}")
end

Logger.info("\n--- Testing with OLD Y-bus (physical model taps) ---")
system_old = %{buses: buses, lines: lines_with_old_taps}

case PowerFlowSolver.NewtonRaphson.solve(system_old, max_iterations: 100, tolerance: 1.0e-6) do
  {:ok, solution, iterations} ->
    Logger.info("Converged in #{iterations} iterations")

    if iterations == 1 do
      Logger.info("\n🎯 RAW FILE VOLTAGES ARE THE SOLUTION FOR OLD Y-BUS!")
    end

  {:error, reason} ->
    Logger.info("Failed: #{reason}")
end

Logger.info("\n" <> String.duplicate("=", 80))
