#!/usr/bin/env elixir

require Logger

test_data_dir = Path.join(__DIR__, "test/test_data")
raw_file_path = Path.join(test_data_dir, "000_6bus_v33.RAW")

{:ok, {buses, lines, _shunts, _metadata}} =
  PowerSystemParsers.PsseRawParser.parse_file(raw_file_path)

system = %{buses: buses, lines: lines}

Logger.info("\n" <> String.duplicate("=", 80))
Logger.info("TESTING WITH EXTREME TOLERANCE: 1e-24")
Logger.info(String.duplicate("=", 80))

case PowerFlowSolver.NewtonRaphson.solve(system,
       max_iterations: 100,
       tolerance: 1.0e-24) do
  {:ok, solution, iterations} ->
    Logger.info("\n✓ Converged in #{iterations} iterations")

    Logger.info("\nFinal voltages:")
    Enum.each(buses, fn bus ->
      {v_mag, v_ang} = Map.get(solution, bus.id)
      v_ang_deg = v_ang * 180.0 / :math.pi()
      Logger.info("  Bus #{bus.id}: V = #{:io_lib.format("~.15f", [v_mag])} ∠ #{:io_lib.format("~.10f", [v_ang_deg])}°")
    end)

  {:error, reason} ->
    Logger.info("\n✗ Failed: #{reason}")

    # Extract the final mismatch from the error message
    if String.contains?(reason, "mismatch:") do
      mismatch = reason
        |> String.split("mismatch: ")
        |> List.last()
        |> String.trim(")")
        |> String.to_float()

      Logger.info("\nFinal mismatch achieved: #{mismatch}")
      Logger.info("Mismatch in scientific notation: #{:io_lib.format("~e", [mismatch])}")
      Logger.info("Machine epsilon (float64): ~2.22e-16")
      Logger.info("Square root of machine epsilon: ~1.49e-8")
    end
end

Logger.info("\n" <> String.duplicate("=", 80))
