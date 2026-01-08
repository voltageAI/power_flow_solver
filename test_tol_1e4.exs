#!/usr/bin/env elixir

test_data_dir = Path.join(__DIR__, "test/test_data")
raw_file_path = Path.join(test_data_dir, "000_6bus_v33.RAW")

{:ok, {buses, lines, _shunts, _metadata}} =
  PowerSystemParsers.PsseRawParser.parse_file(raw_file_path)

system = %{buses: buses, lines: lines}

IO.puts("\nTesting with tolerance = 1.0e-4")

case PowerFlowSolver.NewtonRaphson.solve(system,
       max_iterations: 100,
       tolerance: 1.0e-4) do
  {:ok, _solution, iterations} ->
    IO.puts("✓ Converged in #{iterations} iterations")

  {:error, reason} ->
    IO.puts("✗ Failed: #{reason}")
end
