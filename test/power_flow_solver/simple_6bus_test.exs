defmodule PowerFlowSolver.Simple6BusTest do
  use ExUnit.Case

  @test_data_dir Path.join(__DIR__, "../test_data")
  @raw_file_path Path.join(@test_data_dir, "000_6bus_v33.RAW")

  test "parse RAW file and solve" do
    # Parse
    {:ok, {buses, lines, _shunts, _metadata}} =
      PowerSystemParsers.PsseRawParser.parse_file(@raw_file_path)

    # Solve
    system = %{buses: buses, lines: lines}

    case PowerFlowSolver.NewtonRaphson.solve(system,
           max_iterations: 100,
           tolerance: 1.0e-6) do
      {:ok, _solution, iterations} ->
        assert iterations <= 100

      {:error, reason} ->
        flunk("Failed to converge: #{reason}")
    end
  end
end
