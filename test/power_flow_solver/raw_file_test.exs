defmodule PowerFlowSolver.RawFileTest do
  use ExUnit.Case

  @test_data_dir Path.join(__DIR__, "../test_data")
  @raw_file_path Path.join(@test_data_dir, "0_6bus_start_simple_v33.raw")

  describe "PSS/E RAW file parsing" do
    test "reads and parses 6-bus test case" do
      # Parse the RAW file
      assert File.exists?(@raw_file_path), "Test file not found: #{@raw_file_path}"

      {:ok, {buses, lines, _shunts, _metadata}} =
        PowerSystemParsers.PsseRawParser.parse_file(@raw_file_path)

      # Basic assertions
      assert length(buses) > 0
      assert length(lines) > 0

      # Test solver
      system = %{buses: buses, lines: lines}

      # Test 1: Without Q-limit enforcement
      assert {:ok, _solution, iterations} = PowerFlowSolver.NewtonRaphson.solve(system,
        max_iterations: 100,
        tolerance: 1.0e-6,
        enforce_q_limits: false
      )
      assert iterations <= 100

      # Test 2: With Q-limit enforcement
      assert {:ok, _solution, iterations} = PowerFlowSolver.NewtonRaphson.solve(system,
        max_iterations: 100,
        tolerance: 1.0e-6,
        enforce_q_limits: true
      )
      assert iterations <= 100
    end
  end
end
