defmodule PowerFlowSolver.SixBus000Test do
  use ExUnit.Case

  @test_data_dir Path.join(__DIR__, "../test_data")
  @raw_file_path Path.join(@test_data_dir, "000_6bus_v33.RAW")

  describe "PSS/E RAW 000_6bus case" do
    test "solves 6-bus test case" do
      # Parse the RAW file
      assert File.exists?(@raw_file_path), "Test file not found: #{@raw_file_path}"

      {:ok, {buses, lines, _shunts, _metadata}} =
        PowerSystemParsers.PsseRawParser.parse_file(@raw_file_path)

      # Create system structure for solver
      system = %{buses: buses, lines: lines}

      # Solve power flow
      case PowerFlowSolver.NewtonRaphson.solve(system,
             max_iterations: 100,
             tolerance: 1.0e-6,
             enforce_q_limits: false
           ) do
        {:ok, solution, iterations} ->
          # Test assertions
          assert iterations <= 10, "Solution should converge in 10 iterations or less"

          # Check that voltage magnitudes are within reasonable bounds
          Enum.each(system.buses, fn bus ->
            {v_mag, _v_ang, _q_gen} = Map.get(solution, bus.id)
            assert v_mag >= 0.9 and v_mag <= 1.1, "Bus #{bus.id} voltage out of bounds"
          end)

        {:error, reason} ->
          flunk("Power flow failed to converge: #{reason}")
      end
    end
  end
end
