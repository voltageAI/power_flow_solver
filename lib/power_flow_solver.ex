defmodule PowerFlowSolver do
  @moduledoc """
  High-performance power flow solver using Rust NIFs.

  This library provides a complete Newton-Raphson power flow solver implementation
  in Rust with Elixir bindings. It's optimized for large power systems (10,000+ buses)
  and includes Q-limit enforcement for realistic generator modeling.

  ## Features

  - **Complete Rust Implementation**: Entire Newton-Raphson iteration loop runs in Rust
  - **Zero Boundary Crossings**: Eliminates Elixir/Rust context switches during iteration
  - **Sparse Linear Algebra**: Optimized sparse matrix operations using faer library
  - **Q-Limit Enforcement**: Realistic generator reactive power limit handling
  - **Parallel Jacobian Building**: Multi-threaded Jacobian construction
  - **Pre-compiled NIFs**: Fast installation with no Rust toolchain required

  ## Modules

  - `PowerFlowSolver.NewtonRaphson` - High-level Newton-Raphson solver API
  - `PowerFlowSolver.SparseLinearAlgebra` - Low-level sparse matrix operations

  ## Installation

  Add `power_flow_solver` to your dependencies:

  ```elixir
  def deps do
    [
      {:power_flow_solver, git: "https://github.com/voltageAI/power_flow_solver.git", tag: "v0.1.0"}
    ]
  end
  ```

  To force building from source (requires Rust toolchain):

  ```elixir
  # Set environment variable
  export POWER_FLOW_FORCE_BUILD=1
  mix deps.get
  ```

  ## Quick Start

  ```elixir
  # Define a simple power system
  system = %{
    buses: [
      %{id: 0, type: :slack, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0,
        v_magnitude: 1.0, v_angle: 0.0},
      %{id: 1, type: :pq, p_load: 0.5, q_load: 0.2, p_gen: 0.0, q_gen: 0.0,
        v_magnitude: 1.0, v_angle: 0.0}
    ],
    y_bus: %{
      row_ptrs: [0, 2, 4],
      col_indices: [0, 1, 0, 1],
      values: [
        {10.0, -50.0}, {-10.0, 50.0},
        {-10.0, 50.0}, {10.0, -50.0}
      ]
    }
  }

  # Solve power flow
  {:ok, solution, iterations} = PowerFlowSolver.NewtonRaphson.solve(system,
    max_iterations: 100,
    tolerance: 1.0e-2
  )
  ```

  ## System Format

  The solver expects a map with the following structure:

  ### Buses

  Each bus is a map with:
  - `:id` - Bus ID (integer)
  - `:type` - Bus type (`:slack`, `:pv`, or `:pq`)
  - `:p_load` - Real power load (per-unit)
  - `:q_load` - Reactive power load (per-unit)
  - `:p_gen` - Real power generation (per-unit)
  - `:q_gen` - Reactive power generation (per-unit)
  - `:v_magnitude` - Voltage magnitude (per-unit)
  - `:v_angle` - Voltage angle (radians)
  - `:q_min` - Min reactive power limit (optional, per-unit)
  - `:q_max` - Max reactive power limit (optional, per-unit)

  ### Y-Bus Matrix

  Admittance matrix in Compressed Sparse Row (CSR) format:
  - `:row_ptrs` - Row pointers (list of integers)
  - `:col_indices` - Column indices (list of integers)
  - `:values` - Complex values as tuples `{real, imag}`

  ## Performance

  Typical performance on an M1 MacBook Pro:

  - 100 buses: ~5ms
  - 1,000 buses: ~50ms
  - 10,000 buses: ~1.5s
  - 30,000 buses: ~8s

  Performance scales well due to:
  - Sparse matrix operations (O(nnz) not O(n²))
  - Parallel Jacobian building
  - Efficient Rust implementation
  - Zero Elixir/Rust boundary crossings during iteration

  ## License

  MIT License

  Copyright (c) 2024 Voltage AI
  """

  alias PowerFlowSolver.{NewtonRaphson, SparseLinearAlgebra}

  @doc """
  Returns the version of the PowerFlowSolver library.
  """
  def version, do: "0.1.0"

  @doc """
  Convenience function to solve power flow using Newton-Raphson.

  See `PowerFlowSolver.NewtonRaphson.solve/2` for detailed documentation.
  """
  defdelegate solve(system, opts \\ []), to: NewtonRaphson
end
