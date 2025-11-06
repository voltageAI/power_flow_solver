defmodule PowerFlowSolver.SparseLinearAlgebra do
  @moduledoc """
  Sparse linear algebra operations using native Rust implementation.

  This module provides high-performance sparse matrix operations optimized for
  power flow calculations. It uses Rust NIFs to call efficient sparse solvers.

  ## Supported Operations

  - Sparse LU factorization and solve
  - Compressed Sparse Row (CSR) matrix format
  - Complex number support for AC power flow

  ## Example

      # Define a sparse matrix in CSR format
      # Matrix A:
      # [10+0i,  2+0i,  0+0i]
      # [ 3+0i,  9+0i,  4+0i]
      # [ 0+0i,  1+0i,  7+0i]

      row_ptrs = [0, 2, 5, 7]
      col_indices = [0, 1, 0, 1, 2, 1, 2]
      values = [
        {10.0, 0.0}, {2.0, 0.0},
        {3.0, 0.0}, {9.0, 0.0}, {4.0, 0.0},
        {1.0, 0.0}, {7.0, 0.0}
      ]

      rhs = [{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}]

      {:ok, solution} = PowerFlowSolver.SparseLinearAlgebra.solve_csr(
        row_ptrs,
        col_indices,
        values,
        rhs
      )
  """

  version = Mix.Project.config()[:version]

  use RustlerPrecompiled,
    otp_app: :power_flow_solver,
    crate: "power_flow_solver",
    base_url: "https://github.com/voltageAI/power_flow_solver/releases/download/v#{version}",
    force_build: System.get_env("POWER_FLOW_FORCE_BUILD") in ["1", "true"],
    version: version,
    targets: ~w(
      aarch64-apple-darwin
      x86_64-apple-darwin
      x86_64-unknown-linux-gnu
      x86_64-unknown-linux-musl
    )

  @doc """
  Solves a sparse linear system Ax = b using LU factorization.

  The matrix A is provided in Compressed Sparse Row (CSR) format with complex values.

  ## Arguments

  - `row_ptrs` - Row pointer array indicating where each row starts in the values array
  - `col_indices` - Column index for each non-zero value
  - `values` - List of complex numbers as tuples `{real, imag}` for non-zero entries
  - `rhs` - Right-hand side vector as list of complex numbers `{real, imag}`

  ## Returns

  - `{:ok, solution}` - Solution vector as list of complex numbers
  - `{:error, reason}` - Error message if solve fails

  ## Example

      row_ptrs = [0, 2, 4, 6]
      col_indices = [0, 1, 1, 2, 0, 2]
      values = [{4.0, 0.0}, {1.0, 0.0}, {3.0, 0.0}, {2.0, 0.0}, {1.0, 0.0}, {5.0, 0.0}]
      rhs = [{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}]

      {:ok, x} = solve_csr(row_ptrs, col_indices, values, rhs)
  """
  def solve_csr(_row_ptrs, _col_indices, _values, _rhs),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Performs sparse matrix-vector multiplication: y = A * x

  ## Arguments

  - `row_ptrs` - Row pointer array for CSR format
  - `col_indices` - Column indices for CSR format
  - `values` - Non-zero values as complex tuples `{real, imag}`
  - `x` - Input vector as list of complex numbers

  ## Returns

  - `{:ok, y}` - Result vector
  - `{:error, reason}` - Error message
  """
  def sparse_mv(_row_ptrs, _col_indices, _values, _x),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Validates the analytical Jacobian against numerical finite differences.

  This function compares the analytically computed Jacobian matrix with a
  numerically computed version using central finite differences. This helps
  identify errors in the analytical formulas.

  ## Arguments

  - `buses` - List of bus data tuples: {type, p_sched, q_sched, v_sched, q_min, q_max, q_load}
  - `y_bus_data` - Y-bus matrix as {row_ptrs, col_indices, values}
  - `voltage` - Current voltage as list of {magnitude, angle} tuples
  - `epsilon` - Finite difference step size (default: 1.0e-7)

  ## Returns

  - `{:ok, {max_error, avg_error, num_errors, error_details}}` - Validation results
  - `{:error, reason}` - Error message if validation fails

  ## Example

      buses = [{2, 0.0, 0.0, 1.0, nil, nil, 0.0}, ...]
      y_bus = {row_ptrs, col_indices, values}
      voltage = [{1.0, 0.0}, {0.98, -0.05}, ...]

      {:ok, {max_err, avg_err, num_errs, details}} =
        validate_jacobian_rust(buses, y_bus, voltage, 1.0e-7)

      IO.puts("Max relative error: \#{max_err * 100}%")
      IO.puts("Avg relative error: \#{avg_err * 100}%")
      IO.puts("Number of large errors: \#{num_errs}")
  """
  def validate_jacobian_rust(_buses, _y_bus_data, _voltage, _epsilon \\ 1.0e-7),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Computes LU factorization of a sparse matrix.

  Returns factors that can be reused for multiple solves with different right-hand sides.

  ## Arguments

  - `row_ptrs` - Row pointer array for CSR format
  - `col_indices` - Column indices for CSR format
  - `values` - Non-zero values as complex tuples

  ## Returns

  - `{:ok, factors}` - Opaque reference to factorization (for future use)
  - `{:error, reason}` - Error message

  **Deprecated:** Use `create_symbolic_lu/2` and `factorize_with_symbolic/4` instead.
  """
  def lu_factorize(_row_ptrs, _col_indices, _values),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Creates a symbolic LU factorization from matrix structure.

  This performs expensive symbolic analysis once, which can be reused for
  matrices with the same sparsity pattern but different values. This is
  especially useful in Newton-Raphson iterations where the Jacobian pattern
  remains constant.

  ## Arguments

  - `row_ptrs` - Row pointer array for CSR format
  - `col_indices` - Column indices for CSR format

  ## Returns

  - `{:ok, symbolic}` - Opaque reference to symbolic factorization
  - `{:error, reason}` - Error if symbolic factorization fails

  ## Example

      # Create symbolic factorization once
      {:ok, symbolic} = create_symbolic_lu(row_ptrs, col_indices)

      # Use it for multiple numeric factorizations
      {:ok, lu1} = factorize_with_symbolic(symbolic, row_ptrs, col_indices, values1)
      {:ok, solution1} = solve_with_lu(lu1, rhs1)

      {:ok, lu2} = factorize_with_symbolic(symbolic, row_ptrs, col_indices, values2)
      {:ok, solution2} = solve_with_lu(lu2, rhs2)
  """
  def create_symbolic_lu(_row_ptrs, _col_indices),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Performs numeric LU factorization using pre-computed symbolic factorization.

  This is much faster than full factorization when the matrix pattern is unchanged,
  as it skips the expensive symbolic analysis step.

  ## Arguments

  - `symbolic` - Reference to symbolic factorization from `create_symbolic_lu/2`
  - `row_ptrs` - Row pointer array for CSR format
  - `col_indices` - Column indices for CSR format
  - `values` - Non-zero values as complex tuples

  ## Returns

  - `{:ok, lu}` - Opaque reference to complete LU factorization
  - `{:error, reason}` - Error if factorization fails

  ## Notes

  The `row_ptrs` and `col_indices` must match those used to create the symbolic
  factorization, or a dimension mismatch error will occur.
  """
  def factorize_with_symbolic(_symbolic, _row_ptrs, _col_indices, _values),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Solves a linear system using pre-computed LU factorization.

  This is very fast as it only performs forward/backward substitution.

  ## Arguments

  - `lu` - Reference to LU factorization from `factorize_with_symbolic/4`
  - `rhs` - Right-hand side vector as list of complex tuples

  ## Returns

  - `{:ok, solution}` - Solution vector
  - `{:error, reason}` - Error if solve fails

  ## Example

      {:ok, lu} = factorize_with_symbolic(symbolic, row_ptrs, col_indices, values)

      # Solve with multiple RHS vectors using same factorization
      {:ok, x1} = solve_with_lu(lu, rhs1)
      {:ok, x2} = solve_with_lu(lu, rhs2)
      {:ok, x3} = solve_with_lu(lu, rhs3)
  """
  def solve_with_lu(_lu, _rhs),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Solves multiple linear systems using pre-computed LU factorization.

  Solves AX = B where B has multiple columns. This is more efficient than
  calling `solve_with_lu/2` multiple times.

  ## Arguments

  - `lu` - Reference to LU factorization
  - `rhs_list` - List of RHS vectors

  ## Returns

  - `{:ok, solutions}` - List of solution vectors
  - `{:error, reason}` - Error if solve fails

  ## Example

      rhs_list = [rhs1, rhs2, rhs3]
      {:ok, [x1, x2, x3]} = solve_multiple_with_lu(lu, rhs_list)
  """
  def solve_multiple_with_lu(_lu, _rhs_list),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Builds the Jacobian matrix for Newton-Raphson power flow using Rust NIF.

  This function is 10-15x faster than the Elixir implementation due to:
  - Compiled Rust code with LLVM optimizations
  - Parallel row building using rayon
  - Efficient memory access patterns
  - SIMD-friendly operations

  ## Arguments

  - `y_row_ptrs` - Y-bus row pointers (CSR format)
  - `y_col_indices` - Y-bus column indices (CSR format)
  - `y_values` - Y-bus complex values as `{real, imag}` tuples
  - `voltage` - Voltage vector as `{magnitude, angle}` tuples
  - `bus_types` - List of bus types (0=slack, 1=pv, 2=pq)
  - `angle_vars` - Indices of angle variables
  - `vmag_vars` - Indices of voltage magnitude variables

  ## Returns

  - `{:ok, {row_ptrs, col_indices, values}}` - Jacobian in CSR format
  - `{:error, reason}` - Error description

  ## Example

      {:ok, {j_row_ptrs, j_col_indices, j_values}} =
        build_jacobian_rust(y_row_ptrs, y_col_indices, y_values,
                           voltage, bus_types, angle_vars, vmag_vars)
  """
  def build_jacobian_rust(_y_row_ptrs, _y_col_indices, _y_values, _voltage, _bus_types, _angle_vars, _vmag_vars),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Calculate reactive power injection at a specific bus using Rust NIF.

  This is significantly faster than the Elixir implementation because:
  - Direct memory access to Y-bus and voltage arrays
  - f64 arithmetic instead of Decimal
  - No intermediate allocations
  - Compiled to native code with optimizations

  ## Arguments

  - `row_ptrs` - CSR row pointers for Y-bus
  - `col_indices` - CSR column indices for Y-bus
  - `values` - Y-bus admittance values as `{real, imag}` tuples
  - `voltage` - Voltage vector as `{magnitude, angle}` tuples
  - `bus_idx` - Index of bus to calculate Q for

  ## Returns

  - `{:ok, q_injection}` - Reactive power injection value
  - `{:error, reason}` - Error description

  ## Example

      {:ok, q} = calculate_q_injection_rust(
        y_bus.row_ptrs,
        y_bus.col_indices,
        y_bus.values,
        voltage,
        bus_idx
      )
  """
  def calculate_q_injection_rust(_row_ptrs, _col_indices, _values, _voltage, _bus_idx),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Calculate reactive power injection for multiple buses at once.

  This is more efficient than calling `calculate_q_injection_rust/5` multiple
  times because it minimizes Elixir-Rust boundary crossings.

  ## Arguments

  - `row_ptrs` - CSR row pointers for Y-bus
  - `col_indices` - CSR column indices for Y-bus
  - `values` - Y-bus admittance values as `{real, imag}` tuples
  - `voltage` - Voltage vector as `{magnitude, angle}` tuples
  - `bus_indices` - List of bus indices to calculate Q for

  ## Returns

  - `{:ok, results}` - List of `{bus_idx, q_injection}` tuples
  - `{:error, reason}` - Error description

  ## Example

      {:ok, q_values} = calculate_q_injection_batch_rust(
        y_bus.row_ptrs,
        y_bus.col_indices,
        y_bus.values,
        voltage,
        [5, 10, 15, 20]
      )

      # Convert to map
      q_map = Map.new(q_values)
  """
  def calculate_q_injection_batch_rust(_row_ptrs, _col_indices, _values, _voltage, _bus_indices),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Create a voltage resource that stays in Rust memory.

  This is a **key optimization** that eliminates repeated serialization
  between Elixir and Rust. The voltage array stays in Rust memory and
  can be used for multiple Q calculations without crossing the boundary.

  ## Why This Matters

  **Before** (slow):
  - Every Q calculation: Elixir map → Rust conversion
  - 40 iterations × 374 buses = 14,960 conversions
  - Each conversion allocates memory and performs hash lookups

  **After** (fast):
  - Create resource ONCE
  - Q calculations use direct Rust array (no conversion!)
  - 1 conversion instead of 14,960 ✅

  ## Arguments

  - `voltage` - Voltage vector as `{magnitude, angle}` tuples

  ## Returns

  - `{:ok, resource}` - Opaque Rust resource handle
  - `{:error, reason}` - Error description

  ## Example

      voltage = [{1.0, 0.0}, {1.05, -0.1}, {1.02, -0.05}]
      {:ok, voltage_res} = create_voltage_resource(voltage)

      # Use resource for multiple Q calculations (fast!)
      {:ok, q_values} = calculate_q_batch_from_resource(
        row_ptrs, col_indices, values,
        voltage_res,  # <- Stays in Rust!
        [0, 1, 2]
      )
  """
  def create_voltage_resource(_voltage),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Calculate Q injection for multiple buses using a voltage resource.

  This avoids repeated voltage serialization by using a voltage resource
  that was previously created with `create_voltage_resource/1`.

  **Performance**: 5-10x faster than `calculate_q_injection_batch_rust/5`
  because voltage stays in Rust memory.

  ## Arguments

  - `row_ptrs` - CSR row pointers for Y-bus
  - `col_indices` - CSR column indices for Y-bus
  - `values` - Y-bus admittance values as `{real, imag}` tuples
  - `voltage_res` - Voltage resource (from `create_voltage_resource/1`)
  - `bus_indices` - List of bus indices to calculate Q for

  ## Returns

  - `{:ok, results}` - List of `{bus_idx, q_injection}` tuples
  - `{:error, reason}` - Error description

  ## Example

      # Create voltage resource once
      {:ok, voltage_res} = create_voltage_resource(voltage)

      # Use it multiple times (no repeated serialization!)
      {:ok, q_values_1} = calculate_q_batch_from_resource(
        row_ptrs, col_indices, values, voltage_res, [0, 1, 2]
      )

      {:ok, q_values_2} = calculate_q_batch_from_resource(
        row_ptrs, col_indices, values, voltage_res, [3, 4, 5]
      )

      # Resource automatically freed when no longer referenced
  """
  def calculate_q_batch_from_resource(_row_ptrs, _col_indices, _values, _voltage_res, _bus_indices),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Solve power flow using complete Rust Newton-Raphson implementation.

  This NIF runs the entire iteration loop in Rust, eliminating boundary
  crossings with Elixir for maximum performance.

  ## Arguments

  - `buses` - List of bus data tuples: [{type, p_sched, q_sched, v_sched}, ...]
    - type: 0=slack, 1=PV, 2=PQ
  - `y_bus_data` - Y-bus matrix as {row_ptrs, col_indices, values}
  - `initial_voltage` - Initial voltage as [{mag, ang}, ...]
  - `max_iterations` - Maximum iterations
  - `tolerance` - Convergence tolerance

  ## Returns

  - `{:ok, voltage, iterations, converged, final_mismatch}`
  - `{:error, reason}`

  ## Example

      buses = [{0, 0.0, 0.0, 1.05}, {1, 0.5, 0.0, 1.0}, ...]
      y_bus_data = {row_ptrs, col_indices, values}
      initial_v = [{1.0, 0.0}, {1.0, 0.0}, ...]

      {:ok, final_v, iters, true, mismatch} =
        solve_power_flow_rust(buses, y_bus_data, initial_v, 100, 1.0e-2)
  """
  def solve_power_flow_rust(
        _buses,
        _y_bus_data,
        _initial_voltage,
        _max_iterations,
        _tolerance,
        _enforce_q_limits,
        _q_tolerance
      ),
      do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Validates Jacobian matrix using numerical differentiation.

  Compares analytical Jacobian against numerical derivatives computed
  via finite differences to identify formula errors.

  ## Arguments

  - `buses` - List of bus data tuples: `{type, p_sched, q_sched, v_sched, q_min, q_max, q_load}`
  - `y_bus_data` - Y-bus matrix in CSR format: `{row_ptrs, col_indices, values}`
  - `voltage` - Current voltage as `{magnitude, angle}` tuples
  - `epsilon` - Finite difference step size (default: 1.0e-7)

  ## Returns

  - `{:ok, {max_error, avg_error, num_large_errors, error_details}}` - Validation results
  - `{:error, reason}` - Error message if validation fails

  ## Example

      buses = [{2, 0.5, 0.2, 1.0, nil, nil, 0.0}, ...]
      y_bus = {row_ptrs, col_indices, values}
      voltage = [{1.0, 0.0}, {0.98, -0.05}, ...]

      {:ok, {max_err, avg_err, num_err, details}} =
        validate_jacobian_rust(buses, y_bus, voltage, 1.0e-7)
  """
  def validate_jacobian_rust(_buses, _y_bus_data, _voltage, _epsilon),
    do: :erlang.nif_error(:nif_not_loaded)
end
