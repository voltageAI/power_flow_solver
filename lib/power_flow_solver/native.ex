defmodule PowerFlowSolver.Native do
  @moduledoc """
  Native Rust NIF bindings for power flow calculations.

  This module uses RustlerPrecompiled to download precompiled binaries from
  GitHub Releases, falling back to local compilation only when necessary.

  ## Forcing Local Compilation

  To force building from source (requires Rust toolchain):

      export POWER_FLOW_SOLVER_BUILD=true
      mix deps.compile power_flow_solver --force

  ## Authentication for Private Repos

  When using this as a Git dependency from a private repo, you need to configure
  authentication for downloading the precompiled binaries:

  Option 1: Environment variable (recommended for CI):

      export POWER_FLOW_SOLVER_GITHUB_TOKEN=ghp_xxxxxxxxxxxx

  Option 2: Use `gh` CLI authentication (for local development):

      gh auth login

  The token needs `repo` scope to access private release assets.
  """

  version = Mix.Project.config()[:version]

  use RustlerPrecompiled,
    otp_app: :power_flow_solver,
    crate: "power_flow_solver",
    base_url: "https://github.com/voltageAI/power_flow_solver/releases/download/v#{version}",
    version: version,
    # Force build from source if env var is set
    force_build: System.get_env("POWER_FLOW_SOLVER_BUILD") in ["1", "true", "TRUE"],
    # Targets we build for:
    # - x86_64-unknown-linux-gnu: AWS deployment (standard x86 instances)
    # - aarch64-apple-darwin: Local dev on Mac M1/M2/M3
    # Note: Linux aarch64 and macOS x86_64 have cross-compilation issues with OpenBLAS
    targets: [
      "aarch64-apple-darwin",
      "x86_64-unknown-linux-gnu"
    ],
    # NIF version - use :erlang.system_info(:nif_version) to check yours
    nif_versions: ["2.16", "2.17"]

  # =============================================================================
  # NIF Function Stubs
  # These are replaced by the native implementation when the NIF loads
  # =============================================================================

  @doc """
  Solves a sparse linear system Ax = b using LU factorization.
  """
  def solve_csr(_row_ptrs, _col_indices, _values, _rhs),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Performs sparse matrix-vector multiplication: y = A * x
  """
  def sparse_mv(_row_ptrs, _col_indices, _values, _x),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Validates the analytical Jacobian against numerical finite differences.
  """
  def validate_jacobian_rust(_buses, _y_bus_data, _voltage, _epsilon \\ 1.0e-7),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Computes LU factorization of a sparse matrix.
  """
  def lu_factorize(_row_ptrs, _col_indices, _values),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Creates a symbolic LU factorization from matrix structure.
  """
  def create_symbolic_lu(_row_ptrs, _col_indices),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Performs numeric LU factorization using pre-computed symbolic factorization.
  """
  def factorize_with_symbolic(_symbolic, _row_ptrs, _col_indices, _values),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Solves a linear system using pre-computed LU factorization.
  """
  def solve_with_lu(_lu, _rhs),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Solves multiple linear systems using pre-computed LU factorization.
  """
  def solve_multiple_with_lu(_lu, _rhs_list),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Builds the Jacobian matrix for Newton-Raphson power flow.
  """
  def build_jacobian_rust(
        _y_row_ptrs,
        _y_col_indices,
        _y_values,
        _voltage,
        _bus_types,
        _angle_vars,
        _vmag_vars
      ),
      do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Calculate reactive power injection at a specific bus.
  """
  def calculate_q_injection_rust(_row_ptrs, _col_indices, _values, _voltage, _bus_idx),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Calculate reactive power injection for multiple buses at once.
  """
  def calculate_q_injection_batch_rust(_row_ptrs, _col_indices, _values, _voltage, _bus_indices),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Create a voltage resource that stays in Rust memory.
  """
  def create_voltage_resource(_voltage),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Calculate Q injection for multiple buses using a voltage resource.
  """
  def calculate_q_batch_from_resource(
        _row_ptrs,
        _col_indices,
        _values,
        _voltage_res,
        _bus_indices
      ),
      do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Solve power flow using complete Rust Newton-Raphson implementation.
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
  Calculate Short Circuit Ratio (SCR) for multiple plants.
  """
  def calculate_scr_batch_rust(_y_bus_data, _plants, _system_mva_base, _include_gen_reactance),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Get Thevenin impedances at all buses.
  """
  def get_thevenin_impedances_rust(_y_bus_data, _system_mva_base),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Invert Y-bus to get the full Z-bus matrix.
  """
  def invert_y_bus_rust(_y_bus_data),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Calculate SCR at a bus with a single branch contingency (N-1).
  """
  def calculate_contingency_scr_rust(_y_bus_data, _branch_data, _poi_bus_idx, _system_mva_base),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Calculate SCR for multiple branch contingencies in parallel.
  """
  def calculate_contingency_scr_batch_rust(
        _y_bus_data,
        _branches,
        _poi_bus_idx,
        _system_mva_base
      ),
      do: :erlang.nif_error(:nif_not_loaded)
end
