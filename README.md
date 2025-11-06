# PowerFlowSolver

High-performance power flow solver using Rust NIFs for Elixir. Optimized for large-scale power systems (10,000+ buses) with complete Newton-Raphson implementation in Rust.

## Features

- **🚀 High Performance**: Complete solver loop in Rust eliminates Elixir/Rust boundary crossings
- **📊 Scalable**: Handles systems from 10 to 30,000+ buses efficiently
- **🔧 Sparse Operations**: O(nnz) complexity using optimized sparse linear algebra
- **⚡ Parallel Execution**: Multi-threaded Jacobian building with rayon
- **🎯 Realistic Modeling**: Q-limit enforcement for generator reactive power constraints
- **📦 Pre-compiled**: Fast installation with no Rust toolchain required

## Installation

Add to your `mix.exs`:

```elixir
def deps do
  [
    {:power_flow_solver, git: "https://github.com/voltageAI/power_flow_solver.git", tag: "v0.1.0"}
  ]
end
```

### Building from Source

If you prefer to build from source or pre-compiled binaries aren't available for your platform:

```bash
export POWER_FLOW_FORCE_BUILD=1
mix deps.get
```

Requires:
- Rust 1.70+
- C compiler (gcc/clang)
- OpenBLAS (for linear algebra)

## Quick Start

```elixir
# Define a 2-bus system
system = %{
  buses: [
    %{
      id: 0,
      type: :slack,
      p_load: 0.0,
      q_load: 0.0,
      p_gen: 0.0,
      q_gen: 0.0,
      v_magnitude: 1.0,
      v_angle: 0.0
    },
    %{
      id: 1,
      type: :pq,
      p_load: 0.5,      # 0.5 p.u. real power load
      q_load: 0.2,      # 0.2 p.u. reactive power load
      p_gen: 0.0,
      q_gen: 0.0,
      v_magnitude: 1.0,  # Initial guess
      v_angle: 0.0       # Initial guess
    }
  ],
  y_bus: %{
    row_ptrs: [0, 2, 4],
    col_indices: [0, 1, 0, 1],
    values: [
      {10.0, -50.0},   # Y[0,0]
      {-10.0, 50.0},   # Y[0,1]
      {-10.0, 50.0},   # Y[1,0]
      {10.0, -50.0}    # Y[1,1]
    ]
  }
}

# Solve
{:ok, solution, iterations} = PowerFlowSolver.solve(system,
  max_iterations: 100,
  tolerance: 1.0e-2
)

IO.puts("Converged in #{iterations} iterations")
# solution contains voltage magnitude and angle for each bus
```

## System Format

### Bus Data

Each bus requires:

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `id` | integer | Unique bus identifier (0-indexed) | Yes |
| `type` | atom | `:slack`, `:pv`, or `:pq` | Yes |
| `p_load` | float | Real power load (per-unit) | Yes |
| `q_load` | float | Reactive power load (per-unit) | Yes |
| `p_gen` | float | Real power generation (per-unit) | Yes |
| `q_gen` | float | Reactive power generation (per-unit) | Yes |
| `v_magnitude` | float | Voltage magnitude (per-unit) | Yes |
| `v_angle` | float | Voltage angle (radians) | Yes |
| `q_min` | float | Min reactive power limit (per-unit) | No |
| `q_max` | float | Max reactive power limit (per-unit) | No |

**Bus Types:**
- **`:slack`** - Reference bus with fixed voltage magnitude and angle
- **`:pv`** - Generator bus with fixed real power and voltage magnitude
- **`:pq`** - Load bus with fixed real and reactive power

### Y-Bus Matrix

The admittance matrix in Compressed Sparse Row (CSR) format:

```elixir
y_bus: %{
  row_ptrs: [0, 3, 6, 9],      # Pointers to start of each row
  col_indices: [0, 1, 2, ...],  # Column index for each non-zero
  values: [                     # Complex values as {real, imag} tuples
    {g, b},                     # Conductance + susceptance
    ...
  ]
}
```

## Options

```elixir
PowerFlowSolver.solve(system, options)
```

| Option | Default | Description |
|--------|---------|-------------|
| `max_iterations` | 100 | Maximum Newton-Raphson iterations |
| `tolerance` | 1.0e-2 | Convergence tolerance (p.u.) |
| `initial_voltage` | nil | Custom initial voltage vector |
| `enforce_q_limits` | false | Enable Q-limit enforcement |
| `q_tolerance` | 1.0e-4 | Q-limit tolerance (p.u.) |

## Performance

Benchmarks on M1 MacBook Pro (2021):

| System Size | Iterations | Time | Memory |
|-------------|-----------|------|---------|
| 100 buses | 5 | 5ms | <10MB |
| 1,000 buses | 6 | 50ms | <50MB |
| 10,000 buses | 7 | 1.5s | ~200MB |
| 30,000 buses | 8 | 8s | ~800MB |

Performance characteristics:
- **Linear scaling** with number of non-zeros in Y-bus
- **Parallel Jacobian** building scales with CPU cores
- **Memory efficient** sparse representation
- **Zero GC pressure** during iteration (all in Rust)

## Advanced Usage

### Custom Initial Voltage

Provide better initial guess (e.g., from DC power flow):

```elixir
initial_voltage = [
  {1.0, 0.0},      # Bus 0: 1.0 p.u. at 0 radians
  {0.98, -0.05},   # Bus 1: 0.98 p.u. at -0.05 radians
  ...
]

PowerFlowSolver.solve(system, initial_voltage: initial_voltage)
```

### Q-Limit Enforcement

Enable reactive power limit enforcement:

```elixir
# Add Q limits to PV buses
buses = [
  %{
    id: 1,
    type: :pv,
    p_gen: 1.0,
    q_gen: 0.3,
    q_min: -0.5,    # Minimum reactive power
    q_max: 1.5,     # Maximum reactive power
    ...
  },
  ...
]

PowerFlowSolver.solve(system,
  enforce_q_limits: true,
  q_tolerance: 1.0e-4
)
```

When Q limits are hit, PV buses are temporarily converted to PQ buses.

### Low-Level Sparse Operations

Direct access to sparse linear algebra:

```elixir
alias PowerFlowSolver.SparseLinearAlgebra

# Solve Ax = b
{:ok, x} = SparseLinearAlgebra.solve_csr(
  row_ptrs,
  col_indices,
  values,
  rhs
)

# Sparse matrix-vector multiply
{:ok, y} = SparseLinearAlgebra.sparse_mv(
  row_ptrs,
  col_indices,
  values,
  x
)

# LU factorization (reusable for multiple solves)
{:ok, lu} = SparseLinearAlgebra.lu_factorize(
  row_ptrs,
  col_indices,
  values
)

{:ok, x} = SparseLinearAlgebra.solve_with_lu(lu, rhs)
```

## Architecture

```
PowerFlowSolver (Elixir)
├── NewtonRaphson          # High-level solver API
├── SparseLinearAlgebra    # NIF bindings
└── native/
    └── power_flow_solver/ # Rust implementation
        ├── lib.rs         # NIF exports
        └── power_flow.rs  # Newton-Raphson solver
```

**Key optimizations:**
- **Single boundary crossing**: System data goes to Rust, result comes back
- **Parallel Jacobian**: Uses rayon for multi-threaded matrix building
- **Sparse LU**: faer library with AMD ordering for optimal fill-in
- **SIMD operations**: Vectorized complex number arithmetic
- **Zero-copy**: Direct memory access where possible

## Limitations

- **AC power flow only**: No DC approximation
- **Balanced 3-phase**: Single-phase modeling
- **No distributed slack**: Single slack bus required
- **Fixed topology**: No switching during solve

## Contributing

This is a private library for Voltage AI projects. For issues or improvements, contact the development team.

## License

MIT License - see [LICENSE](LICENSE) for details

Copyright (c) 2024 Voltage AI
