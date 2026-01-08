# Power Flow Solver - Input Data Structure

## Overview

The solver requires a `system` map with two main components: **buses** and **lines** (or branches/transformers).

## Basic Structure

```elixir
system = %{
  buses: [bus1, bus2, ...],
  lines: [line1, line2, ...]
}
```

## Bus Structure

Each bus is a map with the following fields:

### Required Fields

| Field | Type | Description | Values |
|-------|------|-------------|--------|
| `id` | integer | Bus number (0-indexed) | 0, 1, 2, ... |
| `type` | atom | Bus type | `:slack`, `:pv`, or `:pq` |
| `p_gen` | float | Active power generation (pu) | Any float |
| `p_load` | float | Active power load (pu) | Any float |
| `q_gen` | float | Reactive power generation (pu) | Any float |
| `q_load` | float | Reactive power load (pu) | Any float |
| `v_magnitude` | float | Voltage magnitude (pu) | Usually 0.9-1.1 |
| `v_angle` | float | Voltage angle (radians) | Any float |

### Optional Fields

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `q_min` | float/nil | Min reactive power (pu) | nil |
| `q_max` | float/nil | Max reactive power (pu) | nil |
| `base_kv` | float | Base voltage (kV) | N/A |

### Bus Types

- **`:slack`** - Reference bus (voltage magnitude and angle fixed)
- **`:pv`** - Generator bus (voltage magnitude and P fixed, Q calculated)
- **`:pq`** - Load bus (P and Q fixed, voltage calculated)

### Example Bus

```elixir
bus = %{
  id: 0,
  type: :slack,
  p_gen: 1.03056,
  p_load: 1.0,
  q_gen: 0.39759,
  q_load: 0.0,
  v_magnitude: 1.0,
  v_angle: 0.0,
  q_min: -0.5,    # Optional
  q_max: 2.0,     # Optional
  base_kv: 138.0  # Optional
}
```

## Line/Branch Structure

Each line/branch/transformer is a map with these fields:

### Required Fields

| Field | Type | Description | Values |
|-------|------|-------------|--------|
| `from` | integer | From bus ID (0-indexed) | 0, 1, 2, ... |
| `to` | integer | To bus ID (0-indexed) | 0, 1, 2, ... |
| `r` | float | Resistance (pu) | > 0 |
| `x` | float | Reactance (pu) | > 0 |

### Optional Fields

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `b` | float | Total line charging susceptance (pu) | 0.0 |
| `tap` | float | Transformer off-nominal tap ratio | 1.0 |
| `tap_ratio` | float | Alternative field for tap | 1.0 |
| `shift` | float | Phase shift angle (radians) | 0.0 |
| `phase_shift` | float | Alternative field for shift | 0.0 |

### Line vs Transformer

**Regular Line:**
```elixir
line = %{
  from: 0,
  to: 1,
  r: 0.03,
  x: 0.16,
  b: 0.037,  # Line charging
  tap: 1.0   # No transformation
}
```

**Transformer:**
```elixir
transformer = %{
  from: 1,
  to: 3,
  r: 0.015,
  x: 0.08,
  b: 0.0,      # No line charging
  tap: 0.96875 # Off-nominal tap (PowerWorld convention)
}
```

## Complete Example

```elixir
system = %{
  buses: [
    %{
      id: 0,
      type: :slack,
      p_gen: 1.03056,
      p_load: 1.0,
      q_gen: 0.39759,
      q_load: 0.0,
      v_magnitude: 1.0,
      v_angle: 0.0
    },
    %{
      id: 1,
      type: :pq,
      p_gen: 0.0,
      p_load: 2.0,
      q_gen: 0.0,
      q_load: 0.3511,
      v_magnitude: 1.0,
      v_angle: 0.05646  # ~3.23 degrees
    }
  ],
  lines: [
    %{
      from: 0,
      to: 1,
      r: 0.03,
      x: 0.16,
      b: 0.037
    }
  ]
}
```

## Usage

```elixir
# Parse from file
{:ok, {buses, lines, _shunts, _metadata}} =
  PowerSystemParsers.PsseRawParser.parse_file("case.RAW")

system = %{buses: buses, lines: lines}

# Solve
{:ok, solution, iterations} = PowerFlowSolver.NewtonRaphson.solve(
  system,
  max_iterations: 100,
  tolerance: 1.0e-6,
  enforce_q_limits: false
)

# Access results
Enum.each(buses, fn bus ->
  {v_mag, v_angle} = Map.get(solution, bus.id)
  v_angle_deg = v_angle * 180.0 / :math.pi()
  IO.puts("Bus #{bus.id}: V=#{v_mag} pu ∠ #{v_angle_deg}°")
end)
```

## Solver Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_iterations` | integer | 100 | Maximum Newton-Raphson iterations |
| `tolerance` | float | 1.0e-2 | Convergence tolerance (pu) |
| `enforce_q_limits` | boolean | false | Enforce generator Q limits |
| `q_tolerance` | float | 1.0e-4 | Q-limit violation tolerance (pu) |
| `initial_voltage` | map/nil | nil | Custom initial voltages (advanced) |

## Return Values

### Success
```elixir
{:ok, solution, iterations}
```

- `solution` - Map of `bus_id => {v_magnitude, v_angle_radians}`
- `iterations` - Number of iterations to converge

### Failure
```elixir
{:error, reason}
```

- `reason` - String describing why solver failed

## Important Notes

### Per-Unit Convention
- **Base MVA**: 100 MVA (standard)
- **Voltage bases**: Each bus has its own voltage base (from `base_kv` field)
- **Tap ratios**: Off-nominal tap only (PowerWorld convention)
  - Voltage transformation handled by different per-unit bases
  - Example: 138kV/34.5kV transformer with 96.875% tap → `tap: 0.96875`

### Angle Units
- **Input**: Radians (parser converts from degrees)
- **Output**: Radians (convert to degrees: `angle_deg = angle_rad * 180.0 / :math.pi()`)

### Bus Numbering
- **0-indexed**: Bus IDs start at 0
- **Sequential**: Bus IDs should be 0, 1, 2, ..., N-1
- **No gaps**: Don't skip numbers in the sequence

### Power Sign Convention
- **Generation**: Positive
- **Load**: Positive
- **Net injection**: `p_gen - p_load`

## Validation Rules

1. **At least one slack bus** required
2. **R and X must be positive** for all lines
3. **Bus IDs must be sequential** (0, 1, 2, ...)
4. **From/To bus IDs must exist** in buses list
5. **Voltage angles** in radians
6. **All power values** in per-unit on 100 MVA base
