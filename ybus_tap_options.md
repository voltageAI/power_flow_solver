# Y-Bus Tap Ratio: Parser vs Solver Analysis

## Current Flow

### 1. PSS/E RAW File (Bus 4→5 transformer)
```
windv1 = 0.968750  (96.875% tap position)
nomv1  = 138.0 kV
windv2 = 1.000000  (100% tap position)
nomv2  = 34.5 kV
```

### 2. Parser Calculation
```elixir
tap = (windv1 * nomv2) / (windv2 * nomv1)
    = (0.96875 * 34.5) / (1.0 * 138.0)
    = 0.2421875
```

**Location:** `deps/power_system_parsers/lib/power_system_parsers/psse_raw_parser.ex:842-844`

### 3. Solver Y-Bus Calculation
```elixir
# Transformer model (when tap != 1.0)
Y_ii = y_series / tap²
Y_jj = y_series
Y_ij = -y_series / tap
```

**Location:** `lib/power_flow_solver/newton_raphson.ex:339-370`

### 4. Result
```
Y[3][3] += 0.0002 - j170.49  (huge susceptance due to tap²)
```

## If CSV is Correct (tap=1.0 model)

### Option 1: Change Parser
**Modify:** `power_system_parsers` library to set `tap = 1.0` for all transformers

```elixir
# Instead of:
tap = (windv1 * nomv2) / (windv2 * nomv1)

# Use:
tap = 1.0  # Always, for all transformers
```

**Impact:** Voltage transformation would need to be handled elsewhere (not in Y-bus)

### Option 2: Change Solver
**Modify:** `build_y_bus` function to ignore tap ratios

```elixir
# Line 339: Change condition
if tap != 1.0 or shift != 0.0 do
  # TRANSFORMER MODEL
  # ...

# To always use line model:
if false do  # Never use transformer model
  # ...
else
  # Always use regular LINE MODEL
  y_off = {-g, -b_series}
  y_diag = {g, b_series}
```

**Impact:** Tap ratios would be discarded; transformers treated as lines

### Option 3: Use Per-Unit System Differently
**Concept:** In a proper per-unit system, transformers might have tap=1.0

This would require:
1. Converting all buses to same base voltage (e.g., 138 kV)
2. Converting impedances to that common base
3. Tap ratios become 1.0 in the per-unit system

**Question:** Does CSV use a common base voltage?

## Key Questions to Answer

1. **What base voltage does the CSV assume?**
   - Single system base (e.g., 138 kV)?
   - Multiple voltage bases?

2. **How was the CSV Y-bus generated?**
   - PSS/E export command?
   - Custom calculation?
   - Different software?

3. **What convention should be used?**
   - Physical model (tap ratios in Y-bus) ← Current solver
   - Per-unit model (tap=1.0 in Y-bus) ← CSV expectation?

## Recommendation

Before changing code, we need to know:
- **Where did the CSV come from?**
- **What per-unit base was used?**
- **Is this the "correct" representation for your use case?**

The solver currently uses the **standard power system textbook model** for transformers.
The CSV uses what appears to be a **per-unit model with unified base**.

Both can be "correct" depending on the convention being used.
