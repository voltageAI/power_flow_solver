# PowerWorld vs Current Solver: Per-Unit Conventions

## The Difference

### PowerWorld (CSV Y-bus)
**Convention**: "Flat" per-unit with separate voltage bases

```
For a 138kV/34.5kV transformer with X=0.1 pu on 100 MVA base:
- Impedance at 138kV base: Z = 0.1 pu
- In Y-bus: Y = -j10.0 (tap ratio NOT applied)
- Voltage transformation handled by: using different voltage bases at each bus
```

**Result:** Y-bus has tap=1.0 for all transformers

### Current Solver
**Convention**: "Physical" transformer model

```
For same transformer:
- In Y-bus: Y_ii = y_series / tap²
- With tap = 0.2421875: Y_ii = -j10.0 / (0.2421875)² = -j170.5
```

**Result:** Y-bus includes tap ratio effects

## Why Both Are "Correct"

These are **two different but equivalent formulations**:

### PowerWorld Method
- Y-bus elements: Simple, tap=1.0
- Voltage bases: Different at each bus (138kV, 34.5kV, etc.)
- Tap handled in: Per-unit conversion and voltage calculations
- Used by: PowerWorld, some commercial software

### Physical Model (Current Solver)
- Y-bus elements: Include tap ratio effects
- Voltage bases: Can be unified or separate
- Tap handled in: Y-bus matrix elements
- Used by: Academic papers, some textbooks, PSS/E (sometimes)

## The Math (Why They're Equivalent)

For Bus 4→5 transformer (138kV / 34.5kV):

**PowerWorld approach:**
```
V_base_4 = 138 kV,  V_base_5 = 34.5 kV
Z_pu = 0.1 (on 100 MVA base)
Y = -j10.0 (no tap in Y-bus)
But V₄ and V₅ are in different pu bases!
```

**Physical model approach:**
```
V_base_4 = V_base_5 = same base (e.g., 138 kV)
Y_44 = -j10.0 / tap² = -j170.5
Y_54 = -j10.0 / tap = -j41.3
Now V₄ and V₅ are in same pu base
```

They give **identical physical results** when you account for the voltage base differences!

## What Needs to Change to Match PowerWorld

### Option 1: Change Parser (Recommended)
Modify `power_system_parsers` to NOT calculate tap from voltage ratios:

```elixir
# File: deps/power_system_parsers/lib/power_system_parsers/psse_raw_parser.ex
# Line 842-847

# Current (physical model):
tap = (windv1 * nomv2) / (windv2 * nomv1)

# Change to (PowerWorld/flat per-unit):
tap = windv1 / windv2  # Only off-nominal tap, ignore voltage transformation
```

For the 138/34.5kV transformer:
- Current: tap = (0.96875 × 34.5) / (1.0 × 138) = 0.2421875
- PowerWorld: tap = 0.96875 / 1.0 = 0.96875

But wait... that's still not 1.0! Let me check what PowerWorld really does...

### Option 2: Change Solver to Ignore Taps
Force all transformers to use line model:

```elixir
# File: lib/power_flow_solver/newton_raphson.ex
# Line 338-339

# Current:
if tap != 1.0 or shift != 0.0 do
  # TRANSFORMER MODEL

# Change to:
if shift != 0.0 do  # Only use transformer model for phase shifters
  # TRANSFORMER MODEL
else
  # Always LINE MODEL for voltage transformers
```

This treats all transformers as tap=1.0 regardless of parser output.

## Recommendation

Since your CSV came from PowerWorld, you should match PowerWorld's convention.

**Question:** Do you want your solver to:
1. **Match PowerWorld** (Y-bus with tap=1.0, different voltage bases)?
2. **Keep physical model** (Y-bus with tap ratios, unified voltage base)?

The choice depends on:
- Do you need to match PowerWorld results exactly?
- Will you exchange Y-bus data with PowerWorld?
- What convention do your users expect?

Let me know which approach you prefer, and I'll help implement it!
