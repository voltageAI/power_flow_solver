# Y-Bus PowerWorld Compatibility Fix - Summary

## Problem
Y-bus built by solver didn't match PowerWorld-generated CSV, with huge differences at transformer buses:
- Bus 4: Solver had Y[3][3] = 2.63 - j184.50, CSV expected 4.53 - j34.81
- Difference of -j149.69 (150× off!)

## Root Cause
**Parser was using wrong tap ratio formula for PowerWorld per-unit convention**

### PowerWorld Convention (Correct)
- Each bus has its own voltage base (138kV, 34.5kV, etc.)
- Tap ratio = **off-nominal tap only** (windv1/windv2)
- Voltage transformation handled by different per-unit bases
- For 138kV/34.5kV transformer with windv1=0.96875, windv2=1.0:
  - tap = 0.96875 / 1.0 = **0.96875**

### Previous Implementation (Incorrect for PowerWorld)
- Tap ratio = **combined** (off-nominal × voltage transformation)
- Formula: tap = (windv1 × nomv2) / (windv2 × nomv1)
- For same transformer:
  - tap = (0.96875 × 34.5) / (1.0 × 138.0) = **0.2421875**
- This is correct for "physical" model but wrong for PowerWorld's per-unit convention

## The Fix

### Changed in: `power_system_parsers` v0.3.0
**File:** `lib/power_system_parsers/psse_raw_parser.ex`
**Lines:** 848-853

**Before:**
```elixir
tap = (windv1 * nomv2) / (windv2 * nomv1)
```

**After:**
```elixir
tap = windv1 / windv2
```

**Comment added:**
```elixir
# In per-unit power flow models, voltage transformation is handled by
# having different voltage bases at each bus (NOMV1 vs NOMV2).
# The tap ratio should only reflect the off-nominal tap position.
```

## Results

### Before Fix
```
Y-bus matrices DIFFER
Max difference (real): 11.883059
Max difference (imag): 149.686621
```

### After Fix
```
✓ Y-bus matrices MATCH (within tolerance of 0.01)
Max difference (real): 0.004191
Max difference (imag): 0.004708
```

**Improvement:** From 150 pu difference to 0.005 pu difference! ✓

## Verification

### Bus 4→5 Transformer (138kV/34.5kV, windv1=0.96875)

| Component | Before | After | CSV | Match? |
|-----------|--------|-------|-----|--------|
| Parser tap | 0.2421875 | **0.96875** | - | ✓ |
| Y[3][3] | 2.63 - j184.50 | **4.53 - j34.81** | 4.53 - j34.81 | ✓ |
| Y[3][4] | -j41.29 | **-j10.32** | -j10.32 | ✓ |
| Y[4][4] | -j10.00 | **-j10.00** | -j10.00 | ✓ |

All values now match PowerWorld CSV exactly!

## Impact

### What Changed
- Parser now uses PowerWorld/PSS/E flat per-unit convention
- Transformer tap ratios no longer include voltage transformation
- Voltage transformation handled by different voltage bases at each bus

### What Stayed the Same
- Solver Y-bus building logic unchanged
- All other components unchanged
- API/interface unchanged

### Compatibility
- ✅ PowerWorld Y-bus exports
- ✅ PSS/E flat per-unit models
- ✅ Standard industry per-unit convention
- ❌ "Physical" transformer models (would need different parser)

## Testing Performed
1. ✅ Y-bus comparison with PowerWorld CSV
2. ✅ All transformer tap ratios verified
3. ✅ Diagonal and off-diagonal elements checked
4. ✅ All 6 buses validated

## Conclusion
The fix successfully aligns the solver with PowerWorld's per-unit convention by using only the off-nominal tap ratio (windv1/windv2) instead of combining it with the voltage transformation ratio. All Y-bus elements now match the PowerWorld reference within numerical precision.
