#!/usr/bin/env elixir

# Verify what tap PowerWorld uses by reverse-engineering from CSV

require Logger

# From CSV for Bus 4→5 transformer connection
# CSV shows: Y[3][4] = 0.0 + j10.32 (off-diagonal)
#            Y[4][4] = 0.0 - j10.0  (diagonal at Bus 5)

# Transformer parameters
r = 1.0e-7
x = 0.1

# Series admittance
z_mag_sq = r * r + x * x
g = r / z_mag_sq
b_series = -x / z_mag_sq

Logger.info("\nTransformer Bus 4→5 (138kV/34.5kV)")
Logger.info("R = #{r}, X = #{x}")
Logger.info("Series admittance: G = #{g}, B = #{b_series}")

# CSV values
csv_y_33_diag = -10.0  # Y[3][3] contribution from this transformer
csv_y_44_diag = -10.0  # Y[4][4] contribution from this transformer
csv_y_34_off = 10.32   # Y[3][4] off-diagonal (imaginary part)

Logger.info("\nCSV Y-bus values:")
Logger.info("  Y[3][3] contribution: -j#{-csv_y_33_diag}")
Logger.info("  Y[4][4] contribution: -j#{-csv_y_44_diag}")
Logger.info("  Y[3][4] off-diagonal: +j#{csv_y_34_off}")

# Test different tap scenarios
scenarios = [
  %{name: "No tap (tap=1.0)", tap: 1.0},
  %{name: "Only off-nominal (tap=0.96875)", tap: 0.96875},
  %{name: "Only voltage ratio (tap=0.25)", tap: 0.25},
  %{name: "Combined (tap=0.2421875)", tap: 0.2421875}
]

Logger.info("\n" <> String.duplicate("=", 80))
Logger.info("TESTING DIFFERENT TAP SCENARIOS")
Logger.info(String.duplicate("=", 80))

Enum.each(scenarios, fn scenario ->
  tap = scenario.tap
  t_sq = tap * tap

  # Transformer model with this tap
  y_ii_b = b_series / t_sq  # Bus 3 (from) diagonal susceptance
  y_jj_b = b_series          # Bus 4 (to) diagonal susceptance
  y_ij_b = -b_series / tap   # Off-diagonal susceptance

  Logger.info("\n#{scenario.name}:")
  Logger.info("  Y[3][3] = -j#{Float.round(-y_ii_b, 2)} (CSV expects -j#{-csv_y_33_diag})")
  Logger.info("  Y[4][4] = -j#{Float.round(-y_jj_b, 2)} (CSV expects -j#{-csv_y_44_diag})")
  Logger.info("  Y[3][4] = +j#{Float.round(-y_ij_b, 2)} (CSV expects +j#{csv_y_34_off})")

  # Check if it matches
  match_34 = abs(abs(y_ij_b) - csv_y_34_off) < 0.5
  match_44 = abs(abs(y_jj_b) - abs(csv_y_44_diag)) < 0.1

  if match_34 and match_44 do
    Logger.info("  ✓ ✓ ✓ MATCHES CSV! ✓ ✓ ✓")
  end
end)

Logger.info("\n" <> String.duplicate("=", 80))
Logger.info("CONCLUSION")
Logger.info(String.duplicate("=", 80))
Logger.info("""
If tap=1.0 matches, then PowerWorld uses:
  - Flat per-unit system
  - Each bus has its own voltage base
  - NO tap ratios in Y-bus (not even off-nominal taps)
  - Transformation handled by different per-unit bases

If tap=0.96875 matches, then PowerWorld:
  - Uses off-nominal tap in Y-bus
  - But NOT the voltage transformation ratio
  - Only the tap changer position matters
""")
