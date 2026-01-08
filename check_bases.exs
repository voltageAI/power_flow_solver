#!/usr/bin/env elixir

# Check if impedances need to be on different voltage bases

require Logger

# From RAW file
buses_raw = [
  %{id: 1, name: "Bus 1", base_kv: 138.0},
  %{id: 2, name: "Bus 2", base_kv: 138.0},
  %{id: 3, name: "Bus 3", base_kv: 138.0},
  %{id: 4, name: "Bus 4", base_kv: 138.0},
  %{id: 5, name: "Bus 5", base_kv: 34.5},
  %{id: 6, name: "Bus 6", base_kv: 138.0}
]

Logger.info("\n" <> String.duplicate("=", 80))
Logger.info("VOLTAGE BASE ANALYSIS")
Logger.info(String.duplicate("=", 80))

Logger.info("\nBus voltage bases from RAW file:")
Enum.each(buses_raw, fn bus ->
  Logger.info("  Bus #{bus.id}: #{bus.base_kv} kV")
end)

Logger.info("\nWith PowerWorld's per-unit convention:")
Logger.info("  - Each bus uses its own voltage base")
Logger.info("  - Bus 5 is at 34.5 kV base")
Logger.info("  - All others at 138 kV base")
Logger.info("  - Impedances between buses must account for voltage base differences")

Logger.info("\nTransformer 4→5 (138kV → 34.5kV):")
Logger.info("  - In PowerWorld: tap = 0.96875 (off-nominal only)")
Logger.info("  - Voltage bases handle the 138/34.5 transformation")
Logger.info("  - Impedance on 100 MVA base:")

s_base = 100.0  # MVA
v_base_4 = 138.0  # kV
v_base_5 = 34.5   # kV

z_base_4 = (v_base_4 * v_base_4) / s_base
z_base_5 = (v_base_5 * v_base_5) / s_base

Logger.info("    Z_base at Bus 4 (138kV): #{Float.round(z_base_4, 2)} Ω")
Logger.info("    Z_base at Bus 5 (34.5kV): #{Float.round(z_base_5, 2)} Ω")
Logger.info("    Ratio: #{Float.round(z_base_4 / z_base_5, 2)}")

Logger.info("\n❓ QUESTION:")
Logger.info("Does our Y-bus use the same voltage base for all buses,")
Logger.info("or does it account for different bases?")

Logger.info("\nIf all buses use the SAME base (e.g., 138kV):")
Logger.info("  → We'd need to convert Bus 5 quantities to 138kV base")
Logger.info("  → This would change the impedance values")

Logger.info("\nIf each bus uses its OWN base (PowerWorld way):")
Logger.info("  → Voltages at Bus 5 are in per-unit of 34.5kV")
Logger.info("  → Voltages at Bus 4 are in per-unit of 138kV")
Logger.info("  → They can't be directly compared!")
Logger.info("  → Power flow equations need to account for this")

Logger.info("\n" <> String.duplicate("=", 80))
Logger.info("HYPOTHESIS")
Logger.info(String.duplicate("=", 80))
Logger.info("""
PowerWorld's "solved case" assumes:
  - Each bus has its own voltage base
  - V_pu values are relative to that base
  - Y-bus elements are formulated for multi-base system

Our solver might be assuming:
  - Single voltage base for all buses
  - V_pu values are all relative to same base
  - Y-bus uses single-base formulation

This would explain why it doesn't converge in 1 iteration!
""")
