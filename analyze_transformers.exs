#!/usr/bin/env elixir

# Analyze transformer modeling to identify discrepancies

require Logger

test_data_dir = Path.join(__DIR__, "test/test_data")
raw_file_path = Path.join(test_data_dir, "000_6bus_v33.RAW")

{:ok, {buses, lines, _shunts, _metadata}} =
  PowerSystemParsers.PsseRawParser.parse_file(raw_file_path)

Logger.info("\n" <> String.duplicate("=", 100))
Logger.info("TRANSFORMER ANALYSIS")
Logger.info(String.duplicate("=", 100))

Logger.info("\n--- ALL LINES/TRANSFORMERS FROM PARSER ---")

Enum.each(lines, fn line ->
  # Get tap and shift values
  tap_value = Map.get(line, :tap, Map.get(line, :tap_ratio, nil))
  shift_value = Map.get(line, :shift, Map.get(line, :phase_shift, 0.0))

  # Check if this is identified as a transformer
  is_xfmr = (tap_value != nil and tap_value != 1.0) or shift_value != 0.0
  type_str = if is_xfmr, do: "[TRANSFORMER]", else: "[LINE]"

  Logger.info("\n#{type_str} #{line.from} → #{line.to}")
  Logger.info("  ID: #{line.id}")
  Logger.info("  R: #{line.r}, X: #{line.x}")
  Logger.info("  Tap: #{inspect(tap_value)}")
  Logger.info("  Shift: #{inspect(shift_value)}")

  # Show all fields for debugging
  Logger.info("  All fields: #{inspect(Map.keys(line))}")
end)

Logger.info("\n" <> String.duplicate("=", 100))
Logger.info("EXPECTED TRANSFORMERS FROM RAW FILE")
Logger.info(String.duplicate("=", 100))

expected_xfmrs = [
  %{from: 1, to: 3, v1: 345.0, v2: 138.0, r: 0.015, x: 0.08},
  %{from: 1, to: 5, v1: 138.0, v2: 138.0, r: 1.0e-7, x: 0.05},
  %{from: 3, to: 2, v1: 138.0, v2: 345.0, r: 0.015, x: 0.08},
  %{from: 3, to: 4, v1: 138.0, v2: 34.5, r: 1.0e-7, x: 0.1, tap: 0.96875}
]

Enum.each(expected_xfmrs, fn xfmr ->
  Logger.info("\nTransformer Bus#{xfmr.from + 1} → Bus#{xfmr.to + 1} (0-indexed: #{xfmr.from} → #{xfmr.to})")
  Logger.info("  Voltage: #{xfmr.v1}kV / #{xfmr.v2}kV")
  Logger.info("  Voltage ratio (v2/v1): #{xfmr.v2 / xfmr.v1}")
  Logger.info("  R: #{xfmr.r}, X: #{xfmr.x}")
  if Map.has_key?(xfmr, :tap) do
    Logger.info("  Off-nominal tap: #{xfmr.tap}")
    Logger.info("  Combined tap: #{xfmr.tap * xfmr.v2 / xfmr.v1}")
  end
end)

Logger.info("\n" <> String.duplicate("=", 100))
Logger.info("COMPARISON")
Logger.info(String.duplicate("=", 100))

# Find how parser interpreted each expected transformer
Enum.each(expected_xfmrs, fn xfmr ->
  parsed = Enum.find(lines, fn l -> l.from == xfmr.from and l.to == xfmr.to end)

  if parsed do
    tap_value = Map.get(parsed, :tap, Map.get(parsed, :tap_ratio, 1.0))
    voltage_ratio = xfmr.v2 / xfmr.v1
    expected_tap = if Map.has_key?(xfmr, :tap), do: xfmr.tap * voltage_ratio, else: voltage_ratio

    Logger.info("\nBus#{xfmr.from + 1} → Bus#{xfmr.to + 1} (0-indexed: #{xfmr.from} → #{xfmr.to}):")
    Logger.info("  Voltage ratio (v2/v1): #{Float.round(voltage_ratio, 6)}")

    if Map.has_key?(xfmr, :tap) do
      Logger.info("  Off-nominal tap:       #{xfmr.tap}")
      Logger.info("  Combined tap:          #{Float.round(expected_tap, 6)}")
    else
      Logger.info("  Expected tap:          #{Float.round(expected_tap, 6)}")
    end

    Logger.info("  Parser tap:            #{Float.round(tap_value, 6)}")

    if is_number(tap_value) do
      diff = abs(tap_value - expected_tap)
      if diff > 0.001 do
        Logger.info("  ⚠️  MISMATCH! Difference: #{Float.round(diff, 6)}")
      else
        Logger.info("  ✓ MATCH")
      end
    else
      Logger.info("  ⚠️  Parser did not set tap ratio!")
    end
  else
    Logger.info("\n⚠️  Bus#{xfmr.from + 1} → Bus#{xfmr.to + 1} NOT FOUND in parsed data!")
  end
end)

Logger.info("\n" <> String.duplicate("=", 100))
