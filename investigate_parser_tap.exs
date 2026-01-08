#!/usr/bin/env elixir

# Investigate what the parser is doing with transformer tap ratios

require Logger

test_data_dir = Path.join(__DIR__, "test/test_data")
raw_file_path = Path.join(test_data_dir, "000_6bus_v33.RAW")

{:ok, {_buses, lines, _shunts, _metadata}} =
  PowerSystemParsers.PsseRawParser.parse_file(raw_file_path)

Logger.info("\n" <> String.duplicate("=", 100))
Logger.info("PARSER TAP RATIO INVESTIGATION")
Logger.info(String.duplicate("=", 100))

# Focus on transformers
transformers = Enum.filter(lines, fn line ->
  tap = Map.get(line, :tap, 1.0)
  tap != 1.0
end)

Logger.info("\nTransformers found with tap != 1.0:")
Logger.info("Count: #{length(transformers)}")

Enum.each(transformers, fn xfmr ->
  Logger.info("\n" <> String.duplicate("-", 100))
  Logger.info("Transformer: Bus#{xfmr.from + 1} → Bus#{xfmr.to + 1} (0-indexed: #{xfmr.from} → #{xfmr.to})")
  Logger.info(String.duplicate("-", 100))

  # Show all fields the parser set
  Logger.info("All fields from parser:")
  Enum.each(Map.keys(xfmr), fn key ->
    Logger.info("  #{key}: #{inspect(Map.get(xfmr, key))}")
  end)

  # Check for voltage-related fields
  voltage_fields = [:nomv1, :nomv2, :windv1, :windv2, :base_kv, :base_kv_from, :base_kv_to]
  Logger.info("\nVoltage-related fields:")
  Enum.each(voltage_fields, fn field ->
    if Map.has_key?(xfmr, field) do
      Logger.info("  #{field}: #{inspect(Map.get(xfmr, field))}")
    end
  end)
end)

# Now let's check the raw file directly for Bus 4→5 transformer
Logger.info("\n" <> String.duplicate("=", 100))
Logger.info("RAW FILE DATA FOR BUS 4→5 TRANSFORMER")
Logger.info(String.duplicate("=", 100))

raw_lines = File.read!(raw_file_path) |> String.split("\n")

# Find transformer section
in_xfmr_section = false
xfmr_lines = []

Enum.each(raw_lines, fn line ->
  cond do
    String.contains?(line, "BEGIN TRANSFORMER DATA") ->
      in_xfmr_section = true

    String.contains?(line, "END OF TRANSFORMER DATA") ->
      in_xfmr_section = false

    in_xfmr_section and String.starts_with?(String.trim(line), "4,") and String.contains?(line, "5,") ->
      xfmr_lines = [line | xfmr_lines]

    in_xfmr_section and xfmr_lines != [] and not String.starts_with?(String.trim(line), "0") ->
      # Continuation line
      xfmr_lines = [line | xfmr_lines]

    true ->
      :ok
  end
end)

Logger.info("\nRaw transformer record (Bus 4→5):")
Enum.reverse(xfmr_lines) |> Enum.with_index(1) |> Enum.each(fn {line, idx} ->
  Logger.info("Line #{idx}: #{line}")
end)

Logger.info("\n" <> String.duplicate("=", 100))
Logger.info("HYPOTHESIS: If CSV is correct, what should parser do?")
Logger.info(String.duplicate("=", 100))
Logger.info("""
If the CSV Y-bus is correct (transformers as tap=1.0 lines), then one of these must be true:

1. The parser should NOT calculate tap ratios from voltage levels
   - Instead, it should only use the 'off-nominal' tap (windv1/windv2)
   - Voltage transformation should be handled separately (not in Y-bus)

2. The parser should set tap=1.0 for all transformers
   - Tap ratios handled in per-unit conversion or power flow equations
   - Not in the Y-bus matrix itself

3. The parser is using the wrong tap field
   - Maybe it should use windv1/windv2 ratio, not the combined ratio
   - Or tap should always be 1.0 in per-unit system

Let me check what tap values would give the CSV Y-bus...
""")

# Calculate what tap would be needed for each transformer to match CSV
csv_expected_taps = [
  %{from: 1, to: 3, name: "Bus2→4", actual_tap: 0.4, needed_for_csv: 1.0},
  %{from: 3, to: 2, name: "Bus4→3", actual_tap: 2.5, needed_for_csv: 1.0},
  %{from: 3, to: 4, name: "Bus4→5", actual_tap: 0.2421875, needed_for_csv: 1.0}
]

Logger.info("\nFor CSV to be correct:")
Enum.each(csv_expected_taps, fn xfmr ->
  Logger.info("  #{xfmr.name}: Parser gives tap=#{xfmr.actual_tap}, CSV needs tap=#{xfmr.needed_for_csv}")
end)

Logger.info("\n→ Parser would need to ignore voltage ratios completely!")
Logger.info("→ Transformers would be treated as lines with only series impedance")

Logger.info("\n" <> String.duplicate("=", 100))
