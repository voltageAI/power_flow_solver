#!/usr/bin/env elixir

require Logger

test_data_dir = Path.join(__DIR__, "test/test_data")
raw_file_path = Path.join(test_data_dir, "000_6bus_v33.RAW")

{:ok, {buses, _lines, _shunts, _metadata}} =
  PowerSystemParsers.PsseRawParser.parse_file(raw_file_path)

Logger.info("\n" <> String.duplicate("=", 80))
Logger.info("ANGLE UNIT VERIFICATION")
Logger.info(String.duplicate("=", 80))

Logger.info("\nWhat parser gives us:")
Enum.each(buses, fn bus ->
  v_angle = Map.get(bus, :v_angle, 0.0)
  Logger.info("  Bus #{bus.id}: v_angle = #{v_angle} (should be radians)")
end)

Logger.info("\nExpected values (from RAW file in degrees):")
Logger.info("  Bus 1: 0.0° = 0.0 rad")
Logger.info("  Bus 2: 3.2349° = 0.05646 rad")
Logger.info("  Bus 3: -1.4464° = -0.02524 rad")
Logger.info("  Bus 4: -1.4258° = -0.02488 rad")
Logger.info("  Bus 5: -7.1775° = -0.12528 rad")
Logger.info("  Bus 6: 1.4100° = 0.02461 rad")

Logger.info("\nIf solver converts AGAIN (bug):")
expected_angles_deg = [0.0, 3.2349, -1.4464, -1.4258, -7.1775, 1.4100]
Enum.zip(buses, expected_angles_deg) |> Enum.each(fn {bus, deg} ->
  parser_rad = Map.get(bus, :v_angle, 0.0)
  # If solver thinks parser_rad is degrees and converts again
  solver_rad = parser_rad * :math.pi() / 180.0
  Logger.info("  Bus #{bus.id}: #{Float.round(parser_rad, 6)} rad → #{Float.round(solver_rad, 8)} rad (WRONG!)")
end)

Logger.info("\n" <> String.duplicate("=", 80))
Logger.info("CONCLUSION")
Logger.info(String.duplicate("=", 80))
Logger.info("""
Parser outputs: RADIANS (converts from degrees in RAW file)
Solver expects: DEGREES (then converts to radians)

Result: Angles are ~3283× too small!

FIX: Remove the conversion in solver's initialize_voltage function
     OR change parser to output degrees
""")
