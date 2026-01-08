#!/usr/bin/env elixir

# Script to compare Y-bus from solver with CSV reference

require Logger

# Parse the CSV Y-bus file
defmodule YBusCSVParser do
  def parse(csv_path) do
    csv_path
    |> File.read!()
    |> String.split("\n")
    |> Enum.drop(2)  # Skip two header rows
    |> Enum.filter(&(&1 != ""))
    |> Enum.map(&parse_row/1)
    |> Enum.into(%{})
  end

  defp parse_row(row) do
    [bus_num, _name | values] = String.split(row, ",")

    bus_idx = String.to_integer(bus_num) - 1  # Convert to 0-indexed

    parsed_values =
      values
      |> Enum.with_index()
      |> Enum.map(fn {val, col_idx} ->
        {col_idx, parse_complex(String.trim(val))}
      end)
      |> Enum.filter(fn {_idx, val} -> val != nil end)
      |> Enum.into(%{})

    {bus_idx, parsed_values}
  end

  defp parse_complex(""), do: nil
  defp parse_complex(str) do
    # Parse format like "3.40 - j18.09" or "-1.13 + j6.04"
    cond do
      String.contains?(str, "+ j") ->
        [real_str, imag_str] = String.split(str, " + j")
        {String.to_float(String.trim(real_str)), String.to_float(String.trim(imag_str))}

      String.contains?(str, "- j") ->
        [real_str, imag_str] = String.split(str, " - j")
        {String.to_float(String.trim(real_str)), -String.to_float(String.trim(imag_str))}

      true ->
        nil
    end
  end
end

# Build Y-bus from solver
test_data_dir = Path.join(__DIR__, "test/test_data")
raw_file_path = Path.join(test_data_dir, "000_6bus_v33.RAW")

{:ok, {buses, lines, _shunts, _metadata}} =
  PowerSystemParsers.PsseRawParser.parse_file(raw_file_path)

system = %{buses: buses, lines: lines}
solver_ybus = PowerFlowSolver.NewtonRaphson.build_y_bus(system)

# Load CSV Y-bus
csv_path = Path.expand("~/Downloads/000_ybus.csv")
csv_ybus = YBusCSVParser.parse(csv_path)

# Convert solver Y-bus from CSR to dense format
n = length(solver_ybus.row_ptrs) - 1

solver_dense =
  for i <- 0..(n - 1) do
    for j <- 0..(n - 1) do
      row_start = Enum.at(solver_ybus.row_ptrs, i)
      row_end = Enum.at(solver_ybus.row_ptrs, i + 1)

      col_indices_slice = Enum.slice(solver_ybus.col_indices, row_start..(row_end - 1))
      values_slice = Enum.slice(solver_ybus.values, row_start..(row_end - 1))

      case Enum.find_index(col_indices_slice, &(&1 == j)) do
        nil -> {0.0, 0.0}
        idx -> Enum.at(values_slice, idx)
      end
    end
  end
  |> Enum.with_index()
  |> Enum.into(%{}, fn {row_values, i} ->
    {i, Enum.with_index(row_values) |> Enum.into(%{}, fn {val, j} -> {j, val} end)}
  end)

# Compare the two Y-bus matrices
Logger.info("\n" <> String.duplicate("=", 80))
Logger.info("Y-BUS COMPARISON: Solver vs CSV Reference")
Logger.info(String.duplicate("=", 80))

{max_diff_real, max_diff_imag, max_diff_location, all_match} =
  Enum.reduce(0..(n - 1), {0.0, 0.0, nil, true}, fn i, {max_r, max_i, max_loc, match} ->
    Enum.reduce(0..(n - 1), {max_r, max_i, max_loc, match}, fn j, {mr, mi, ml, m} ->
      {solver_real, solver_imag} = solver_dense[i][j]
      {csv_real, csv_imag} = csv_ybus[i][j] || {0.0, 0.0}

      diff_real = abs(solver_real - csv_real)
      diff_imag = abs(solver_imag - csv_imag)

      new_match =
        if diff_real > 0.01 or diff_imag > 0.01 do
          Logger.info("\nDifference at Y[#{i}][#{j}]:")
          Logger.info("  Solver: #{Float.round(solver_real, 2)} + j#{Float.round(solver_imag, 2)}")
          Logger.info("  CSV:    #{Float.round(csv_real, 2)} + j#{Float.round(csv_imag, 2)}")
          Logger.info("  Diff:   #{Float.round(diff_real, 4)} + j#{Float.round(diff_imag, 4)}")
          false
        else
          m
        end

      new_mr = if diff_real > mr, do: diff_real, else: mr
      new_mi = if diff_imag > mi, do: diff_imag, else: mi
      new_ml = if diff_imag > mi, do: {i, j}, else: ml

      {new_mr, new_mi, new_ml, new_match}
    end)
  end)

Logger.info("\n" <> String.duplicate("-", 80))
Logger.info("COMPARISON SUMMARY")
Logger.info(String.duplicate("-", 80))
Logger.info("Max difference (real): #{Float.round(max_diff_real, 6)}")
Logger.info("Max difference (imag): #{Float.round(max_diff_imag, 6)}")

if max_diff_location do
  {i, j} = max_diff_location
  Logger.info("Max difference location: Y[#{i}][#{j}]")
end

if all_match do
  Logger.info("\n✓ Y-bus matrices MATCH (within tolerance of 0.01)")
else
  Logger.info("\n✗ Y-bus matrices DIFFER (see differences above)")
end

Logger.info(String.duplicate("=", 80))
