#!/usr/bin/env elixir

# Detailed side-by-side Y-bus comparison

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

defmodule YBusFormatter do
  # Format complex number
  def format_complex(re, im) do
    re_str = :io_lib.format("~7.2f", [re]) |> to_string() |> String.pad_leading(7)

    if im >= 0 do
      im_str = :io_lib.format("~7.2f", [im]) |> to_string() |> String.pad_leading(7)
      "#{re_str} +j#{im_str}"
    else
      im_str = :io_lib.format("~7.2f", [-im]) |> to_string() |> String.pad_leading(7)
      "#{re_str} -j#{im_str}"
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

# Print side-by-side comparison
Logger.info("\n" <> String.duplicate("=", 120))
Logger.info("DETAILED Y-BUS COMPARISON")
Logger.info(String.duplicate("=", 120))

for i <- 0..(n - 1) do
  Logger.info("\n" <> String.duplicate("-", 120))
  Logger.info("ROW #{i} (Bus #{i + 1})")
  Logger.info(String.duplicate("-", 120))
  Logger.info("Col | Solver Y-bus              | CSV Y-bus                 | Difference (Real, Imag)")
  Logger.info(String.duplicate("-", 120))

  for j <- 0..(n - 1) do
    {solver_real, solver_imag} = solver_dense[i][j]
    {csv_real, csv_imag} = csv_ybus[i][j] || {0.0, 0.0}

    diff_real = solver_real - csv_real
    diff_imag = solver_imag - csv_imag

    solver_str = YBusFormatter.format_complex(solver_real, solver_imag)
    csv_str = YBusFormatter.format_complex(csv_real, csv_imag)
    diff_str = YBusFormatter.format_complex(diff_real, diff_imag)

    marker = if abs(diff_real) > 0.01 or abs(diff_imag) > 0.01, do: "  ← DIFF", else: ""

    Logger.info(" #{j}  | #{solver_str} | #{csv_str} | #{diff_str}#{marker}")
  end
end

Logger.info("\n" <> String.duplicate("=", 120))
