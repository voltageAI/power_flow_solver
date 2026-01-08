#!/usr/bin/env elixir

# Calculate initial power mismatches step by step

require Logger

test_data_dir = Path.join(__DIR__, "test/test_data")
raw_file_path = Path.join(test_data_dir, "000_6bus_v33.RAW")

{:ok, {buses, lines, _shunts, _metadata}} =
  PowerSystemParsers.PsseRawParser.parse_file(raw_file_path)

# Build Y-bus
system = %{buses: buses, lines: lines}
y_bus = PowerFlowSolver.NewtonRaphson.build_y_bus(system)

# Convert Y-bus to dense
n = length(y_bus.row_ptrs) - 1

y_dense =
  for i <- 0..(n - 1) do
    for j <- 0..(n - 1) do
      row_start = Enum.at(y_bus.row_ptrs, i)
      row_end = Enum.at(y_bus.row_ptrs, i + 1)
      col_indices_slice = Enum.slice(y_bus.col_indices, row_start..(row_end - 1))
      values_slice = Enum.slice(y_bus.values, row_start..(row_end - 1))

      case Enum.find_index(col_indices_slice, &(&1 == j)) do
        nil -> {0.0, 0.0}
        idx -> Enum.at(values_slice, idx)
      end
    end
  end

Logger.info("\n" <> String.duplicate("=", 100))
Logger.info("INITIAL POWER MISMATCH CALCULATION")
Logger.info(String.duplicate("=", 100))

Logger.info("\nStep 1: Initial Voltages (from RAW file)")
Logger.info(String.duplicate("-", 100))

# Get initial voltages
voltages = Enum.map(buses, fn bus ->
  v_mag = bus.v_magnitude
  v_ang = Map.get(bus, :v_angle, 0.0)  # Already in radians
  v_ang_deg = v_ang * 180.0 / :math.pi()

  Logger.info("  Bus #{bus.id}: V = #{Float.round(v_mag, 6)} ∠ #{Float.round(v_ang_deg, 4)}° (#{Float.round(v_ang, 6)} rad)")

  {v_mag, v_ang}
end)

Logger.info("\nStep 2: Scheduled Powers (P_gen - P_load, Q_gen - Q_load)")
Logger.info(String.duplicate("-", 100))

scheduled = Enum.map(buses, fn bus ->
  p_sched = bus.p_gen - bus.p_load
  q_sched = bus.q_gen - bus.q_load
  Logger.info("  Bus #{bus.id}: P_sched = #{Float.round(p_sched, 6)} pu, Q_sched = #{Float.round(q_sched, 6)} pu")
  {p_sched, q_sched}
end)

Logger.info("\nStep 3: Calculate Injected Powers (S = V * conj(I) = V * conj(Y * V))")
Logger.info(String.duplicate("-", 100))

calculated = Enum.with_index(voltages) |> Enum.map(fn {{v_mag, v_ang}, i} ->
  # Calculate current injection: I_i = sum(Y_ij * V_j)
  {i_real, i_imag} = Enum.reduce(0..(n-1), {0.0, 0.0}, fn j, {i_r, i_i} ->
    {y_real, y_imag} = Enum.at(Enum.at(y_dense, i), j)
    {vj_mag, vj_ang} = Enum.at(voltages, j)

    # V_j in rectangular
    vj_real = vj_mag * :math.cos(vj_ang)
    vj_imag = vj_mag * :math.sin(vj_ang)

    # Y_ij * V_j (complex multiplication)
    prod_real = y_real * vj_real - y_imag * vj_imag
    prod_imag = y_real * vj_imag + y_imag * vj_real

    {i_r + prod_real, i_i + prod_imag}
  end)

  # V_i in rectangular
  vi_real = v_mag * :math.cos(v_ang)
  vi_imag = v_mag * :math.sin(v_ang)

  # S_i = V_i * conj(I_i)
  p_calc = vi_real * i_real + vi_imag * i_imag
  q_calc = vi_imag * i_real - vi_real * i_imag

  {p_calc, q_calc}
end)

Enum.zip([buses, calculated]) |> Enum.each(fn {bus, {p_calc, q_calc}} ->
  Logger.info("  Bus #{bus.id}: P_calc = #{Float.round(p_calc, 6)} pu, Q_calc = #{Float.round(q_calc, 6)} pu")
end)

Logger.info("\nStep 4: Power Mismatches (ΔP = P_sched - P_calc, ΔQ = Q_sched - Q_calc)")
Logger.info(String.duplicate("-", 100))

mismatches = Enum.zip([buses, scheduled, calculated]) |> Enum.map(fn {bus, {p_sched, q_sched}, {p_calc, q_calc}} ->
  delta_p = p_sched - p_calc
  delta_q = q_sched - q_calc

  bus_type = Map.get(bus, :type, :unknown)

  Logger.info("  Bus #{bus.id} (#{bus_type}): ΔP = #{Float.round(delta_p, 6)} pu, ΔQ = #{Float.round(delta_q, 6)} pu")

  {delta_p, delta_q, bus_type}
end)

Logger.info("\nStep 5: Mismatch Vector (excluding slack bus P and Q, PV bus Q)")
Logger.info(String.duplicate("-", 100))

mismatch_vector = Enum.zip([buses, mismatches])
  |> Enum.flat_map(fn {bus, {delta_p, delta_q, bus_type}} ->
    case bus_type do
      :slack -> []  # Skip slack bus entirely
      :pv -> [delta_p]  # PV bus: only P equation
      :pq -> [delta_p, delta_q]  # PQ bus: both P and Q equations
      _ -> []
    end
  end)

Logger.info("Mismatch vector length: #{length(mismatch_vector)}")
Logger.info("Values:")
Enum.with_index(mismatch_vector) |> Enum.each(fn {val, idx} ->
  Logger.info("  [#{idx}] = #{Float.round(val, 6)}")
end)

# Calculate norm
mismatch_norm = :math.sqrt(Enum.reduce(mismatch_vector, 0.0, fn x, acc -> acc + x * x end))

Logger.info("\nStep 6: Mismatch Norm")
Logger.info(String.duplicate("-", 100))
Logger.info("||Δ|| = #{mismatch_norm}")

tolerance = 1.0e-6
Logger.info("\nTolerance: #{tolerance}")
Logger.info("Converged? #{if mismatch_norm < tolerance, do: "YES ✓", else: "NO ✗"}")
Logger.info("Ratio: mismatch/tolerance = #{Float.round(mismatch_norm / tolerance, 2)}×")

Logger.info("\n" <> String.duplicate("=", 100))
