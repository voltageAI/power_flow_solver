# Power Flow Solver - Contingency SCR Implementation Plan

**Project:** power_flow_solver
**Created:** 2025-12-03
**Purpose:** Add N-1 contingency short circuit ratio calculations for FRT assessment support

---

## Overview

This document specifies the minimal additions to power_flow_solver needed to support Static FRT (Fault Ride Through) assessment. Following the library's design philosophy, we add only **computationally intensive operations** that benefit from Rust's performance.

The domain logic (assessment, thresholds, mitigations, reporting) belongs in gridvar.

---

## Scope

### In Scope (This Project)

1. **N-1 Contingency SCR Calculation** - Efficiently compute short circuit ratios with branches removed
2. **Batch Thevenin Impedance with Contingencies** - Get Z_th at POI for multiple contingency scenarios

### Out of Scope (Belongs in gridvar)

- FRT assessment logic and thresholds
- Voltage recovery calculations (simple arithmetic)
- PLL stability classification
- Mitigation recommendations
- Facility/plant domain concepts
- Report generation

---

## Technical Design

### New Rust Functions

#### 1. `calculate_contingency_scr`

Calculate SCR at a bus with a specific branch out of service.

```rust
/// Calculate short circuit ratio with a branch contingency
///
/// This modifies the Y-bus by removing a branch, inverts to get Z-bus,
/// and extracts the Thevenin impedance at the specified bus.
///
/// # Arguments
/// * `y_bus_data` - Original Y-bus in CSR format
/// * `branch_to_remove` - Branch defined as (from_bus_idx, to_bus_idx, y_series, y_shunt)
/// * `poi_bus_idx` - Bus index where to calculate Z_th
/// * `system_mva_base` - System MVA base (typically 100.0)
///
/// # Returns
/// * `{:ok, (z_th_real, z_th_imag, short_circuit_mva)}` - Thevenin impedance and S_sc
/// * `{:error, reason}` - If calculation fails (e.g., network islands)
#[rustler::nif]
fn calculate_contingency_scr_rust(
    y_bus_data: (Vec<usize>, Vec<usize>, Vec<ComplexTuple>),
    branch_to_remove: (usize, usize, ComplexTuple, ComplexTuple), // (from, to, y_series, y_shunt)
    poi_bus_idx: usize,
    system_mva_base: f64,
) -> NifResult<(rustler::Atom, f64, f64, f64)>
```

#### 2. `calculate_contingency_scr_batch`

Efficiently scan multiple contingencies for a single POI.

```rust
/// Calculate SCR for multiple contingencies at once
///
/// More efficient than calling calculate_contingency_scr_rust repeatedly
/// because it can reuse intermediate computations.
///
/// # Arguments
/// * `y_bus_data` - Original Y-bus in CSR format
/// * `branches` - List of branches that could be contingencies
/// * `poi_bus_idx` - Bus index where to calculate Z_th
/// * `system_mva_base` - System MVA base
///
/// # Returns
/// * `{:ok, results}` - List of (branch_idx, z_th_real, z_th_imag, s_sc, success)
/// * `{:error, reason}` - If base calculation fails
#[rustler::nif]
fn calculate_contingency_scr_batch_rust(
    y_bus_data: (Vec<usize>, Vec<usize>, Vec<ComplexTuple>),
    branches: Vec<(usize, usize, usize, ComplexTuple, ComplexTuple)>, // (branch_id, from, to, y_series, y_shunt)
    poi_bus_idx: usize,
    system_mva_base: f64,
) -> NifResult<(rustler::Atom, Vec<(usize, f64, f64, f64, bool)>)>
```

---

## Rust Implementation

### File: `native/power_flow_solver/src/contingency_scr.rs`

```rust
//! Contingency Short Circuit Ratio Calculations
//!
//! Provides efficient N-1 contingency analysis for SCR calculations.
//! Used by FRT assessment to find worst-case grid strength scenarios.

use num_complex::Complex64;
use rayon::prelude::*;

use crate::scr::{YBusCsr, calculate_z_bus, calculate_short_circuit_mva, ScrConfig};

/// Branch data for contingency analysis
#[derive(Clone, Debug)]
pub struct BranchData {
    pub id: usize,
    pub from_bus: usize,
    pub to_bus: usize,
    pub y_series: Complex64,  // Series admittance (1/Z)
    pub y_shunt: Complex64,   // Total shunt admittance (line charging)
}

/// Result of a single contingency SCR calculation
#[derive(Clone, Debug)]
pub struct ContingencyScrResult {
    pub branch_id: usize,
    pub z_thevenin: Complex64,
    pub short_circuit_mva: f64,
    pub success: bool,
    pub error: Option<String>,
}

/// Modify Y-bus by removing a branch
///
/// When a branch is removed:
/// - Subtract y_series from diagonal elements Y[from,from] and Y[to,to]
/// - Add y_series to off-diagonal elements Y[from,to] and Y[to,from]
/// - Subtract y_shunt/2 from each diagonal (line charging)
///
/// Note: This creates a new Y-bus; original is not modified.
pub fn remove_branch_from_ybus(
    y_bus: &YBusCsr,
    branch: &BranchData,
) -> Result<YBusCsr, String> {
    let n = y_bus.n;

    if branch.from_bus >= n || branch.to_bus >= n {
        return Err(format!(
            "Branch buses ({}, {}) out of range for {}-bus system",
            branch.from_bus, branch.to_bus, n
        ));
    }

    // Convert to dense for modification (simpler than sparse updates)
    let mut y_dense = y_bus.to_dense();

    let y_s = branch.y_series;
    let y_sh_half = branch.y_shunt * 0.5;

    // Remove branch contribution from Y-bus
    // Diagonal: subtract series admittance and half shunt
    y_dense[branch.from_bus][branch.from_bus] -= y_s + y_sh_half;
    y_dense[branch.to_bus][branch.to_bus] -= y_s + y_sh_half;

    // Off-diagonal: add back the negative admittance (removing -y_s means adding y_s)
    y_dense[branch.from_bus][branch.to_bus] += y_s;
    y_dense[branch.to_bus][branch.from_bus] += y_s;

    // Convert back to CSR
    Ok(dense_to_csr(&y_dense))
}

/// Convert dense matrix to CSR format
fn dense_to_csr(dense: &[Vec<Complex64>]) -> YBusCsr {
    let n = dense.len();
    let mut row_ptrs = vec![0usize];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for row in dense.iter() {
        for (col, &val) in row.iter().enumerate() {
            // Only store non-zero elements
            if val.norm() > 1e-15 {
                col_indices.push(col);
                values.push(val);
            }
        }
        row_ptrs.push(col_indices.len());
    }

    YBusCsr::new(row_ptrs, col_indices, values)
}

/// Calculate SCR with a single branch contingency
pub fn calculate_contingency_scr(
    y_bus: &YBusCsr,
    branch: &BranchData,
    poi_bus: usize,
    config: &ScrConfig,
) -> ContingencyScrResult {
    // Modify Y-bus
    let modified_ybus = match remove_branch_from_ybus(y_bus, branch) {
        Ok(y) => y,
        Err(e) => {
            return ContingencyScrResult {
                branch_id: branch.id,
                z_thevenin: Complex64::new(0.0, 0.0),
                short_circuit_mva: 0.0,
                success: false,
                error: Some(e),
            };
        }
    };

    // Calculate Z-bus
    let z_bus_result = calculate_z_bus(&modified_ybus);

    if !z_bus_result.success {
        return ContingencyScrResult {
            branch_id: branch.id,
            z_thevenin: Complex64::new(0.0, 0.0),
            short_circuit_mva: 0.0,
            success: false,
            error: z_bus_result.error,
        };
    }

    // Check POI bus is valid
    if poi_bus >= z_bus_result.n {
        return ContingencyScrResult {
            branch_id: branch.id,
            z_thevenin: Complex64::new(0.0, 0.0),
            short_circuit_mva: 0.0,
            success: false,
            error: Some(format!("POI bus {} out of range", poi_bus)),
        };
    }

    // Extract Thevenin impedance
    let z_th = z_bus_result.z_bus[poi_bus][poi_bus];
    let s_sc = calculate_short_circuit_mva(z_th, config.system_mva_base);

    ContingencyScrResult {
        branch_id: branch.id,
        z_thevenin: z_th,
        short_circuit_mva: s_sc,
        success: true,
        error: None,
    }
}

/// Calculate SCR for multiple contingencies in parallel
pub fn calculate_contingency_scr_batch(
    y_bus: &YBusCsr,
    branches: &[BranchData],
    poi_bus: usize,
    config: &ScrConfig,
) -> Vec<ContingencyScrResult> {
    branches
        .par_iter()
        .map(|branch| calculate_contingency_scr(y_bus, branch, poi_bus, config))
        .collect()
}

/// Find the worst-case (lowest SCR) contingency
pub fn find_worst_contingency(
    y_bus: &YBusCsr,
    branches: &[BranchData],
    poi_bus: usize,
    config: &ScrConfig,
) -> Option<ContingencyScrResult> {
    let results = calculate_contingency_scr_batch(y_bus, branches, poi_bus, config);

    results
        .into_iter()
        .filter(|r| r.success)
        .min_by(|a, b| {
            a.short_circuit_mva
                .partial_cmp(&b.short_circuit_mva)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_3bus_ybus() -> YBusCsr {
        // Simple 3-bus radial: Bus0 -- Bus1 -- Bus2
        // Branch 0-1: Z = 0.01 + j0.1
        // Branch 1-2: Z = 0.01 + j0.1

        let z = Complex64::new(0.01, 0.1);
        let y = Complex64::new(1.0, 0.0) / z;

        let row_ptrs = vec![0, 2, 5, 7];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let values = vec![
            y, -y,           // Row 0
            -y, y * 2.0, -y, // Row 1
            -y, y,           // Row 2
        ];

        YBusCsr::new(row_ptrs, col_indices, values)
    }

    #[test]
    fn test_remove_branch() {
        let y_bus = create_test_3bus_ybus();

        let z = Complex64::new(0.01, 0.1);
        let y_series = Complex64::new(1.0, 0.0) / z;

        let branch = BranchData {
            id: 1,
            from_bus: 1,
            to_bus: 2,
            y_series,
            y_shunt: Complex64::new(0.0, 0.0),
        };

        let modified = remove_branch_from_ybus(&y_bus, &branch).unwrap();

        // After removing branch 1-2:
        // Bus 2 should be isolated (Y[2,2] = 0, Y[2,1] = Y[1,2] = 0)
        let y_22 = modified.get(2, 2);
        assert!(y_22.norm() < 1e-10, "Bus 2 should be isolated");
    }

    #[test]
    fn test_contingency_reduces_ssc() {
        let y_bus = create_test_3bus_ybus();
        let config = ScrConfig::default();

        // Calculate base case S_sc at bus 2
        let z_bus_base = calculate_z_bus(&y_bus);
        assert!(z_bus_base.success);
        let s_sc_base = calculate_short_circuit_mva(
            z_bus_base.z_bus[2][2],
            config.system_mva_base
        );

        // Calculate with branch 0-1 out (weakens connection to slack)
        let z = Complex64::new(0.01, 0.1);
        let y_series = Complex64::new(1.0, 0.0) / z;

        let branch = BranchData {
            id: 0,
            from_bus: 0,
            to_bus: 1,
            y_series,
            y_shunt: Complex64::new(0.0, 0.0),
        };

        let result = calculate_contingency_scr(&y_bus, &branch, 2, &config);

        // With branch 0-1 out, bus 2 is isolated from slack
        // S_sc should be much lower (or zero if fully isolated)
        assert!(
            !result.success || result.short_circuit_mva < s_sc_base,
            "Contingency should reduce S_sc or fail due to islanding"
        );
    }

    #[test]
    fn test_batch_calculation() {
        let y_bus = create_test_3bus_ybus();
        let config = ScrConfig::default();

        let z = Complex64::new(0.01, 0.1);
        let y_series = Complex64::new(1.0, 0.0) / z;

        let branches = vec![
            BranchData {
                id: 0,
                from_bus: 0,
                to_bus: 1,
                y_series,
                y_shunt: Complex64::new(0.0, 0.0),
            },
            BranchData {
                id: 1,
                from_bus: 1,
                to_bus: 2,
                y_series,
                y_shunt: Complex64::new(0.0, 0.0),
            },
        ];

        let results = calculate_contingency_scr_batch(&y_bus, &branches, 1, &config);

        assert_eq!(results.len(), 2);
    }
}
```

---

## Elixir Wrapper

### File: `lib/power_flow_solver/scr.ex` (additions to existing module)

```elixir
defmodule PowerFlowSolver.SCR do
  # ... existing code ...

  @doc """
  Calculate SCR at a bus with a specific branch contingency (N-1).

  This is useful for FRT assessment where you need to find the worst-case
  grid strength under contingency conditions.

  ## Arguments

  - `system` - Power system map
  - `poi_bus_id` - Bus ID where to calculate SCR
  - `branch` - Branch to remove: `%{from_bus: id, to_bus: id}` or branch struct
  - `opts` - Optional configuration

  ## Returns

  - `{:ok, result}` - Contingency SCR result with Z_th and S_sc
  - `{:error, reason}` - If calculation fails (e.g., network islands)

  ## Example

      {:ok, result} = PowerFlowSolver.SCR.calculate_contingency(
        system,
        poi_bus_id: 5,
        branch: %{from_bus: 1, to_bus: 2}
      )

      if result.success do
        IO.puts("S_sc with contingency: \#{result.short_circuit_mva} MVA")
      else
        IO.puts("Contingency causes islanding")
      end
  """
  @spec calculate_contingency(map(), keyword()) ::
          {:ok, map()} | {:error, String.t()}
  def calculate_contingency(system, opts) do
    poi_bus_id = Keyword.fetch!(opts, :poi_bus_id)
    branch = Keyword.fetch!(opts, :branch)
    system_mva_base = Keyword.get(opts, :system_mva_base, 100.0)

    with {:ok, y_bus_data, bus_id_to_index} <- prepare_y_bus(system),
         {:ok, poi_idx} <- get_bus_index(poi_bus_id, bus_id_to_index),
         {:ok, branch_data} <- prepare_branch_for_contingency(branch, system, bus_id_to_index) do

      case SparseLinearAlgebra.calculate_contingency_scr_rust(
             y_bus_data,
             branch_data,
             poi_idx,
             system_mva_base
           ) do
        {:ok, z_real, z_imag, s_sc} ->
          {:ok, %{
            z_thevenin_real: z_real,
            z_thevenin_imag: z_imag,
            z_thevenin_magnitude: :math.sqrt(z_real * z_real + z_imag * z_imag),
            short_circuit_mva: s_sc,
            success: true
          }}

        {:error, _reason} ->
          {:ok, %{success: false, error: "Contingency causes network islanding"}}
      end
    end
  end

  @doc """
  Calculate SCR for multiple contingencies at a POI.

  Efficiently scans N-1 contingencies to find worst-case grid strength.
  Returns results sorted by short circuit MVA (weakest first).

  ## Arguments

  - `system` - Power system map
  - `poi_bus_id` - Bus ID where to calculate SCR
  - `branches` - List of branches to test as contingencies (or `:all` for all branches)
  - `opts` - Optional configuration

  ## Returns

  - `{:ok, results}` - List of contingency results, sorted weakest first
  - `{:error, reason}` - If base calculation fails

  ## Example

      # Test specific branches
      {:ok, results} = PowerFlowSolver.SCR.calculate_contingency_batch(
        system,
        poi_bus_id: 5,
        branches: [branch1, branch2, branch3]
      )

      # Find worst case
      worst = List.first(results)
      IO.puts("Worst contingency: Branch \#{worst.branch_id}, S_sc = \#{worst.short_circuit_mva} MVA")

      # Test all branches
      {:ok, all_results} = PowerFlowSolver.SCR.calculate_contingency_batch(
        system,
        poi_bus_id: 5,
        branches: :all
      )
  """
  @spec calculate_contingency_batch(map(), keyword()) ::
          {:ok, [map()]} | {:error, String.t()}
  def calculate_contingency_batch(system, opts) do
    poi_bus_id = Keyword.fetch!(opts, :poi_bus_id)
    branches_opt = Keyword.get(opts, :branches, :all)
    system_mva_base = Keyword.get(opts, :system_mva_base, 100.0)

    branches = if branches_opt == :all, do: system.branches, else: branches_opt

    with {:ok, y_bus_data, bus_id_to_index} <- prepare_y_bus(system),
         {:ok, poi_idx} <- get_bus_index(poi_bus_id, bus_id_to_index),
         {:ok, branches_data} <- prepare_branches_for_contingency(branches, system, bus_id_to_index) do

      case SparseLinearAlgebra.calculate_contingency_scr_batch_rust(
             y_bus_data,
             branches_data,
             poi_idx,
             system_mva_base
           ) do
        {:ok, raw_results} ->
          results =
            raw_results
            |> Enum.map(fn {branch_id, z_real, z_imag, s_sc, success} ->
              %{
                branch_id: branch_id,
                z_thevenin_real: z_real,
                z_thevenin_imag: z_imag,
                z_thevenin_magnitude: :math.sqrt(z_real * z_real + z_imag * z_imag),
                short_circuit_mva: s_sc,
                success: success
              }
            end)
            |> Enum.filter(& &1.success)
            |> Enum.sort_by(& &1.short_circuit_mva)

          {:ok, results}

        {:error, _} ->
          {:error, "Contingency SCR calculation failed"}
      end
    end
  end

  @doc """
  Find the worst-case (lowest SCR) contingency at a POI.

  Convenience function that returns only the single worst contingency.

  ## Arguments

  - `system` - Power system map
  - `poi_bus_id` - Bus ID where to calculate SCR
  - `p_rated_mw` - Plant rated power for SCR calculation
  - `opts` - Optional configuration including `:branches` list

  ## Returns

  - `{:ok, result}` - Worst contingency with SCR calculated
  - `{:ok, nil}` - If no valid contingencies (all cause islanding)
  - `{:error, reason}` - If calculation fails

  ## Example

      {:ok, worst} = PowerFlowSolver.SCR.find_worst_contingency(
        system,
        poi_bus_id: 5,
        p_rated_mw: 100.0
      )

      IO.puts("Worst case SCR: \#{worst.scr} (branch \#{worst.branch_id} out)")
  """
  @spec find_worst_contingency(map(), keyword()) ::
          {:ok, map() | nil} | {:error, String.t()}
  def find_worst_contingency(system, opts) do
    p_rated_mw = Keyword.fetch!(opts, :p_rated_mw)

    case calculate_contingency_batch(system, opts) do
      {:ok, []} ->
        {:ok, nil}

      {:ok, [worst | _]} ->
        scr = if p_rated_mw > 0, do: worst.short_circuit_mva / p_rated_mw, else: :infinity
        {:ok, Map.merge(worst, %{scr: scr, grid_strength: classify_grid_strength(scr)})}

      error ->
        error
    end
  end

  # Private helpers for contingency functions

  defp get_bus_index(bus_id, bus_id_to_index) do
    case Map.fetch(bus_id_to_index, bus_id) do
      {:ok, idx} -> {:ok, idx}
      :error -> {:error, "Bus #{bus_id} not found in system"}
    end
  end

  defp prepare_branch_for_contingency(branch, system, bus_id_to_index) do
    from_idx = Map.get(bus_id_to_index, branch.from_bus)
    to_idx = Map.get(bus_id_to_index, branch.to_bus)

    if is_nil(from_idx) or is_nil(to_idx) do
      {:error, "Branch endpoints not found in system"}
    else
      # Calculate branch admittance
      {y_series, y_shunt} = calculate_branch_admittance(branch, system)
      {:ok, {from_idx, to_idx, y_series, y_shunt}}
    end
  end

  defp prepare_branches_for_contingency(branches, system, bus_id_to_index) do
    results =
      branches
      |> Enum.with_index()
      |> Enum.map(fn {branch, idx} ->
        from_idx = Map.get(bus_id_to_index, branch.from_bus)
        to_idx = Map.get(bus_id_to_index, branch.to_bus)

        if is_nil(from_idx) or is_nil(to_idx) do
          nil
        else
          {y_series, y_shunt} = calculate_branch_admittance(branch, system)
          branch_id = Map.get(branch, :id, idx)
          {branch_id, from_idx, to_idx, y_series, y_shunt}
        end
      end)
      |> Enum.reject(&is_nil/1)

    {:ok, results}
  end

  defp calculate_branch_admittance(branch, _system) do
    r = Map.get(branch, :r, 0.0)
    x = Map.get(branch, :x, 0.01)
    b = Map.get(branch, :b, 0.0)

    # Series admittance: Y = 1 / (R + jX)
    z_mag_sq = r * r + x * x
    y_series_real = r / z_mag_sq
    y_series_imag = -x / z_mag_sq

    # Shunt admittance (line charging)
    y_shunt_real = 0.0
    y_shunt_imag = b

    {{y_series_real, y_series_imag}, {y_shunt_real, y_shunt_imag}}
  end
end
```

---

## NIF Registration

### Update `native/power_flow_solver/src/lib.rs`

```rust
// Add to module imports
mod contingency_scr;

// Add NIF functions
#[rustler::nif]
fn calculate_contingency_scr_rust(
    y_bus_data: (Vec<usize>, Vec<usize>, Vec<ComplexTuple>),
    branch_to_remove: (usize, usize, ComplexTuple, ComplexTuple),
    poi_bus_idx: usize,
    system_mva_base: f64,
) -> NifResult<(rustler::Atom, f64, f64, f64)> {
    let (row_ptrs, col_indices, values_tuples) = y_bus_data;
    let (from_bus, to_bus, y_series, y_shunt) = branch_to_remove;

    let values: Vec<Complex64> = values_tuples
        .into_iter()
        .map(|(re, im)| Complex64::new(re, im))
        .collect();

    let y_bus = scr::YBusCsr::new(row_ptrs, col_indices, values);

    let branch = contingency_scr::BranchData {
        id: 0,
        from_bus,
        to_bus,
        y_series: Complex64::new(y_series.0, y_series.1),
        y_shunt: Complex64::new(y_shunt.0, y_shunt.1),
    };

    let config = scr::ScrConfig {
        system_mva_base,
        ..Default::default()
    };

    let result = contingency_scr::calculate_contingency_scr(&y_bus, &branch, poi_bus_idx, &config);

    if result.success {
        Ok((atoms::ok(), result.z_thevenin.re, result.z_thevenin.im, result.short_circuit_mva))
    } else {
        Ok((atoms::error(), 0.0, 0.0, 0.0))
    }
}

#[rustler::nif]
fn calculate_contingency_scr_batch_rust(
    y_bus_data: (Vec<usize>, Vec<usize>, Vec<ComplexTuple>),
    branches: Vec<(usize, usize, usize, ComplexTuple, ComplexTuple)>,
    poi_bus_idx: usize,
    system_mva_base: f64,
) -> NifResult<(rustler::Atom, Vec<(usize, f64, f64, f64, bool)>)> {
    let (row_ptrs, col_indices, values_tuples) = y_bus_data;

    let values: Vec<Complex64> = values_tuples
        .into_iter()
        .map(|(re, im)| Complex64::new(re, im))
        .collect();

    let y_bus = scr::YBusCsr::new(row_ptrs, col_indices, values);

    let branch_data: Vec<contingency_scr::BranchData> = branches
        .into_iter()
        .map(|(id, from, to, y_s, y_sh)| contingency_scr::BranchData {
            id,
            from_bus: from,
            to_bus: to,
            y_series: Complex64::new(y_s.0, y_s.1),
            y_shunt: Complex64::new(y_sh.0, y_sh.1),
        })
        .collect();

    let config = scr::ScrConfig {
        system_mva_base,
        ..Default::default()
    };

    let results = contingency_scr::calculate_contingency_scr_batch(&y_bus, &branch_data, poi_bus_idx, &config);

    let output: Vec<(usize, f64, f64, f64, bool)> = results
        .into_iter()
        .map(|r| (r.branch_id, r.z_thevenin.re, r.z_thevenin.im, r.short_circuit_mva, r.success))
        .collect();

    Ok((atoms::ok(), output))
}

// Update rustler::init! to include new NIFs
rustler::init!(
    "Elixir.PowerFlowSolver.SparseLinearAlgebra",
    [
        // ... existing NIFs ...
        calculate_contingency_scr_rust,
        calculate_contingency_scr_batch_rust,
    ],
    load = load
);
```

---

## Tests

### Elixir Tests: `test/power_flow_solver/scr_contingency_test.exs`

```elixir
defmodule PowerFlowSolver.SCR.ContingencyTest do
  use ExUnit.Case, async: true

  alias PowerFlowSolver.SCR

  describe "calculate_contingency/2" do
    test "returns reduced S_sc when branch is removed" do
      system = build_3bus_system()

      # Base case
      {:ok, base} = SCR.calculate_at_bus(system, 2, 100.0)
      s_sc_base = base.short_circuit_mva

      # With contingency (remove branch 0-1)
      {:ok, contingency} = SCR.calculate_contingency(system,
        poi_bus_id: 2,
        branch: %{from_bus: 0, to_bus: 1}
      )

      # Contingency should reduce S_sc or cause islanding
      if contingency.success do
        assert contingency.short_circuit_mva < s_sc_base
      end
    end

    test "handles islanding gracefully" do
      system = build_radial_system()

      # Remove the only connection to bus 2
      {:ok, result} = SCR.calculate_contingency(system,
        poi_bus_id: 2,
        branch: %{from_bus: 1, to_bus: 2}
      )

      refute result.success
    end
  end

  describe "calculate_contingency_batch/2" do
    test "returns results sorted by S_sc" do
      system = build_meshed_system()

      {:ok, results} = SCR.calculate_contingency_batch(system,
        poi_bus_id: 5,
        branches: :all
      )

      # Should be sorted weakest first
      s_sc_values = Enum.map(results, & &1.short_circuit_mva)
      assert s_sc_values == Enum.sort(s_sc_values)
    end

    test "filters out failed contingencies" do
      system = build_3bus_system()

      {:ok, results} = SCR.calculate_contingency_batch(system,
        poi_bus_id: 2,
        branches: :all
      )

      # All results should be successful
      assert Enum.all?(results, & &1.success)
    end
  end

  describe "find_worst_contingency/2" do
    test "returns contingency with lowest SCR" do
      system = build_meshed_system()

      {:ok, worst} = SCR.find_worst_contingency(system,
        poi_bus_id: 5,
        p_rated_mw: 100.0
      )

      assert worst.scr > 0
      assert worst.grid_strength in [:weak, :moderate, :strong, :very_strong]
    end

    test "returns nil when all contingencies cause islanding" do
      system = build_radial_system()

      {:ok, result} = SCR.find_worst_contingency(system,
        poi_bus_id: 2,
        p_rated_mw: 100.0,
        branches: [%{from_bus: 1, to_bus: 2}]
      )

      assert is_nil(result)
    end
  end

  # Test helpers
  defp build_3bus_system do
    %{
      buses: [
        %{id: 0, type: :slack, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0},
        %{id: 1, type: :pq, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.5, q_load: 0.2, p_gen: 0.0, q_gen: 0.0},
        %{id: 2, type: :pq, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.3, q_load: 0.1, p_gen: 0.0, q_gen: 0.0}
      ],
      branches: [
        %{id: 0, from_bus: 0, to_bus: 1, r: 0.01, x: 0.1, b: 0.02},
        %{id: 1, from_bus: 1, to_bus: 2, r: 0.01, x: 0.1, b: 0.02}
      ]
    }
  end

  defp build_radial_system do
    # Same as 3-bus but explicitly radial
    build_3bus_system()
  end

  defp build_meshed_system do
    # 5-bus system with mesh (multiple paths)
    %{
      buses: [
        %{id: 0, type: :slack, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.0, q_load: 0.0, p_gen: 0.0, q_gen: 0.0},
        %{id: 1, type: :pq, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.2, q_load: 0.1, p_gen: 0.0, q_gen: 0.0},
        %{id: 2, type: :pq, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.2, q_load: 0.1, p_gen: 0.0, q_gen: 0.0},
        %{id: 3, type: :pq, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.3, q_load: 0.15, p_gen: 0.0, q_gen: 0.0},
        %{id: 4, type: :pq, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.4, q_load: 0.2, p_gen: 0.0, q_gen: 0.0},
        %{id: 5, type: :pq, v_magnitude: 1.0, v_angle: 0.0, p_load: 0.5, q_load: 0.25, p_gen: 0.0, q_gen: 0.0}
      ],
      branches: [
        %{id: 0, from_bus: 0, to_bus: 1, r: 0.01, x: 0.1, b: 0.02},
        %{id: 1, from_bus: 0, to_bus: 2, r: 0.01, x: 0.1, b: 0.02},
        %{id: 2, from_bus: 1, to_bus: 3, r: 0.01, x: 0.1, b: 0.02},
        %{id: 3, from_bus: 2, to_bus: 3, r: 0.01, x: 0.1, b: 0.02},
        %{id: 4, from_bus: 3, to_bus: 4, r: 0.01, x: 0.1, b: 0.02},
        %{id: 5, from_bus: 3, to_bus: 5, r: 0.01, x: 0.1, b: 0.02},
        %{id: 6, from_bus: 4, to_bus: 5, r: 0.01, x: 0.1, b: 0.02}
      ]
    }
  end
end
```

---

## Summary

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `native/power_flow_solver/src/contingency_scr.rs` | Create | Rust contingency SCR logic |
| `native/power_flow_solver/src/lib.rs` | Modify | Add NIF registrations |
| `lib/power_flow_solver/scr.ex` | Modify | Add contingency functions |
| `test/power_flow_solver/scr_contingency_test.exs` | Create | Elixir tests |

### API Surface (New Functions)

```elixir
# Single contingency
PowerFlowSolver.SCR.calculate_contingency(system, poi_bus_id: 5, branch: branch)

# Batch contingency scan
PowerFlowSolver.SCR.calculate_contingency_batch(system, poi_bus_id: 5, branches: :all)

# Find worst case
PowerFlowSolver.SCR.find_worst_contingency(system, poi_bus_id: 5, p_rated_mw: 100.0)
```

### What This Does NOT Include

- FRT assessment logic (gridvar)
- Voltage recovery formulas (gridvar)
- PLL stability thresholds (gridvar)
- Mitigation recommendations (gridvar)
- Facility/plant domain types (gridvar)
- Report generation (gridvar)
