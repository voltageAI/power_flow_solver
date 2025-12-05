use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::HashMap;

/// Bus type for power flow
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BusType {
    Slack = 0,
    PV = 1,
    PQ = 2,
}

impl From<u8> for BusType {
    fn from(value: u8) -> Self {
        match value {
            0 => BusType::Slack,
            1 => BusType::PV,
            2 => BusType::PQ,
            _ => BusType::PQ,
        }
    }
}

/// Bus data for power flow calculations
#[derive(Clone, Debug)]
pub struct BusData {
    pub bus_type: BusType,
    pub p_scheduled: f64,
    pub q_scheduled: f64,
    #[allow(dead_code)]
    pub v_scheduled: f64,
    pub q_min: Option<f64>,
    pub q_max: Option<f64>,
    pub original_type: Option<BusType>,
    pub q_load: f64,  // Needed to convert Q injection to Q generation
}

/// Power system structure
pub struct PowerSystem {
    #[allow(dead_code)]
    pub n_buses: usize,
    pub y_bus: YBusData,
    pub buses: Vec<BusData>,
}

/// Y-bus matrix in CSR format
pub struct YBusData {
    pub row_ptrs: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<Complex64>,
}

/// Solver configuration
#[derive(Clone, Debug)]
pub struct SolverConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub enforce_q_limits: bool,
    pub q_tolerance: f64,
}

/// Power flow result
pub struct PowerFlowResult {
    pub voltage: Vec<(f64, f64)>, // (magnitude, angle) pairs
    pub q_generation: Vec<f64>,   // Q generation at each bus (computed for PV/slack)
    pub iterations: usize,
    pub converged: bool,
    pub final_mismatch: f64,
}

/// Main power flow solver
///
/// This implements the complete Newton-Raphson iteration loop in Rust,
/// eliminating boundary crossings with Elixir.
pub fn solve_power_flow(
    mut system: PowerSystem,
    initial_voltage: Vec<(f64, f64)>,
    config: SolverConfig,
) -> Result<PowerFlowResult, String> {

    // Debug logging (commented out for cleaner test output)
    // eprintln!("\n=== RUST POWER FLOW SOLVER DEBUG ===");
    // eprintln!("System: {} buses, max_iter={}, tol={}",
    //     system.n_buses, config.max_iterations, config.tolerance);

    // Convert voltage to complex numbers
    let mut voltage: Vec<Complex64> = initial_voltage
        .iter()
        .map(|&(mag, ang)| Complex64::new(mag * ang.cos(), mag * ang.sin()))
        .collect();

    let mut iteration = 0;
    let mut mismatch_norm = f64::MAX;
    let mut converged = false;

    // Pre-compute variable indices (mutable for Q-limit adjustments)
    let (mut angle_vars, mut vmag_vars) = determine_variables(&system.buses);

    // Create symbolic LU factorization once
    // (This will use the existing create_symbolic_lu function)

    while iteration < config.max_iterations {
        iteration += 1;

        // 1. Calculate power injections
        let power_injections = compute_all_power_injections(&system.y_bus, &voltage);

        // Debug logging (commented out)
        // if iteration == 1 {
        //     eprintln!("Power injections (first 3 buses):");
        // }

        // 2. Build Jacobian matrix
        let jacobian = build_jacobian(
            &system,
            &voltage,
            &power_injections,
            &angle_vars,
            &vmag_vars,
        )?;

        // Debug logging (commented out)

        // 3. Calculate mismatch vector
        let mismatch = calculate_mismatch(
            &system,
            &voltage,
            &power_injections,
            &angle_vars,
            &vmag_vars,
        )?;

        // 4. Check convergence
        mismatch_norm = calculate_norm(&mismatch);

        if mismatch_norm < config.tolerance {
            converged = true;
            break;
        }

        // 5. Solve linear system: J * delta = mismatch
        // (J = ∂S_calc/∂x, mismatch = S_sched - S_calc, so J*Δx = mismatch)
        let delta = solve_sparse_system_direct(&jacobian, &mismatch)?;

        // 6. Line search to find optimal step size
        let step_size = line_search_backtracking(
            &voltage,
            &delta,
            &angle_vars,
            &vmag_vars,
            mismatch_norm,
            &system.buses,
            &system.y_bus,
        );

        // 7. Update voltage with adaptive step size
        update_voltage_with_step_size(&mut voltage, &delta, &angle_vars, &vmag_vars, step_size);

        // Debug logging (commented out)

        // 8. Check and enforce Q-limits
        if config.enforce_q_limits {
            let q_changes = check_and_enforce_q_limits(
                &mut system.buses,
                &system.y_bus,
                &voltage,
                config.q_tolerance,
            );

            if !q_changes.is_empty() {
                // Debug logging (commented out)
                // eprintln!("Q-limit violations detected: {} buses changed", q_changes.len());

                // Rebuild variable indices after bus type changes
                angle_vars = determine_variables(&system.buses).0;
                vmag_vars = determine_variables(&system.buses).1;
            }
        }
    }

    // Convert voltage back to (magnitude, angle) format
    let final_voltage: Vec<(f64, f64)> = voltage
        .iter()
        .map(|v| {
            let mag = v.norm();
            let ang = v.arg();
            (mag, ang)
        })
        .collect();

    // Calculate Q generation for all buses
    // For PV and slack buses, Q is not a variable, so we compute it from the solved voltages
    // Q_gen = Q_injection + Q_load (where Q_injection comes from power flow equations)
    let q_injections = compute_q_injections(&system.y_bus, &voltage);
    let q_generation: Vec<f64> = system.buses
        .iter()
        .enumerate()
        .map(|(i, bus)| {
            // Q_gen = Q_injection + Q_load
            // Q_injection is the net reactive power flowing out of the bus
            // Q_load is the reactive load at the bus (positive = consuming)
            // So Q_gen = Q_injection + Q_load gives us the generation needed
            q_injections[i] + bus.q_load
        })
        .collect();

    Ok(PowerFlowResult {
        voltage: final_voltage,
        q_generation,
        iterations: iteration,
        converged,
        final_mismatch: mismatch_norm,
    })
}

/// Determine which buses have angle and voltage magnitude variables
pub fn determine_variables(buses: &[BusData]) -> (Vec<usize>, Vec<usize>) {
    let mut angle_vars = Vec::new();
    let mut vmag_vars = Vec::new();

    for (i, bus) in buses.iter().enumerate() {
        match bus.bus_type {
            BusType::Slack => {
                // Slack: no variables (both V and theta are specified)
            }
            BusType::PV => {
                // PV bus: theta is a variable
                angle_vars.push(i);
            }
            BusType::PQ => {
                // PQ bus: both theta and |V| are variables
                angle_vars.push(i);
                vmag_vars.push(i);
            }
        }
    }

    (angle_vars, vmag_vars)
}

/// Compute power injections for all buses
fn compute_all_power_injections(
    y_bus: &YBusData,
    voltage: &[Complex64],
) -> Vec<(f64, f64)> {
    (0..voltage.len())
        .into_par_iter()
        .map(|i| compute_power_injection(y_bus, voltage, i))
        .collect()
}

/// Compute power injection (P, Q) for a single bus
fn compute_power_injection(
    y_bus: &YBusData,
    voltage: &[Complex64],
    bus_idx: usize,
) -> (f64, f64) {
    let v_i = voltage[bus_idx];
    let row_start = y_bus.row_ptrs[bus_idx];
    let row_end = y_bus.row_ptrs[bus_idx + 1];

    let mut p_sum = 0.0;
    let mut q_sum = 0.0;

    for idx in row_start..row_end {
        let k = y_bus.col_indices[idx];
        let y_ik = y_bus.values[idx];
        let v_k = voltage[k];

        // S_i = V_i * conj(I_i) = V_i * conj(sum(Y_ik * V_k))
        // Standard power flow convention: S = V * conj(I)
        let s_contribution = v_i * (y_ik * v_k).conj();
        p_sum += s_contribution.re;
        q_sum += s_contribution.im;
    }

    (p_sum, q_sum)
}

/// Build Jacobian matrix
fn build_jacobian(
    system: &PowerSystem,
    voltage: &[Complex64],
    power_injections: &[(f64, f64)],
    angle_vars: &[usize],
    vmag_vars: &[usize],
) -> Result<SparseMatrix, String> {
    let num_p_eqs = angle_vars.len();
    let num_q_eqs = vmag_vars.len();
    let num_vars = num_p_eqs + num_q_eqs;

    // Build Y-bus lookup map for fast access
    let y_bus_map = build_ybus_map(&system.y_bus);

    // Build Jacobian rows in parallel
    let all_rows: Vec<Vec<(usize, usize, f64)>> = (0..num_vars)
        .into_par_iter()
        .map(|eq_idx| {
            build_jacobian_row(
                eq_idx,
                voltage,
                power_injections,
                &y_bus_map,
                angle_vars,
                vmag_vars,
                num_p_eqs,
            )
        })
        .collect();

    // Convert to CSR format
    let mut row_ptrs = vec![0];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for mut row_triplets in all_rows {
        // Sort by column index (required for CSR)
        row_triplets.sort_by_key(|(_, col, _)| *col);

        for (_, col, val) in row_triplets {
            col_indices.push(col);
            values.push(val);
        }
        row_ptrs.push(col_indices.len());
    }

    Ok(SparseMatrix {
        row_ptrs,
        col_indices,
        values,
    })
}

pub struct SparseMatrix {
    pub row_ptrs: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
}

/// Build a fast lookup map for Y-bus elements
fn build_ybus_map(y_bus: &YBusData) -> HashMap<(usize, usize), Complex64> {
    let mut map = HashMap::new();

    for i in 0..y_bus.row_ptrs.len() - 1 {
        let row_start = y_bus.row_ptrs[i];
        let row_end = y_bus.row_ptrs[i + 1];

        for idx in row_start..row_end {
            let j = y_bus.col_indices[idx];
            let y_ij = y_bus.values[idx];
            map.insert((i, j), y_ij);
        }
    }

    map
}

/// Build a single row of the Jacobian matrix
fn build_jacobian_row(
    eq_idx: usize,
    voltage: &[Complex64],
    power_injections: &[(f64, f64)],
    y_bus_map: &HashMap<(usize, usize), Complex64>,
    angle_vars: &[usize],
    vmag_vars: &[usize],
    num_p_eqs: usize,
) -> Vec<(usize, usize, f64)> {
    let mut triplets = Vec::new();
    let is_p_equation = eq_idx < num_p_eqs;

    if is_p_equation {
        // P equation
        let bus_i = angle_vars[eq_idx];
        let v_i = voltage[bus_i];
        let (p_i, q_i) = power_injections[bus_i];

        // ∂P/∂θ elements
        for (var_offset, &bus_k) in angle_vars.iter().enumerate() {
            let value = if bus_i == bus_k {
                // Diagonal: ∂P_i/∂θ_i = -Q_i - V_i² * B_ii
                let y_ii = y_bus_map.get(&(bus_i, bus_i)).copied().unwrap_or(Complex64::new(0.0, 0.0));
                -q_i - v_i.norm_sqr() * y_ii.im
            } else {
                // Off-diagonal: ∂P_i/∂θ_k = V_i * V_k * (G_ik * sin(θ_ik) - B_ik * cos(θ_ik))
                if let Some(&y_ik) = y_bus_map.get(&(bus_i, bus_k)) {
                    let v_k = voltage[bus_k];
                    let theta_ik = v_i.arg() - v_k.arg();
                    let g_ik = y_ik.re;
                    let b_ik = y_ik.im;
                    v_i.norm() * v_k.norm() * (g_ik * theta_ik.sin() - b_ik * theta_ik.cos())
                } else {
                    0.0
                }
            };

            if value.abs() > 1e-15 {
                triplets.push((eq_idx, var_offset, value));
            }
        }

        // ∂P/∂V elements
        for (var_offset, &bus_k) in vmag_vars.iter().enumerate() {
            let value = if bus_i == bus_k {
                // Diagonal: ∂P_i/∂V_i = P_i/V_i + V_i * G_ii
                let y_ii = y_bus_map.get(&(bus_i, bus_i)).copied().unwrap_or(Complex64::new(0.0, 0.0));
                p_i / v_i.norm() + v_i.norm() * y_ii.re
            } else {
                // Off-diagonal: ∂P_i/∂V_k = V_i * (G_ik * cos(θ_ik) + B_ik * sin(θ_ik))
                if let Some(&y_ik) = y_bus_map.get(&(bus_i, bus_k)) {
                    let v_k = voltage[bus_k];
                    let theta_ik = v_i.arg() - v_k.arg();
                    let g_ik = y_ik.re;
                    let b_ik = y_ik.im;
                    v_i.norm() * (g_ik * theta_ik.cos() + b_ik * theta_ik.sin())
                } else {
                    0.0
                }
            };

            if value.abs() > 1e-15 {
                let col_idx = num_p_eqs + var_offset;
                triplets.push((eq_idx, col_idx, value));
            }
        }
    } else {
        // Q equation
        let q_eq_idx = eq_idx - num_p_eqs;
        let bus_i = vmag_vars[q_eq_idx];
        let v_i = voltage[bus_i];
        let (p_i, q_i) = power_injections[bus_i];

        // ∂Q/∂θ elements
        for (var_offset, &bus_k) in angle_vars.iter().enumerate() {
            let value = if bus_i == bus_k {
                // Diagonal: ∂Q_i/∂θ_i = P_i - V_i² * G_ii
                let y_ii = y_bus_map.get(&(bus_i, bus_i)).copied().unwrap_or(Complex64::new(0.0, 0.0));
                p_i - v_i.norm_sqr() * y_ii.re
            } else {
                // Off-diagonal: ∂Q_i/∂θ_k = -V_i * V_k * (G_ik * cos(θ_ik) + B_ik * sin(θ_ik))
                if let Some(&y_ik) = y_bus_map.get(&(bus_i, bus_k)) {
                    let v_k = voltage[bus_k];
                    let theta_ik = v_i.arg() - v_k.arg();
                    let g_ik = y_ik.re;
                    let b_ik = y_ik.im;
                    -v_i.norm() * v_k.norm() * (g_ik * theta_ik.cos() + b_ik * theta_ik.sin())
                } else {
                    0.0
                }
            };

            if value.abs() > 1e-15 {
                triplets.push((eq_idx, var_offset, value));
            }
        }

        // ∂Q/∂V elements
        for (var_offset, &bus_k) in vmag_vars.iter().enumerate() {
            let value = if bus_i == bus_k {
                // Diagonal: ∂Q_i/∂V_i = Q_i/V_i - V_i * B_ii
                let y_ii = y_bus_map.get(&(bus_i, bus_i)).copied().unwrap_or(Complex64::new(0.0, 0.0));
                q_i / v_i.norm() - v_i.norm() * y_ii.im
            } else {
                // Off-diagonal: ∂Q_i/∂V_k = V_i * (G_ik * sin(θ_ik) - B_ik * cos(θ_ik))
                if let Some(&y_ik) = y_bus_map.get(&(bus_i, bus_k)) {
                    let v_k = voltage[bus_k];
                    let theta_ik = v_i.arg() - v_k.arg();
                    let g_ik = y_ik.re;
                    let b_ik = y_ik.im;
                    v_i.norm() * (g_ik * theta_ik.sin() - b_ik * theta_ik.cos())
                } else {
                    0.0
                }
            };

            if value.abs() > 1e-15 {
                let col_idx = num_p_eqs + var_offset;
                triplets.push((eq_idx, col_idx, value));
            }
        }
    }

    triplets
}

/// Calculate mismatch vector
fn calculate_mismatch(
    system: &PowerSystem,
    _voltage: &[Complex64],
    power_injections: &[(f64, f64)],
    angle_vars: &[usize],
    vmag_vars: &[usize],
) -> Result<Vec<f64>, String> {
    let mut mismatch = Vec::new();

    // P mismatches
    for &bus_i in angle_vars {
        let (p_calc, _) = power_injections[bus_i];
        let p_sched = system.buses[bus_i].p_scheduled;
        mismatch.push(p_sched - p_calc);
    }

    // Q mismatches
    for &bus_i in vmag_vars {
        let (_, q_calc) = power_injections[bus_i];
        let q_sched = system.buses[bus_i].q_scheduled;
        mismatch.push(q_sched - q_calc);
    }

    Ok(mismatch)
}

/// Calculate Euclidean norm of vector
fn calculate_norm(vec: &[f64]) -> f64 {
    vec.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Update voltage from delta corrections
/// Convention: J * delta = -mismatch, then x_new = x_old + delta
#[allow(dead_code)]
fn update_voltage(
    voltage: &mut [Complex64],
    delta: &[f64],
    angle_vars: &[usize],
    vmag_vars: &[usize],
) {
    update_voltage_with_step_size(voltage, delta, angle_vars, vmag_vars, 1.0);
}

/// Update voltage from delta corrections with adaptive step size
/// step_size: multiplier for delta (0 < step_size <= 1.0)
fn update_voltage_with_step_size(
    voltage: &mut [Complex64],
    delta: &[f64],
    angle_vars: &[usize],
    vmag_vars: &[usize],
    step_size: f64,
) {
    // Update angles (ADD step_size * delta)
    for (i, &bus_idx) in angle_vars.iter().enumerate() {
        let d_theta = step_size * delta[i];
        let mag = voltage[bus_idx].norm();
        let theta = voltage[bus_idx].arg() + d_theta;
        voltage[bus_idx] = Complex64::new(mag * theta.cos(), mag * theta.sin());
    }

    // Update magnitudes (ADD step_size * delta)
    let num_p_eqs = angle_vars.len();
    for (i, &bus_idx) in vmag_vars.iter().enumerate() {
        let d_mag = step_size * delta[num_p_eqs + i];
        let mag = voltage[bus_idx].norm() + d_mag;
        let theta = voltage[bus_idx].arg();
        voltage[bus_idx] = Complex64::new(mag * theta.cos(), mag * theta.sin());
    }
}

/// Line search backtracking to find optimal step size
///
/// Tries full Newton step (step_size = 1.0) first. If mismatch increases,
/// reduces step size by factor of 4 and retries until mismatch decreases.
///
/// Returns: step_size in range (0, 1.0]
fn line_search_backtracking(
    voltage: &[Complex64],
    delta: &[f64],
    angle_vars: &[usize],
    vmag_vars: &[usize],
    current_mismatch_norm: f64,
    buses: &[BusData],
    y_bus: &YBusData,
) -> f64 {
    const MAX_TRIALS: usize = 5;
    const REDUCTION_FACTOR: f64 = 0.25; // Divide step size by 4 each trial

    let mut step_size = 1.0;

    for _trial in 0..MAX_TRIALS {
        // Create trial voltage with this step size
        let mut trial_voltage = voltage.to_vec();
        update_voltage_with_step_size(&mut trial_voltage, delta, angle_vars, vmag_vars, step_size);

        // Compute power injections at trial voltage
        let trial_power = compute_all_power_injections(y_bus, &trial_voltage);

        // Compute mismatch at trial voltage
        let mut trial_mismatch = Vec::new();

        // P mismatches
        for &bus_i in angle_vars {
            let (p_calc, _) = trial_power[bus_i];
            let p_sched = buses[bus_i].p_scheduled;
            trial_mismatch.push(p_sched - p_calc);
        }

        // Q mismatches
        for &bus_i in vmag_vars {
            let (_, q_calc) = trial_power[bus_i];
            let q_sched = buses[bus_i].q_scheduled;
            trial_mismatch.push(q_sched - q_calc);
        }

        let trial_mismatch_norm = calculate_norm(&trial_mismatch);

        // Accept step if mismatch decreased
        if trial_mismatch_norm < current_mismatch_norm {
            return step_size;
        }

        // Reduce step size and try again
        step_size *= REDUCTION_FACTOR;
    }

    // If all trials failed, return smallest step size tried
    // This prevents divergence - small step is safer than full step
    step_size
}

/// Solve sparse linear system using faer
fn solve_sparse_system_direct(
    matrix: &SparseMatrix,
    rhs: &[f64],
) -> Result<Vec<f64>, String> {
    use faer::prelude::*;
    use faer::sparse::linalg::solvers::{SymbolicLu, Lu};

    let n = matrix.row_ptrs.len() - 1;

    if rhs.len() != n {
        return Err(format!("Dimension mismatch: matrix {}x{}, RHS {}", n, n, rhs.len()));
    }

    // Build triplets for faer
    let mut triplets = Vec::new();
    for i in 0..n {
        let row_start = matrix.row_ptrs[i];
        let row_end = matrix.row_ptrs[i + 1];

        for idx in row_start..row_end {
            let j = matrix.col_indices[idx];
            let val = matrix.values[idx];
            triplets.push((i, j, val));
        }
    }

    // Build faer sparse matrix
    let sparse_mat = faer::sparse::SparseColMat::try_new_from_triplets(
        n,
        n,
        &triplets,
    ).map_err(|e| format!("Failed to create sparse matrix: {:?}", e))?;

    // Convert RHS to faer format
    let mut b_faer = faer::Mat::zeros(n, 1);
    for (i, &val) in rhs.iter().enumerate() {
        b_faer.write(i, 0, val);
    }

    // Symbolic factorization
    let symbolic = SymbolicLu::try_new(sparse_mat.symbolic())
        .map_err(|e| format!("Symbolic factorization failed: {:?}", e))?;

    // Numeric factorization
    let lu = Lu::try_new_with_symbolic(symbolic, sparse_mat.as_ref())
        .map_err(|e| format!("LU factorization failed: {:?}", e))?;

    // Solve
    let x_faer = lu.solve(b_faer.as_ref());

    // Convert solution back
    let solution: Vec<f64> = (0..n)
        .map(|i| x_faer.read(i, 0))
        .collect();

    Ok(solution)
}

/// Check and enforce Q-limits on PV buses
/// Returns: Vector of bus indices that changed type
fn check_and_enforce_q_limits(
    buses: &mut [BusData],
    y_bus: &YBusData,
    voltage: &[Complex64],
    q_tolerance: f64,
) -> Vec<usize> {
    const MAX_CONVERSIONS_PER_ITERATION: usize = 50;

    // Calculate Q injection for all buses
    let q_injections = compute_q_injections(y_bus, voltage);

    // Collect violations with their severity
    let mut violations: Vec<(usize, f64, bool)> = Vec::new();  // (bus_idx, severity, is_max_violation)

    for (bus_idx, bus) in buses.iter().enumerate() {
        // Only check PV buses with Q-limits defined
        if bus.bus_type != BusType::PV {
            continue;
        }

        if let (Some(q_min), Some(q_max)) = (bus.q_min, bus.q_max) {
            let q_injection = q_injections[bus_idx];
            // Convert injection to generation: Q_gen = Q_injection + Q_load
            let q_gen = q_injection + bus.q_load;

            // Check if Q generation exceeds limits
            if q_gen > q_max + q_tolerance {
                let severity = q_gen - q_max;
                violations.push((bus_idx, severity, true));
            } else if q_gen < q_min - q_tolerance {
                let severity = q_min - q_gen;
                violations.push((bus_idx, severity, false));
            }
        }
    }

    // Sort by severity (descending) to handle worst violations first
    violations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Limit number of conversions per iteration
    let violations_to_apply = violations.iter().take(MAX_CONVERSIONS_PER_ITERATION);
    let mut changed_buses = Vec::new();

    for &(bus_idx, _, is_max_violation) in violations_to_apply {
        let bus = &mut buses[bus_idx];
        let q_injection = q_injections[bus_idx];
        let q_gen = q_injection + bus.q_load;

        bus.original_type = Some(BusType::PV);
        bus.bus_type = BusType::PQ;

        if is_max_violation {
            bus.q_scheduled = bus.q_max.unwrap() - bus.q_load;
            eprintln!("  Bus {}: Q_gen={:.4} > Q_max={:.4}, converting PV->PQ",
                bus_idx, q_gen, bus.q_max.unwrap());
        } else {
            bus.q_scheduled = bus.q_min.unwrap() - bus.q_load;
            eprintln!("  Bus {}: Q_gen={:.4} < Q_min={:.4}, converting PV->PQ",
                bus_idx, q_gen, bus.q_min.unwrap());
        }

        changed_buses.push(bus_idx);
    }

    if violations.len() > MAX_CONVERSIONS_PER_ITERATION {
        eprintln!("  (Limited to {} of {} violations this iteration)",
            MAX_CONVERSIONS_PER_ITERATION, violations.len());
    }

    changed_buses
}

/// Compute Q injection for all buses
fn compute_q_injections(y_bus: &YBusData, voltage: &[Complex64]) -> Vec<f64> {
    let n_buses = voltage.len();
    let mut q_injections = vec![0.0; n_buses];

    for i in 0..n_buses {
        let v_i = voltage[i];
        let row_start = y_bus.row_ptrs[i];
        let row_end = y_bus.row_ptrs[i + 1];

        let mut sum = Complex64::new(0.0, 0.0);
        for idx in row_start..row_end {
            let k = y_bus.col_indices[idx];
            let y_ik = y_bus.values[idx];
            let v_k = voltage[k];

            // I_i = sum(Y_ik * V_k)
            sum += y_ik * v_k;
        }

        // S = V_i * conj(I_i), Q = Im(S)
        let s_contribution = v_i * sum.conj();
        q_injections[i] = s_contribution.im;
    }

    q_injections
}

/// Jacobian validation result
pub struct JacobianValidation {
    pub max_error: f64,
    pub mean_error: f64,
    pub errors: Vec<(usize, usize, f64, f64, f64)>,
}

/// Validate analytical Jacobian against numerical (finite difference) Jacobian
/// Returns: (max_error, avg_error, num_large_errors, detailed_error_strings)
pub fn validate_jacobian(
    system: &PowerSystem,
    voltage: &[Complex64],
    angle_vars: &[usize],
    vmag_vars: &[usize],
    epsilon: f64,
) -> Result<(f64, f64, usize, Vec<String>), String> {
    let validation = validate_jacobian_numerical(system, voltage, angle_vars, vmag_vars, epsilon)?;

    // Convert to simpler format for NIF return
    let mut error_strings = Vec::new();
    let num_p_eqs = angle_vars.len();

    for (_i, &(row, col, analytical, numerical, error)) in validation.errors.iter().take(20).enumerate() {
        let eq_type = if row < num_p_eqs { "P" } else { "Q" };
        let eq_bus = if row < num_p_eqs {
            angle_vars[row]
        } else {
            vmag_vars[row - num_p_eqs]
        };

        let var_type = if col < num_p_eqs { "θ" } else { "V" };
        let var_bus = if col < num_p_eqs {
            angle_vars[col]
        } else {
            vmag_vars[col - num_p_eqs]
        };

        error_strings.push(format!(
            "∂{}{}/∂{}{}: analytical={:.6e}, numerical={:.6e}, error={:.2}%",
            eq_type, eq_bus, var_type, var_bus,
            analytical, numerical, error * 100.0
        ));
    }

    Ok((validation.max_error, validation.mean_error, validation.errors.len(), error_strings))
}

/// Validate analytical Jacobian against numerical (finite difference) Jacobian
///
/// Returns comparison data showing element-by-element differences.
/// Uses central difference formula for better accuracy:
/// df/dx ≈ (f(x + h) - f(x - h)) / (2h)
fn validate_jacobian_numerical(
    system: &PowerSystem,
    voltage: &[Complex64],
    angle_vars: &[usize],
    vmag_vars: &[usize],
    epsilon: f64,
) -> Result<JacobianValidation, String> {
    eprintln!("\n=== JACOBIAN NUMERICAL VALIDATION ===");
    eprintln!("Using central differences with epsilon = {:.2e}", epsilon);

    // Build analytical Jacobian
    let power_injections = compute_all_power_injections(&system.y_bus, voltage);
    let analytical = build_jacobian(
        system,
        voltage,
        &power_injections,
        angle_vars,
        vmag_vars,
    )?;

    let num_p_eqs = angle_vars.len();
    let num_q_eqs = vmag_vars.len();
    let num_vars = num_p_eqs + num_q_eqs;

    eprintln!("Matrix dimensions: {}x{} ({} P eqs, {} Q eqs)",
        num_vars, num_vars, num_p_eqs, num_q_eqs);

    // Convert analytical Jacobian to dense format for easy comparison
    let mut analytical_dense = vec![vec![0.0; num_vars]; num_vars];
    for row in 0..num_vars {
        let row_start = analytical.row_ptrs[row];
        let row_end = analytical.row_ptrs[row + 1];
        for idx in row_start..row_end {
            let col = analytical.col_indices[idx];
            let val = analytical.values[idx];
            analytical_dense[row][col] = val;
        }
    }

    // Build numerical Jacobian using finite differences
    let numerical_dense = compute_numerical_jacobian(
        system,
        voltage,
        angle_vars,
        vmag_vars,
        epsilon,
    );

    // Compare element by element
    let mut errors = Vec::new();
    let mut sum_error = 0.0;
    let mut max_error = 0.0;
    let mut count = 0;

    for row in 0..num_vars {
        for col in 0..num_vars {
            let analytical_val = analytical_dense[row][col];
            let numerical_val = numerical_dense[row][col];

            // Calculate relative error, handling near-zero values
            let abs_diff = (analytical_val - numerical_val).abs();
            let max_abs = analytical_val.abs().max(numerical_val.abs());

            // Use relative error for large values, absolute error for small values
            let error = if max_abs > 1e-6 {
                abs_diff / max_abs
            } else {
                abs_diff
            };

            sum_error += error;
            count += 1;

            if error > max_error {
                max_error = error;
            }

            // Store significant errors (>0.1% relative or >1e-6 absolute)
            if error > 1e-3 || abs_diff > 1e-6 {
                errors.push((row, col, analytical_val, numerical_val, error));
            }
        }
    }

    let mean_error = sum_error / count as f64;

    eprintln!("Validation complete:");
    eprintln!("  Max error: {:.6e}", max_error);
    eprintln!("  Mean error: {:.6e}", mean_error);
    eprintln!("  Significant discrepancies: {}", errors.len());

    // Sort errors by magnitude (largest first)
    errors.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));

    // Print top 10 errors
    eprintln!("\nTop 10 largest errors:");
    for (i, &(row, col, analytical_val, numerical_val, error)) in errors.iter().take(10).enumerate() {
        let eq_type = if row < num_p_eqs { "P" } else { "Q" };
        let eq_bus = if row < num_p_eqs {
            angle_vars[row]
        } else {
            vmag_vars[row - num_p_eqs]
        };

        let var_type = if col < num_p_eqs { "θ" } else { "V" };
        let var_bus = if col < num_p_eqs {
            angle_vars[col]
        } else {
            vmag_vars[col - num_p_eqs]
        };

        eprintln!("  {}. ∂{}{}/∂{}{}: analytical={:.6e}, numerical={:.6e}, error={:.6e}",
            i + 1, eq_type, eq_bus, var_type, var_bus,
            analytical_val, numerical_val, error);
    }

    Ok(JacobianValidation {
        max_error,
        mean_error,
        errors,
    })
}

/// Compute numerical Jacobian using central finite differences
fn compute_numerical_jacobian(
    system: &PowerSystem,
    voltage: &[Complex64],
    angle_vars: &[usize],
    vmag_vars: &[usize],
    epsilon: f64,
) -> Vec<Vec<f64>> {
    let num_p_eqs = angle_vars.len();
    let num_q_eqs = vmag_vars.len();
    let num_vars = num_p_eqs + num_q_eqs;

    let mut jacobian = vec![vec![0.0; num_vars]; num_vars];

    // Compute base mismatch
    let base_power = compute_all_power_injections(&system.y_bus, voltage);
    let _base_mismatch = compute_mismatch_vec(system, &base_power, angle_vars, vmag_vars);

    // Perturb each variable and compute derivatives
    for var_idx in 0..num_vars {
        // Determine which variable we're perturbing
        let (is_angle, bus_idx) = if var_idx < num_p_eqs {
            (true, angle_vars[var_idx])
        } else {
            (false, vmag_vars[var_idx - num_p_eqs])
        };

        // Forward perturbation: x + epsilon
        let mut voltage_plus = voltage.to_vec();
        perturb_voltage(&mut voltage_plus, bus_idx, is_angle, epsilon);
        let power_plus = compute_all_power_injections(&system.y_bus, &voltage_plus);
        let mismatch_plus = compute_mismatch_vec(system, &power_plus, angle_vars, vmag_vars);

        // Backward perturbation: x - epsilon
        let mut voltage_minus = voltage.to_vec();
        perturb_voltage(&mut voltage_minus, bus_idx, is_angle, -epsilon);
        let power_minus = compute_all_power_injections(&system.y_bus, &voltage_minus);
        let mismatch_minus = compute_mismatch_vec(system, &power_minus, angle_vars, vmag_vars);

        // Central difference: (f(x+h) - f(x-h)) / (2h)
        // BUT: We're computing derivatives of mismatch, and the Jacobian
        // represents how mismatch changes with variables, so we need:
        // J[eq, var] = d(mismatch[eq])/d(var)
        for eq_idx in 0..num_vars {
            jacobian[eq_idx][var_idx] =
                (mismatch_plus[eq_idx] - mismatch_minus[eq_idx]) / (2.0 * epsilon);
        }
    }

    jacobian
}

/// Perturb voltage at a specific bus
fn perturb_voltage(
    voltage: &mut [Complex64],
    bus_idx: usize,
    is_angle: bool,
    delta: f64,
) {
    let v = voltage[bus_idx];
    let mag = v.norm();
    let theta = v.arg();

    if is_angle {
        // Perturb angle
        let new_theta = theta + delta;
        voltage[bus_idx] = Complex64::new(mag * new_theta.cos(), mag * new_theta.sin());
    } else {
        // Perturb magnitude
        let new_mag = mag + delta;
        voltage[bus_idx] = Complex64::new(new_mag * theta.cos(), new_mag * theta.sin());
    }
}

/// Compute mismatch vector from power injections
fn compute_mismatch_vec(
    system: &PowerSystem,
    power_injections: &[(f64, f64)],
    angle_vars: &[usize],
    vmag_vars: &[usize],
) -> Vec<f64> {
    let mut mismatch = Vec::new();

    // P mismatches
    for &bus_i in angle_vars {
        let (p_calc, _) = power_injections[bus_i];
        let p_sched = system.buses[bus_i].p_scheduled;
        mismatch.push(p_sched - p_calc);
    }

    // Q mismatches
    for &bus_i in vmag_vars {
        let (_, q_calc) = power_injections[bus_i];
        let q_sched = system.buses[bus_i].q_scheduled;
        mismatch.push(q_sched - q_calc);
    }

    mismatch
}
