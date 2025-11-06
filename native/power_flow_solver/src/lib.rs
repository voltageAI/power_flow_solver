use rustler::{NifResult, ResourceArc, Encoder, Env, Term};
use num_complex::Complex64;
use sprs::CsMat;
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;
use faer::complex_native::c64;
use faer::prelude::SpSolver;
use std::sync::Arc;
use std::collections::HashMap;

mod power_flow;

mod atoms {
    rustler::atoms! {
        ok,
        error,
        singular_matrix,
        dimension_mismatch,
        invalid_format,
    }
}

/// Complex number as (real, imaginary) tuple
type ComplexTuple = (f64, f64);

/// Convert Elixir complex tuple to Rust Complex64
fn tuple_to_complex(tuple: ComplexTuple) -> Complex64 {
    Complex64::new(tuple.0, tuple.1)
}

/// Convert Rust Complex64 to Elixir tuple
fn complex_to_tuple(c: Complex64) -> ComplexTuple {
    (c.re, c.im)
}

/// Resource to hold symbolic LU factorization
struct SymbolicLuResource {
    symbolic: Arc<faer::sparse::linalg::solvers::SymbolicLu<usize>>,
    n: usize,
}

/// Resource to hold complete LU factorization (symbolic + numeric)
struct LuFactorResource {
    lu: Arc<faer::sparse::linalg::solvers::Lu<usize, c64>>,
    n: usize,
}

/// Resource to hold voltage array across multiple Rust calls
/// This eliminates repeated serialization between Elixir and Rust
struct VoltageResource {
    voltage: Vec<(f64, f64)>,  // (magnitude, angle) pairs
    n: usize,
}

/// Solve a sparse linear system Ax = b using LU factorization
///
/// # Arguments
/// * `row_ptrs` - CSR row pointer array
/// * `col_indices` - CSR column index array
/// * `values` - Non-zero values as complex tuples
/// * `rhs` - Right-hand side vector as complex tuples
///
/// # Returns
/// * `{:ok, solution}` - Solution vector as complex tuples
/// * `{:error, reason}` - Error atom
#[rustler::nif]
fn solve_csr(
    row_ptrs: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<ComplexTuple>,
    rhs: Vec<ComplexTuple>,
) -> NifResult<(rustler::Atom, Vec<ComplexTuple>)> {
    // Validate dimensions
    if values.len() != col_indices.len() {
        return Ok((atoms::error(), vec![]));
    }

    let n = row_ptrs.len() - 1;
    if rhs.len() != n {
        return Ok((atoms::error(), vec![]));
    }

    // Convert values to Complex64
    let complex_values: Vec<Complex64> = values.iter()
        .map(|&t| tuple_to_complex(t))
        .collect();

    // Build CSR matrix
    let mat = CsMat::new(
        (n, n),
        row_ptrs.clone(),
        col_indices.clone(),
        complex_values,
    );

    // Convert RHS to Complex64
    let b: Vec<Complex64> = rhs.iter()
        .map(|&t| tuple_to_complex(t))
        .collect();

    // Solve using appropriate method
    match solve_sparse_system(&mat, &b) {
        Ok(solution) => {
            let result: Vec<ComplexTuple> = solution.iter()
                .map(|&c| complex_to_tuple(c))
                .collect();
            Ok((atoms::ok(), result))
        }
        Err(_) => Ok((atoms::error(), vec![])),
    }
}

/// Create symbolic LU factorization that can be reused
///
/// This performs the expensive symbolic analysis of the matrix pattern once,
/// which can then be reused for multiple numeric factorizations when only
/// the values change but the pattern remains the same.
#[rustler::nif]
fn create_symbolic_lu(
    env: rustler::Env,
    row_ptrs: Vec<usize>,
    col_indices: Vec<usize>,
) -> NifResult<(rustler::Atom, rustler::Term)> {
    let n = row_ptrs.len() - 1;

    // Create dummy values for pattern analysis
    let dummy_values: Vec<c64> = vec![c64::new(1.0, 0.0); col_indices.len()];

    // Build triplets for faer
    let mut triplets = Vec::new();
    for row in 0..n {
        let row_start = row_ptrs[row];
        let row_end = row_ptrs[row + 1];

        for idx in row_start..row_end {
            let col = col_indices[idx];
            let val = dummy_values[idx];
            triplets.push((row, col, val));
        }
    }

    // Build faer sparse matrix
    let sparse_mat = match faer::sparse::SparseColMat::try_new_from_triplets(
        n,
        n,
        &triplets,
    ) {
        Ok(mat) => mat,
        Err(_) => return Ok((atoms::error(), atoms::error().encode(env))),
    };

    // Perform symbolic factorization
    use faer::sparse::linalg::solvers::SymbolicLu;

    let symbolic = match SymbolicLu::try_new(sparse_mat.symbolic()) {
        Ok(sym) => sym,
        Err(_) => return Ok((atoms::error(), atoms::error().encode(env))),
    };

    // Store in resource
    let resource = ResourceArc::new(SymbolicLuResource {
        symbolic: Arc::new(symbolic),
        n,
    });

    Ok((atoms::ok(), resource.encode(env)))
}

/// Perform numeric LU factorization using pre-computed symbolic factorization
///
/// This is much faster than full factorization when the matrix pattern is unchanged.
#[rustler::nif]
fn factorize_with_symbolic(
    env: rustler::Env,
    symbolic_resource: ResourceArc<SymbolicLuResource>,
    row_ptrs: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<ComplexTuple>,
) -> NifResult<(rustler::Atom, rustler::Term)> {
    let n = row_ptrs.len() - 1;

    // Validate dimensions
    if n != symbolic_resource.n {
        return Ok((atoms::dimension_mismatch(), atoms::error().encode(env)));
    }

    if values.len() != col_indices.len() {
        return Ok((atoms::error(), atoms::error().encode(env)));
    }

    // Convert values to faer format and build triplets
    use rayon::prelude::*;

    let triplets: Vec<_> = (0..n)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_triplets = Vec::new();
            let row_start = row_ptrs[row];
            let row_end = row_ptrs[row + 1];

            for idx in row_start..row_end {
                let col = col_indices[idx];
                let val = tuple_to_complex(values[idx]);

                if val.re.is_finite() && val.im.is_finite() {
                    row_triplets.push((row, col, c64::new(val.re, val.im)));
                }
            }
            row_triplets
        })
        .collect();

    // Build faer sparse matrix
    let sparse_mat = match faer::sparse::SparseColMat::try_new_from_triplets(
        n,
        n,
        &triplets,
    ) {
        Ok(mat) => mat,
        Err(_) => return Ok((atoms::error(), atoms::error().encode(env))),
    };

    // Perform numeric factorization using symbolic
    use faer::sparse::linalg::solvers::Lu;

    // Clone the Arc to get the symbolic factorization
    let symbolic = symbolic_resource.symbolic.clone();

    let lu = match Lu::try_new_with_symbolic(
        (*symbolic).clone(),
        sparse_mat.as_ref()
    ) {
        Ok(lu) => lu,
        Err(_) => return Ok((atoms::singular_matrix(), atoms::error().encode(env))),
    };

    // Store in resource
    let resource = ResourceArc::new(LuFactorResource {
        lu: Arc::new(lu),
        n,
    });

    Ok((atoms::ok(), resource.encode(env)))
}

/// Solve using pre-computed LU factorization
///
/// Very fast when the factorization is already computed.
#[rustler::nif]
fn solve_with_lu(
    lu_resource: ResourceArc<LuFactorResource>,
    rhs: Vec<ComplexTuple>,
) -> NifResult<(rustler::Atom, Vec<ComplexTuple>)> {
    let n = lu_resource.n;

    if rhs.len() != n {
        return Ok((atoms::dimension_mismatch(), vec![]));
    }

    // Convert RHS to faer format
    let mut b_faer = faer::Mat::zeros(n, 1);
    for (i, &val) in rhs.iter().enumerate() {
        let complex_val = tuple_to_complex(val);
        if !complex_val.re.is_finite() || !complex_val.im.is_finite() {
            return Ok((atoms::error(), vec![]));
        }
        b_faer.write(i, 0, c64::new(complex_val.re, complex_val.im));
    }

    // Solve using pre-computed LU
    let x_faer = lu_resource.lu.solve(b_faer.as_ref());

    // Convert solution back
    use rayon::prelude::*;
    let solution: Vec<ComplexTuple> = (0..n)
        .into_par_iter()
        .map(|i| {
            let val = x_faer.read(i, 0);
            (val.re, val.im)
        })
        .collect();

    Ok((atoms::ok(), solution))
}

/// Solve multiple RHS vectors using pre-computed LU factorization
///
/// Solves AX = B where B has multiple columns.
#[rustler::nif]
fn solve_multiple_with_lu(
    lu_resource: ResourceArc<LuFactorResource>,
    rhs_list: Vec<Vec<ComplexTuple>>,
) -> NifResult<(rustler::Atom, Vec<Vec<ComplexTuple>>)> {
    let n = lu_resource.n;
    let num_rhs = rhs_list.len();

    if num_rhs == 0 {
        return Ok((atoms::ok(), vec![]));
    }

    // Validate all RHS vectors have correct dimension
    for rhs in &rhs_list {
        if rhs.len() != n {
            return Ok((atoms::dimension_mismatch(), vec![]));
        }
    }

    // Convert all RHS vectors to faer matrix
    let mut b_faer = faer::Mat::zeros(n, num_rhs);
    for (col, rhs) in rhs_list.iter().enumerate() {
        for (row, &val) in rhs.iter().enumerate() {
            let complex_val = tuple_to_complex(val);
            if !complex_val.re.is_finite() || !complex_val.im.is_finite() {
                return Ok((atoms::error(), vec![]));
            }
            b_faer.write(row, col, c64::new(complex_val.re, complex_val.im));
        }
    }

    // Solve all systems at once
    let x_faer = lu_resource.lu.solve(b_faer.as_ref());

    // Convert solutions back
    use rayon::prelude::*;
    let solutions: Vec<Vec<ComplexTuple>> = (0..num_rhs)
        .into_par_iter()
        .map(|col| {
            (0..n).map(|row| {
                let val = x_faer.read(row, col);
                (val.re, val.im)
            }).collect()
        })
        .collect();

    Ok((atoms::ok(), solutions))
}

/// Sparse matrix-vector multiplication: y = A * x
#[rustler::nif]
fn sparse_mv(
    row_ptrs: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<ComplexTuple>,
    x: Vec<ComplexTuple>,
) -> NifResult<(rustler::Atom, Vec<ComplexTuple>)> {
    // Validate dimensions
    if values.len() != col_indices.len() {
        return Ok((atoms::error(), vec![]));
    }

    let n = row_ptrs.len() - 1;
    if x.len() != n {
        return Ok((atoms::error(), vec![]));
    }

    // Convert to Complex64
    let complex_values: Vec<Complex64> = values.iter()
        .map(|&t| tuple_to_complex(t))
        .collect();

    let x_vec: Vec<Complex64> = x.iter()
        .map(|&t| tuple_to_complex(t))
        .collect();

    // Perform sparse matrix-vector multiplication manually
    // y[i] = sum_j A[i,j] * x[j]
    let mut result_vec = vec![Complex64::new(0.0, 0.0); n];

    for row in 0..n {
        let row_start = row_ptrs[row];
        let row_end = row_ptrs[row + 1];

        for idx in row_start..row_end {
            let col = col_indices[idx];
            let val = complex_values[idx];
            result_vec[row] += val * x_vec[col];
        }
    }

    let result: Vec<ComplexTuple> = result_vec.iter()
        .map(|&c| complex_to_tuple(c))
        .collect();

    Ok((atoms::ok(), result))
}

/// Legacy LU factorization placeholder (deprecated)
#[rustler::nif]
fn lu_factorize(
    _row_ptrs: Vec<usize>,
    _col_indices: Vec<usize>,
    _values: Vec<ComplexTuple>,
) -> NifResult<rustler::Atom> {
    // Deprecated - use create_symbolic_lu and factorize_with_symbolic instead
    Ok(atoms::error())
}

/// Solve using dense LU decomposition (for small systems)
fn solve_dense(mat: &CsMat<Complex64>, b: &[Complex64]) -> Result<Vec<Complex64>, String> {
    let n = mat.rows();

    // Convert sparse to dense
    let mut dense = Array2::<Complex64>::zeros((n, n));

    for row in 0..n {
        let row_view = mat.outer_view(row);
        if let Some(view) = row_view {
            for (col, &val) in view.iter() {
                // Check for NaN or Inf
                if !val.re.is_finite() || !val.im.is_finite() {
                    return Err("Matrix contains NaN or Inf values".to_string());
                }
                dense[[row, col]] = val;
            }
        }
    }

    // Convert RHS to Array1
    let b_array = Array1::from_vec(b.to_vec());

    // Check RHS
    for val in b {
        if !val.re.is_finite() || !val.im.is_finite() {
            return Err("RHS contains NaN or Inf values".to_string());
        }
    }

    // Solve using LU decomposition
    match dense.solve(&b_array) {
        Ok(solution) => Ok(solution.to_vec()),
        Err(_) => Err("Matrix is singular or near-singular".to_string()),
    }
}

/// Solve using faer's sparse LU decomposition with parallel support
/// When rayon feature is enabled, faer automatically uses parallel algorithms
fn solve_sparse_faer(mat: &CsMat<Complex64>, b: &[Complex64]) -> Result<Vec<Complex64>, String> {
    let n = mat.rows();

    // Convert sprs CSR matrix to faer CSC format using parallel iteration
    // Build triplets for faer as (row, col, value) tuples
    use rayon::prelude::*;

    let triplets: Vec<_> = (0..n)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_triplets = Vec::new();
            if let Some(view) = mat.outer_view(row) {
                for (col, &val) in view.iter() {
                    // Check for NaN or Inf
                    if val.re.is_finite() && val.im.is_finite() {
                        // Convert num_complex::Complex64 to faer::c64
                        row_triplets.push((row, col, c64::new(val.re, val.im)));
                    }
                }
            }
            row_triplets
        })
        .collect();

    // Validate triplets
    if triplets.iter().any(|(_, _, val)| !val.re.is_finite() || !val.im.is_finite()) {
        return Err("Matrix contains NaN or Inf values".to_string());
    }

    // Build faer sparse matrix from triplets
    let sparse_mat = match faer::sparse::SparseColMat::try_new_from_triplets(
        n,
        n,
        &triplets,
    ) {
        Ok(mat) => mat,
        Err(_) => return Err("Failed to create sparse matrix".to_string()),
    };

    // Convert RHS to faer format
    let mut b_faer = faer::Mat::zeros(n, 1);
    for (i, &val) in b.iter().enumerate() {
        if !val.re.is_finite() || !val.im.is_finite() {
            return Err("RHS contains NaN or Inf values".to_string());
        }
        b_faer.write(i, 0, c64::new(val.re, val.im));
    }

    // Perform symbolic factorization
    // With rayon feature enabled, faer uses parallel algorithms internally
    // NOTE: faer's SymbolicLu::try_new automatically applies AMD ordering internally
    // for better sparsity preservation and reduced fill-in
    use faer::sparse::linalg::solvers::{SymbolicLu, Lu};

    let symbolic = match SymbolicLu::try_new(sparse_mat.symbolic()) {
        Ok(sym) => sym,
        Err(_) => return Err("Symbolic factorization failed".to_string()),
    };

    // Perform numeric LU factorization
    // Parallel execution happens automatically with rayon feature
    let lu = match Lu::try_new_with_symbolic(symbolic, sparse_mat.as_ref()) {
        Ok(lu) => lu,
        Err(_) => return Err("Matrix is singular or near-singular".to_string()),
    };

    // Solve the system
    // The solve operation also benefits from parallel triangular solves
    let x_faer = lu.solve(b_faer.as_ref());

    // Convert solution back to Vec<Complex64> using parallel iteration
    let solution: Vec<Complex64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let val = x_faer.read(i, 0);
            Complex64::new(val.re, val.im)
        })
        .collect();

    Ok(solution)
}

/// Helper function to solve sparse linear system
/// Now uses faer's sparse LU solver (similar performance to UMFPACK/SuperLU)
fn solve_sparse_system(
    mat: &CsMat<Complex64>,
    b: &[Complex64],
) -> Result<Vec<Complex64>, String> {
    let n = mat.rows();

    // Check for empty system
    if n == 0 {
        return Ok(vec![]);
    }

    // For very small systems (<= 10x10), dense LU is actually faster
    // For larger systems, use faer's sparse LU (SuperLU-like performance)
    if n <= 10 {
        return solve_dense(mat, b);
    }

    // Use faer's sparse solver for all other sizes
    solve_sparse_faer(mat, b)
}

/// Build Jacobian matrix for power flow in parallel using Rust
///
/// This function builds the Jacobian matrix for Newton-Raphson power flow.
/// It matches the Elixir implementation exactly, using pre-computed power injections.
///
/// # Arguments
/// * `y_bus_data` - Y-bus matrix in CSR format (row_ptrs, col_indices, values)
/// * `voltage` - Voltage vector as (magnitude, angle) tuples
/// * `bus_types` - Bus type for each bus (0=slack, 1=pv, 2=pq)
/// * `angle_vars` - Indices of angle variables
/// * `vmag_vars` - Indices of voltage magnitude variables
///
/// # Returns
/// * `{:ok, (row_ptrs, col_indices, values)}` - Jacobian in CSR format
/// * `{:error, reason}` - Error description
#[rustler::nif]
fn build_jacobian_rust(
    // Y-bus in CSR format
    y_row_ptrs: Vec<usize>,
    y_col_indices: Vec<usize>,
    y_values: Vec<ComplexTuple>,

    // Voltage vector
    voltage: Vec<ComplexTuple>,

    // Bus information
    bus_types: Vec<u8>,  // 0=slack, 1=pv, 2=pq

    // Variable indices
    angle_vars: Vec<usize>,
    vmag_vars: Vec<usize>,
) -> NifResult<(rustler::Atom, Vec<usize>, Vec<usize>, Vec<ComplexTuple>)> {
    use rayon::prelude::*;

    let num_vars = angle_vars.len() + vmag_vars.len();
    let num_p_eqs = angle_vars.len();

    // Convert voltage to (magnitude, angle) for calculations
    let voltage_data: Vec<(f64, f64)> = voltage.iter()
        .map(|&(mag, ang)| (mag, ang))
        .collect();

    // Convert Y-bus values
    let y_vals: Vec<(f64, f64)> = y_values.iter()
        .map(|&(re, im)| (re, im))
        .collect();

    // Pre-compute power injections (P_i, Q_i) for all buses
    // This matches the Elixir precompute_power_injections function
    let power_injections: Vec<(f64, f64)> = (0..bus_types.len())
        .map(|bus_idx| {
            compute_power_injection(
                bus_idx,
                &y_row_ptrs,
                &y_col_indices,
                &y_vals,
                &voltage_data,
            )
        })
        .collect();

    // Build a Y-bus lookup map for fast access to Y_ik elements
    let y_bus_map = build_ybus_map(&y_row_ptrs, &y_col_indices, &y_vals, bus_types.len());

    // Build Jacobian rows in parallel
    let all_rows: Vec<Vec<(usize, usize, f64)>> = (0..num_vars)
        .into_par_iter()
        .map(|eq_idx| {
            build_jacobian_row_correct(
                eq_idx,
                &voltage_data,
                &power_injections,
                &y_bus_map,
                &angle_vars,
                &vmag_vars,
                num_p_eqs,
            )
        })
        .collect();

    // Combine rows into CSR format
    let mut row_ptrs = vec![0];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for mut row_triplets in all_rows {
        // Sort this row's elements by column index (required for CSR format)
        row_triplets.sort_by_key(|(_row, col, _val)| *col);

        for (_row, col, val) in row_triplets {
            col_indices.push(col);
            values.push((val, 0.0)); // Jacobian values are real
        }
        row_ptrs.push(col_indices.len());
    }

    Ok((atoms::ok(), row_ptrs, col_indices, values))
}

/// Compute power injection (P_i, Q_i) for a single bus
/// This matches the Elixir calculate_power_injection function
fn compute_power_injection(
    bus_idx: usize,
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[(f64, f64)],
    voltage: &[(f64, f64)],
) -> (f64, f64) {
    let (v_i_mag, v_i_angle) = voltage[bus_idx];
    let row_start = row_ptrs[bus_idx];
    let row_end = row_ptrs[bus_idx + 1];

    if row_start >= row_end {
        return (0.0, 0.0);
    }

    let mut p_sum = 0.0;
    let mut q_sum = 0.0;

    for idx in row_start..row_end {
        let col = col_indices[idx];
        let (y_real, y_imag) = values[idx];
        let (v_k_mag, v_k_angle) = voltage[col];

        // Y_ik in polar form
        let y_mag = (y_real * y_real + y_imag * y_imag).sqrt();
        let y_angle = y_imag.atan2(y_real);

        // Power injection contribution
        let angle_diff = y_angle - v_i_angle + v_k_angle;
        p_sum += v_i_mag * v_k_mag * y_mag * angle_diff.cos();
        q_sum += v_i_mag * v_k_mag * y_mag * angle_diff.sin();
    }

    (p_sum, q_sum)
}

/// Build a fast lookup map for Y-bus elements
/// Returns a map from (i, j) -> (G_ij, B_ij)
fn build_ybus_map(
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[(f64, f64)],
    num_buses: usize,
) -> std::collections::HashMap<(usize, usize), (f64, f64)> {
    let mut map = std::collections::HashMap::new();

    for i in 0..num_buses {
        let row_start = row_ptrs[i];
        let row_end = row_ptrs[i + 1];

        for idx in row_start..row_end {
            let j = col_indices[idx];
            let (g_ij, b_ij) = values[idx];
            map.insert((i, j), (g_ij, b_ij));
        }
    }

    map
}

/// Build a single row of the Jacobian matrix using correct formulas
fn build_jacobian_row_correct(
    eq_idx: usize,
    voltage: &[(f64, f64)],
    power_injections: &[(f64, f64)],
    y_bus_map: &std::collections::HashMap<(usize, usize), (f64, f64)>,
    angle_vars: &[usize],
    vmag_vars: &[usize],
    num_p_eqs: usize,
) -> Vec<(usize, usize, f64)> {
    let mut triplets = Vec::new();

    // Determine if this is a P equation or Q equation
    let is_p_equation = eq_idx < num_p_eqs;

    if is_p_equation {
        // P equation - get the bus index for this P equation
        let bus_i = angle_vars[eq_idx];
        let (v_i_mag, v_i_angle) = voltage[bus_i];
        let (p_i, q_i) = power_injections[bus_i];

        // Build ∂P/∂θ elements for all angle variables
        for (var_offset, &bus_k) in angle_vars.iter().enumerate() {
            let value = if bus_i == bus_k {
                // Diagonal: ∂mismatch/∂θ_i = Q_i + V_i² * B_ii
                let (_g_ii, b_ii) = y_bus_map.get(&(bus_i, bus_i)).unwrap_or(&(0.0, 0.0));
                q_i + v_i_mag * v_i_mag * b_ii
            } else {
                // Off-diagonal: ∂mismatch/∂θ_k = -V_i * V_k * (G_ik * sin(θ_ik) - B_ik * cos(θ_ik))
                if let Some(&(g_ik, b_ik)) = y_bus_map.get(&(bus_i, bus_k)) {
                    let (v_k_mag, v_k_angle) = voltage[bus_k];
                    let theta_ik = v_i_angle - v_k_angle;
                    -v_i_mag * v_k_mag * (g_ik * theta_ik.sin() - b_ik * theta_ik.cos())
                } else {
                    0.0 // No connection
                }
            };

            if value.abs() > 1e-15 {
                triplets.push((eq_idx, var_offset, value));
            }
        }

        // Build ∂P/∂V elements for all voltage magnitude variables
        for (var_offset, &bus_k) in vmag_vars.iter().enumerate() {
            let value = if bus_i == bus_k {
                // Diagonal: ∂mismatch/∂V_i = -(P_i/V_i + V_i * G_ii)
                let (g_ii, _b_ii) = y_bus_map.get(&(bus_i, bus_i)).unwrap_or(&(0.0, 0.0));
                -(p_i / v_i_mag + v_i_mag * g_ii)
            } else {
                // Off-diagonal: ∂mismatch/∂V_k = -V_i * (G_ik * cos(θ_ik) + B_ik * sin(θ_ik))
                if let Some(&(g_ik, b_ik)) = y_bus_map.get(&(bus_i, bus_k)) {
                    let (_v_k_mag, v_k_angle) = voltage[bus_k];
                    let theta_ik = v_i_angle - v_k_angle;
                    -v_i_mag * (g_ik * theta_ik.cos() + b_ik * theta_ik.sin())
                } else {
                    0.0 // No connection
                }
            };

            if value.abs() > 1e-15 {
                let col_idx = num_p_eqs + var_offset;
                triplets.push((eq_idx, col_idx, value));
            }
        }
    } else {
        // Q equation - get the bus index for this Q equation
        let q_eq_idx = eq_idx - num_p_eqs;
        let bus_i = vmag_vars[q_eq_idx];
        let (v_i_mag, v_i_angle) = voltage[bus_i];
        let (p_i, q_i) = power_injections[bus_i];

        // Build ∂Q/∂θ elements for all angle variables
        for (var_offset, &bus_k) in angle_vars.iter().enumerate() {
            let value = if bus_i == bus_k {
                // Diagonal: ∂mismatch/∂θ_i = P_i - V_i² * G_ii
                let (g_ii, _b_ii) = y_bus_map.get(&(bus_i, bus_i)).unwrap_or(&(0.0, 0.0));
                p_i - v_i_mag * v_i_mag * g_ii
            } else {
                // Off-diagonal: ∂mismatch/∂θ_k = -V_i * V_k * (G_ik * cos(θ_ik) + B_ik * sin(θ_ik))
                if let Some(&(g_ik, b_ik)) = y_bus_map.get(&(bus_i, bus_k)) {
                    let (v_k_mag, v_k_angle) = voltage[bus_k];
                    let theta_ik = v_i_angle - v_k_angle;
                    -v_i_mag * v_k_mag * (g_ik * theta_ik.cos() + b_ik * theta_ik.sin())
                } else {
                    0.0 // No connection
                }
            };

            if value.abs() > 1e-15 {
                triplets.push((eq_idx, var_offset, value));
            }
        }

        // Build ∂Q/∂V elements for all voltage magnitude variables
        for (var_offset, &bus_k) in vmag_vars.iter().enumerate() {
            let value = if bus_i == bus_k {
                // Diagonal: ∂mismatch/∂V_i = Q_i/V_i - V_i * B_ii
                let (_g_ii, b_ii) = y_bus_map.get(&(bus_i, bus_i)).unwrap_or(&(0.0, 0.0));
                q_i / v_i_mag - v_i_mag * b_ii
            } else {
                // Off-diagonal: ∂mismatch/∂V_k = V_i * (G_ik * sin(θ_ik) - B_ik * cos(θ_ik))
                if let Some(&(g_ik, b_ik)) = y_bus_map.get(&(bus_i, bus_k)) {
                    let (_v_k_mag, v_k_angle) = voltage[bus_k];
                    let theta_ik = v_i_angle - v_k_angle;
                    v_i_mag * (g_ik * theta_ik.sin() - b_ik * theta_ik.cos())
                } else {
                    0.0 // No connection
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

/// Calculate reactive power injection at a specific bus
///
/// Q_i = Im(V_i * conj(I_i)) = Σ_k |V_i||V_k||Y_ik| sin(θ_ik - δ_i + δ_k)
///
/// # Arguments
/// * `row_ptrs` - CSR row pointer array for Y-bus
/// * `col_indices` - CSR column index array for Y-bus
/// * `values` - Y-bus admittance values as complex tuples (real, imag)
/// * `voltage` - Voltage vector as magnitude-angle tuples (mag, angle)
/// * `bus_idx` - Index of the bus to calculate Q for
///
/// # Returns
/// * `{:ok, q_injection}` - Reactive power injection value
/// * `{:error, reason}` - Error if calculation fails
#[rustler::nif]
fn calculate_q_injection_rust(
    row_ptrs: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<ComplexTuple>,
    voltage: Vec<(f64, f64)>,  // (magnitude, angle) tuples
    bus_idx: usize,
) -> NifResult<(rustler::Atom, f64)> {
    // Validate bus index
    let n = row_ptrs.len() - 1;
    if bus_idx >= n {
        return Ok((atoms::error(), 0.0));
    }

    // Validate dimensions
    if values.len() != col_indices.len() || voltage.len() != n {
        return Ok((atoms::error(), 0.0));
    }

    // Get voltage at this bus
    let (v_i_mag, v_i_angle) = voltage[bus_idx];

    // Get row start and end for this bus in CSR format
    let row_start = row_ptrs[bus_idx];
    let row_end = row_ptrs[bus_idx + 1];

    // Calculate Q injection by summing contributions from all connected buses
    let mut q_sum = 0.0;

    for idx in row_start..row_end {
        let col = col_indices[idx];
        let (y_real, y_imag) = values[idx];
        let (v_k_mag, v_k_angle) = voltage[col];

        // Convert Y_ik to polar form
        let y_mag = (y_real * y_real + y_imag * y_imag).sqrt();
        let y_angle = y_imag.atan2(y_real);

        // Calculate Q contribution:
        // Q_contribution = |V_i| * |V_k| * |Y_ik| * sin(θ_ik - δ_i + δ_k)
        let angle_diff = y_angle - v_i_angle + v_k_angle;
        let q_contrib = v_i_mag * v_k_mag * y_mag * angle_diff.sin();

        q_sum += q_contrib;
    }

    Ok((atoms::ok(), q_sum))
}

/// Calculate reactive power injection for multiple buses at once
///
/// This is more efficient than calling calculate_q_injection_rust multiple times
/// because it avoids repeated boundary crossings.
///
/// # Arguments
/// * `row_ptrs` - CSR row pointer array for Y-bus
/// * `col_indices` - CSR column index array for Y-bus
/// * `values` - Y-bus admittance values as complex tuples
/// * `voltage` - Voltage vector as magnitude-angle tuples
/// * `bus_indices` - List of bus indices to calculate Q for
///
/// # Returns
/// * `{:ok, q_values}` - Map of bus_idx => q_injection
/// * `{:error, reason}` - Error if calculation fails
#[rustler::nif]
fn calculate_q_injection_batch_rust(
    row_ptrs: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<ComplexTuple>,
    voltage: Vec<(f64, f64)>,
    bus_indices: Vec<usize>,
) -> NifResult<(rustler::Atom, Vec<(usize, f64)>)> {
    // Validate dimensions
    let n = row_ptrs.len() - 1;
    if values.len() != col_indices.len() || voltage.len() != n {
        return Ok((atoms::error(), vec![]));
    }

    let mut results = Vec::with_capacity(bus_indices.len());

    for &bus_idx in &bus_indices {
        // Skip invalid indices
        if bus_idx >= n {
            continue;
        }

        // Get voltage at this bus
        let (v_i_mag, v_i_angle) = voltage[bus_idx];

        // Get row bounds
        let row_start = row_ptrs[bus_idx];
        let row_end = row_ptrs[bus_idx + 1];

        // Calculate Q injection
        let mut q_sum = 0.0;

        for idx in row_start..row_end {
            let col = col_indices[idx];
            let (y_real, y_imag) = values[idx];
            let (v_k_mag, v_k_angle) = voltage[col];

            // Convert Y_ik to polar form
            let y_mag = (y_real * y_real + y_imag * y_imag).sqrt();
            let y_angle = y_imag.atan2(y_real);

            // Calculate Q contribution
            let angle_diff = y_angle - v_i_angle + v_k_angle;
            let q_contrib = v_i_mag * v_k_mag * y_mag * angle_diff.sin();

            q_sum += q_contrib;
        }

        results.push((bus_idx, q_sum));
    }

    Ok((atoms::ok(), results))
}

/// Create a voltage resource from voltage array
///
/// This keeps voltage in Rust memory across multiple calls,
/// eliminating repeated serialization overhead.
///
/// # Arguments
/// * `voltage` - Voltage vector as (magnitude, angle) tuples
///
/// # Returns
/// * `{:ok, resource}` - Voltage resource handle
/// * `{:error, reason}` - Error if creation fails
#[rustler::nif]
fn create_voltage_resource(
    env: rustler::Env,
    voltage: Vec<(f64, f64)>,
) -> NifResult<(rustler::Atom, ResourceArc<VoltageResource>)> {
    let n = voltage.len();
    let resource = ResourceArc::new(VoltageResource { voltage, n });
    Ok((atoms::ok(), resource))
}

/// Calculate Q injection for multiple buses using voltage resource
///
/// This is more efficient than passing voltage each time because:
/// - Voltage stays in Rust memory (no serialization)
/// - Direct array access (cache-friendly)
/// - Can be called multiple times with same voltage
///
/// # Arguments
/// * `row_ptrs` - CSR row pointers for Y-bus
/// * `col_indices` - CSR column indices for Y-bus
/// * `values` - Y-bus admittance values as complex tuples
/// * `voltage_res` - Voltage resource (stays in Rust)
/// * `bus_indices` - List of bus indices to calculate Q for
///
/// # Returns
/// * `{:ok, results}` - List of (bus_idx, q_injection) tuples
/// * `{:error, reason}` - Error if calculation fails
#[rustler::nif]
fn calculate_q_batch_from_resource(
    row_ptrs: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<ComplexTuple>,
    voltage_res: ResourceArc<VoltageResource>,
    bus_indices: Vec<usize>,
) -> NifResult<(rustler::Atom, Vec<(usize, f64)>)> {
    // Validate dimensions
    let n = row_ptrs.len() - 1;
    if values.len() != col_indices.len() || voltage_res.n != n {
        return Ok((atoms::error(), vec![]));
    }

    // Use voltage from resource (no serialization!)
    let voltage = &voltage_res.voltage;
    let mut results = Vec::with_capacity(bus_indices.len());

    for &bus_idx in &bus_indices {
        // Skip invalid indices
        if bus_idx >= n {
            continue;
        }

        // Get voltage at this bus
        let (v_i_mag, v_i_angle) = voltage[bus_idx];

        // Get row bounds
        let row_start = row_ptrs[bus_idx];
        let row_end = row_ptrs[bus_idx + 1];

        // Calculate Q injection
        let mut q_sum = 0.0;

        for idx in row_start..row_end {
            let col = col_indices[idx];
            let (y_real, y_imag) = values[idx];
            let (v_k_mag, v_k_angle) = voltage[col];

            // Convert Y_ik to polar form
            let y_mag = (y_real * y_real + y_imag * y_imag).sqrt();
            let y_angle = y_imag.atan2(y_real);

            // Calculate Q contribution
            let angle_diff = y_angle - v_i_angle + v_k_angle;
            let q_contrib = v_i_mag * v_k_mag * y_mag * angle_diff.sin();

            q_sum += q_contrib;
        }

        results.push((bus_idx, q_sum));
    }

    Ok((atoms::ok(), results))
}

/// Solve power flow using complete Rust Newton-Raphson solver
///
/// This NIF implements the entire iteration loop in Rust, eliminating
/// boundary crossings with Elixir.
///
/// # Arguments
/// * `buses` - List of bus data: [(type, p_sched, q_sched, v_sched), ...]
/// * `y_bus_data` - Y-bus matrix: (row_ptrs, col_indices, values)
/// * `initial_voltage` - Initial voltage: [(mag, ang), ...]
/// * `max_iterations` - Maximum iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
/// * `{:ok, {voltage, iterations, converged, final_mismatch}}`
/// * `{:error, reason}`
#[rustler::nif]
fn solve_power_flow_rust(
    buses: Vec<(u8, f64, f64, f64, Option<f64>, Option<f64>, f64)>,  // (type, p_sched, q_sched, v_sched, q_min, q_max, q_load)
    y_bus_data: (Vec<usize>, Vec<usize>, Vec<ComplexTuple>),
    initial_voltage: Vec<(f64, f64)>,
    max_iterations: usize,
    tolerance: f64,
    enforce_q_limits: bool,
    q_tolerance: f64,
) -> NifResult<(rustler::Atom, Vec<(f64, f64)>, usize, bool, f64)> {

    eprintln!("\n=== NIF ENTRY: solve_power_flow_rust ===");
    eprintln!("Buses received: {}", buses.len());
    eprintln!("Q-limits enabled: {}", enforce_q_limits);
    eprintln!("First 3 buses:");
    for (i, (btype, p, q, v, q_min, q_max, q_load)) in buses.iter().take(3).enumerate() {
        eprintln!("  Bus {}: type={}, P={:.6}, Q={:.6}, V={:.6}, Q_min={:?}, Q_max={:?}, Q_load={:.6}",
            i, btype, p, q, v, q_min, q_max, q_load);
    }

    // Convert bus data
    let bus_data: Vec<power_flow::BusData> = buses
        .into_iter()
        .map(|(bus_type, p, q, v, q_min, q_max, q_load)| power_flow::BusData {
            bus_type: power_flow::BusType::from(bus_type),
            p_scheduled: p,
            q_scheduled: q,
            v_scheduled: v,
            q_min,
            q_max,
            original_type: None,
            q_load,
        })
        .collect();

    // Convert Y-bus data
    let (row_ptrs, col_indices, values_tuples) = y_bus_data;

    eprintln!("Y-bus CSR: {} row_ptrs, {} col_indices, {} values",
        row_ptrs.len(), col_indices.len(), values_tuples.len());
    eprintln!("Row ptrs (first 5): {:?}", &row_ptrs[..row_ptrs.len().min(5)]);
    eprintln!("First 3 Y-bus values:");
    for (i, (re, im)) in values_tuples.iter().take(3).enumerate() {
        eprintln!("  Y[{}] = {:.6} + j{:.6}", i, re, im);
    }

    let values: Vec<Complex64> = values_tuples
        .into_iter()
        .map(|(re, im)| Complex64::new(re, im))
        .collect();

    let y_bus = power_flow::YBusData {
        row_ptrs,
        col_indices,
        values,
    };

    let system = power_flow::PowerSystem {
        n_buses: bus_data.len(),
        y_bus,
        buses: bus_data,
    };

    let config = power_flow::SolverConfig {
        max_iterations,
        tolerance,
        enforce_q_limits,
        q_tolerance,
    };

    // Solve
    match power_flow::solve_power_flow(system, initial_voltage, config) {
        Ok(result) => Ok((
            atoms::ok(),
            result.voltage,
            result.iterations,
            result.converged,
            result.final_mismatch,
        )),
        Err(err) => Ok((
            atoms::error(),
            vec![],
            0,
            false,
            0.0,
        )),
    }
}


// Load the resources into the Rustler environment
fn load(env: rustler::Env, _: rustler::Term) -> bool {
    rustler::resource!(SymbolicLuResource, env);
    rustler::resource!(LuFactorResource, env);
    rustler::resource!(VoltageResource, env);
    true
}

/// Validate Jacobian using numerical differentiation
///
/// Compares analytical Jacobian against numerical derivatives computed
/// via finite differences to identify formula errors.
///
/// # Arguments
/// * `buses` - List of bus data
/// * `y_bus_data` - Y-bus matrix in CSR format
/// * `voltage` - Current voltage as (magnitude, angle) tuples
/// * `epsilon` - Finite difference step size (e.g., 1e-7)
///
/// # Returns
/// * `{:ok, {max_error, avg_error, num_large_errors, error_details}}`
/// * `{:error, reason}`
#[rustler::nif]
fn validate_jacobian_rust(
    buses: Vec<(u8, f64, f64, f64, Option<f64>, Option<f64>, f64)>,
    y_bus_data: (Vec<usize>, Vec<usize>, Vec<ComplexTuple>),
    voltage: Vec<(f64, f64)>,
    epsilon: f64,
) -> NifResult<(rustler::Atom, f64, f64, usize, Vec<String>)> {
    // Convert bus data
    let bus_data: Vec<power_flow::BusData> = buses
        .into_iter()
        .map(|(bus_type, p, q, v, q_min, q_max, q_load)| power_flow::BusData {
            bus_type: power_flow::BusType::from(bus_type),
            p_scheduled: p,
            q_scheduled: q,
            v_scheduled: v,
            q_min,
            q_max,
            original_type: None,
            q_load,
        })
        .collect();

    // Convert Y-bus data
    let (row_ptrs, col_indices, values_tuples) = y_bus_data;
    let values: Vec<Complex64> = values_tuples
        .into_iter()
        .map(|(re, im)| Complex64::new(re, im))
        .collect();

    let y_bus = power_flow::YBusData {
        row_ptrs,
        col_indices,
        values,
    };

    let system = power_flow::PowerSystem {
        n_buses: bus_data.len(),
        y_bus,
        buses: bus_data,
    };

    // Convert voltage to Complex64
    let voltage_complex: Vec<Complex64> = voltage
        .iter()
        .map(|&(mag, ang)| Complex64::new(mag * ang.cos(), mag * ang.sin()))
        .collect();

    // Determine variables
    let (angle_vars, vmag_vars) = power_flow::determine_variables(&system.buses);

    // Run validation
    match power_flow::validate_jacobian(&system, &voltage_complex, &angle_vars, &vmag_vars, epsilon) {
        Ok((max_error, avg_error, num_large_errors, error_details)) => {
            Ok((atoms::ok(), max_error, avg_error, num_large_errors, error_details))
        }
        Err(err) => Ok((atoms::error(), 0.0, 0.0, 0, vec![err])),
    }
}

rustler::init!(
    "Elixir.PowerFlowSolver.SparseLinearAlgebra",
    [
        solve_csr,
        sparse_mv,
        lu_factorize,
        create_symbolic_lu,
        factorize_with_symbolic,
        solve_with_lu,
        solve_multiple_with_lu,
        build_jacobian_rust,
        calculate_q_injection_rust,
        calculate_q_injection_batch_rust,
        create_voltage_resource,
        calculate_q_batch_from_resource,
        solve_power_flow_rust,
        validate_jacobian_rust
    ],
    load = load
);

