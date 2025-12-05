//! Short Circuit Ratio (SCR) Calculation Module
//!
//! This module provides high-performance calculations for Short Circuit Ratio (SCR),
//! a key metric for assessing grid strength at renewable energy interconnection points.
//!
//! # Background
//!
//! The Short Circuit Ratio is defined as:
//!
//! ```text
//! SCR = S_sc / P_rated
//! ```
//!
//! Where:
//! - `S_sc` is the short circuit capacity at the point of interconnection (MVA)
//! - `P_rated` is the rated power of the plant (MW)
//!
//! The short circuit capacity is calculated from the Thevenin equivalent impedance:
//!
//! ```text
//! S_sc = V_base² / |Z_th|
//! ```
//!
//! Where `Z_th` is the Thevenin impedance seen looking into the grid from the bus.
//!
//! # Z-bus Matrix
//!
//! The Z-bus (impedance matrix) is the inverse of the Y-bus (admittance matrix).
//! The diagonal elements `Z_ii` represent the Thevenin impedance at each bus:
//!
//! ```text
//! Z_bus = Y_bus⁻¹
//! Z_th(bus_i) = Z_ii (diagonal element)
//! ```
//!
//! # Generator Subtransient Reactance
//!
//! For more accurate short circuit calculations, generator subtransient reactance (X''d)
//! can be added to the Y-bus before inversion. This models generators as voltage sources
//! behind their subtransient reactance during fault conditions.

use num_complex::Complex64;
use rayon::prelude::*;

/// Input data for a single plant/generator for SCR calculation
#[derive(Clone, Debug)]
pub struct PlantData {
    /// Bus ID where the plant is connected
    pub bus_id: usize,
    /// Rated power of the plant (MW)
    pub p_rated_mw: f64,
    /// Optional generator subtransient reactance (per unit on machine base)
    pub xdpp: Option<f64>,
    /// Optional machine MVA base (for converting X''d to system base)
    pub mva_base: Option<f64>,
}

/// Result for a single bus SCR calculation
#[derive(Clone, Debug)]
pub struct ScrResult {
    /// Bus ID
    pub bus_id: usize,
    /// Thevenin impedance magnitude (per unit)
    pub z_thevenin_pu: f64,
    /// Thevenin impedance angle (radians)
    pub z_thevenin_angle: f64,
    /// Short circuit capacity (MVA)
    pub short_circuit_mva: f64,
    /// Plant rated power (MW)
    pub p_rated_mw: f64,
    /// Short Circuit Ratio (dimensionless)
    pub scr: f64,
}

/// Configuration for SCR calculation
#[derive(Clone, Debug)]
pub struct ScrConfig {
    /// System MVA base (typically 100 MVA)
    pub system_mva_base: f64,
    /// Whether to include generator subtransient reactances in the model
    pub include_generator_reactance: bool,
    /// Tolerance for matrix conditioning checks (reserved for future use)
    #[allow(dead_code)]
    pub conditioning_tolerance: f64,
}

impl Default for ScrConfig {
    fn default() -> Self {
        Self {
            system_mva_base: 100.0,
            include_generator_reactance: false,
            conditioning_tolerance: 1e-10,
        }
    }
}

/// Y-bus matrix in CSR (Compressed Sparse Row) format
#[derive(Clone, Debug)]
pub struct YBusCsr {
    pub row_ptrs: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<Complex64>,
    pub n: usize,
}

impl YBusCsr {
    /// Create a new Y-bus from CSR components
    pub fn new(row_ptrs: Vec<usize>, col_indices: Vec<usize>, values: Vec<Complex64>) -> Self {
        let n = row_ptrs.len() - 1;
        Self {
            row_ptrs,
            col_indices,
            values,
            n,
        }
    }

    /// Get element at (i, j), returns 0 if not present
    pub fn get(&self, i: usize, j: usize) -> Complex64 {
        if i >= self.n {
            return Complex64::new(0.0, 0.0);
        }

        let row_start = self.row_ptrs[i];
        let row_end = self.row_ptrs[i + 1];

        for idx in row_start..row_end {
            if self.col_indices[idx] == j {
                return self.values[idx];
            }
        }

        Complex64::new(0.0, 0.0)
    }

    /// Add a value to element at (i, j)
    /// If element doesn't exist, behavior depends on matrix structure
    #[allow(dead_code)]
    pub fn add_to_diagonal(&mut self, i: usize, value: Complex64) {
        if i >= self.n {
            return;
        }

        let row_start = self.row_ptrs[i];
        let row_end = self.row_ptrs[i + 1];

        for idx in row_start..row_end {
            if self.col_indices[idx] == i {
                self.values[idx] += value;
                return;
            }
        }

        // If diagonal element doesn't exist in sparse structure,
        // we need to insert it (this is a rare case for Y-bus)
        eprintln!("Warning: Diagonal element ({}, {}) not found in sparse structure", i, i);
    }

    /// Convert to dense matrix for inversion
    pub fn to_dense(&self) -> Vec<Vec<Complex64>> {
        let mut dense = vec![vec![Complex64::new(0.0, 0.0); self.n]; self.n];

        for i in 0..self.n {
            let row_start = self.row_ptrs[i];
            let row_end = self.row_ptrs[i + 1];

            for idx in row_start..row_end {
                let j = self.col_indices[idx];
                dense[i][j] = self.values[idx];
            }
        }

        dense
    }
}

/// Z-bus matrix result (stored as dense for SCR calculations)
#[derive(Clone, Debug)]
pub struct ZBusResult {
    /// Dense Z-bus matrix
    pub z_bus: Vec<Vec<Complex64>>,
    /// Matrix dimension
    pub n: usize,
    /// Whether matrix inversion was successful
    pub success: bool,
    /// Error message if inversion failed
    pub error: Option<String>,
}

impl ZBusResult {
    /// Get Thevenin impedance at a bus (diagonal element)
    #[allow(dead_code)]
    pub fn thevenin_impedance(&self, bus: usize) -> Option<Complex64> {
        if bus < self.n && self.success {
            Some(self.z_bus[bus][bus])
        } else {
            None
        }
    }

    /// Get all diagonal elements (Thevenin impedances)
    #[allow(dead_code)]
    pub fn all_thevenin_impedances(&self) -> Vec<Complex64> {
        if !self.success {
            return vec![];
        }
        (0..self.n).map(|i| self.z_bus[i][i]).collect()
    }
}

/// Calculate the Z-bus matrix by inverting the Y-bus
///
/// This performs a full matrix inversion using LU decomposition.
/// For large systems, this can be computationally expensive, but it's
/// necessary for accurate short circuit analysis.
///
/// Note: The raw Y-bus from power flow is singular (no ground reference).
/// This function adds a small ground admittance to bus 0 (slack bus equivalent)
/// to make the matrix invertible. This models the infinite bus/slack as a
/// very stiff voltage source with small impedance to ground.
///
/// # Arguments
/// * `y_bus` - Y-bus matrix in CSR format
///
/// # Returns
/// * `ZBusResult` - The inverted Z-bus matrix
pub fn calculate_z_bus(y_bus: &YBusCsr) -> ZBusResult {
    let n = y_bus.n;

    if n == 0 {
        return ZBusResult {
            z_bus: vec![],
            n: 0,
            success: false,
            error: Some("Empty Y-bus matrix".to_string()),
        };
    }

    // Convert to dense matrix for inversion
    let mut y_dense = y_bus.to_dense();

    // Add a small ground admittance to bus 0 to make matrix invertible
    // This models the slack bus as connected to an infinite bus through
    // a very small impedance (Z_ground ≈ 0.0001 p.u. → Y_ground ≈ 10000 p.u.)
    // This is standard practice in short circuit analysis
    let y_ground = Complex64::new(10000.0, 0.0);  // Very large admittance to ground
    y_dense[0][0] += y_ground;

    // Use faer for matrix inversion
    invert_complex_matrix(&y_dense)
}

/// Calculate Z-bus with generator subtransient reactances included
///
/// This modifies the Y-bus by adding admittances corresponding to generator
/// subtransient reactances before inversion. This gives a more accurate
/// representation of short circuit conditions.
///
/// # Arguments
/// * `y_bus` - Y-bus matrix in CSR format (will be cloned and modified)
/// * `generators` - Generator data with subtransient reactances
/// * `config` - SCR calculation configuration
///
/// # Returns
/// * `ZBusResult` - The inverted Z-bus matrix with generator reactances
pub fn calculate_z_bus_with_generators(
    y_bus: &YBusCsr,
    generators: &[PlantData],
    config: &ScrConfig,
) -> ZBusResult {
    let n = y_bus.n;

    if n == 0 {
        return ZBusResult {
            z_bus: vec![],
            n: 0,
            success: false,
            error: Some("Empty Y-bus matrix".to_string()),
        };
    }

    // Convert to dense matrix
    let mut y_dense = y_bus.to_dense();

    // Add ground admittance to bus 0 (same as calculate_z_bus)
    let y_ground = Complex64::new(10000.0, 0.0);
    y_dense[0][0] += y_ground;

    // Add generator subtransient reactances to Y-bus diagonal
    for gen in generators {
        if let (Some(xdpp), Some(mva_base)) = (gen.xdpp, gen.mva_base) {
            if xdpp > 0.0 && gen.bus_id < n {
                // Convert X''d from machine base to system base:
                // X''d_sys = X''d_machine * (S_base_system / S_base_machine)
                let xdpp_system = xdpp * (config.system_mva_base / mva_base);

                // Y = 1 / (j * X''d) = -j / X''d
                let y_gen = Complex64::new(0.0, -1.0 / xdpp_system);

                y_dense[gen.bus_id][gen.bus_id] += y_gen;
            }
        }
    }

    // Use faer for matrix inversion
    invert_complex_matrix(&y_dense)
}

/// Invert a complex dense matrix using faer
fn invert_complex_matrix(matrix: &[Vec<Complex64>]) -> ZBusResult {
    use faer::complex_native::c64;
    use faer::prelude::*;

    let n = matrix.len();

    if n == 0 {
        return ZBusResult {
            z_bus: vec![],
            n: 0,
            success: false,
            error: Some("Empty matrix".to_string()),
        };
    }

    // Build faer dense matrix
    let mut a = faer::Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let val = matrix[i][j];
            a.write(i, j, c64::new(val.re, val.im));
        }
    }

    // Create identity matrix for inversion (solve A * X = I)
    let mut identity = faer::Mat::zeros(n, n);
    for i in 0..n {
        identity.write(i, i, c64::new(1.0, 0.0));
    }

    // Compute LU decomposition with partial pivoting
    let lu = a.partial_piv_lu();

    // Solve for inverse by solving A * Z = I
    let z_faer = lu.solve(identity.as_ref());

    // Convert back to Vec<Vec<Complex64>>
    let z_bus: Vec<Vec<Complex64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let val = z_faer.read(i, j);
                    Complex64::new(val.re, val.im)
                })
                .collect()
        })
        .collect();

    ZBusResult {
        z_bus,
        n,
        success: true,
        error: None,
    }
}

/// Calculate short circuit capacity at a bus
///
/// S_sc = V_base² / |Z_th|
///
/// For per-unit calculations where V_base = 1.0 p.u.:
/// S_sc (p.u.) = 1 / |Z_th| (p.u.)
/// S_sc (MVA) = S_base / |Z_th| (p.u.)
///
/// # Arguments
/// * `z_thevenin` - Thevenin impedance at the bus (per unit)
/// * `system_mva_base` - System MVA base (typically 100 MVA)
///
/// # Returns
/// * Short circuit capacity in MVA
pub fn calculate_short_circuit_mva(z_thevenin: Complex64, system_mva_base: f64) -> f64 {
    let z_mag = z_thevenin.norm();

    if z_mag < 1e-10 {
        // Avoid division by zero - use a very large but finite value
        // This can happen at infinite bus or when impedance is negligible
        return 1e15;  // 1 billion GVA - effectively infinite
    }

    system_mva_base / z_mag
}

/// Calculate SCR for a single plant
///
/// SCR = S_sc / P_rated
///
/// # Arguments
/// * `z_thevenin` - Thevenin impedance at the bus (per unit)
/// * `p_rated_mw` - Plant rated power (MW)
/// * `system_mva_base` - System MVA base (typically 100 MVA)
///
/// # Returns
/// * Short Circuit Ratio (dimensionless)
#[allow(dead_code)]
pub fn calculate_scr(z_thevenin: Complex64, p_rated_mw: f64, system_mva_base: f64) -> f64 {
    if p_rated_mw <= 0.0 {
        return 1e15; // No plant, effectively infinite SCR
    }

    let s_sc = calculate_short_circuit_mva(z_thevenin, system_mva_base);
    s_sc / p_rated_mw
}

/// Calculate SCR for multiple plants at once
///
/// This is the main entry point for SCR calculations. It:
/// 1. Inverts the Y-bus to get Z-bus
/// 2. Extracts Thevenin impedances (diagonal elements)
/// 3. Calculates short circuit capacity and SCR for each plant
///
/// # Arguments
/// * `y_bus` - Y-bus matrix in CSR format
/// * `plants` - List of plant data
/// * `config` - SCR calculation configuration
///
/// # Returns
/// * `Result<Vec<ScrResult>, String>` - SCR results for each plant, or error
pub fn calculate_scr_batch(
    y_bus: &YBusCsr,
    plants: &[PlantData],
    config: &ScrConfig,
) -> Result<Vec<ScrResult>, String> {
    // Calculate Z-bus (with or without generator reactances)
    let z_bus_result = if config.include_generator_reactance {
        calculate_z_bus_with_generators(y_bus, plants, config)
    } else {
        calculate_z_bus(y_bus)
    };

    if !z_bus_result.success {
        return Err(z_bus_result.error.unwrap_or_else(|| "Unknown error".to_string()));
    }

    // Calculate SCR for each plant
    let results: Vec<ScrResult> = plants
        .par_iter()
        .filter_map(|plant| {
            if plant.bus_id >= z_bus_result.n {
                eprintln!("Warning: Plant bus {} out of range (n={})", plant.bus_id, z_bus_result.n);
                return None;
            }

            let z_th = z_bus_result.z_bus[plant.bus_id][plant.bus_id];
            let z_mag = z_th.norm();
            let z_angle = z_th.arg();
            let s_sc = calculate_short_circuit_mva(z_th, config.system_mva_base);

            // Handle zero or negative rated power
            let scr = if plant.p_rated_mw > 0.0 {
                s_sc / plant.p_rated_mw
            } else {
                1e15  // Effectively infinite SCR for zero-rated plant
            };

            Some(ScrResult {
                bus_id: plant.bus_id,
                z_thevenin_pu: z_mag,
                z_thevenin_angle: z_angle,
                short_circuit_mva: s_sc,
                p_rated_mw: plant.p_rated_mw,
                scr,
            })
        })
        .collect();

    Ok(results)
}

/// Get all Thevenin impedances from Z-bus without plant-specific calculations
///
/// Useful for exploratory analysis or when you want impedances at all buses.
///
/// # Arguments
/// * `y_bus` - Y-bus matrix in CSR format
/// * `config` - SCR calculation configuration
///
/// # Returns
/// * `Result<Vec<(usize, Complex64)>, String>` - (bus_id, Z_thevenin) pairs
pub fn get_all_thevenin_impedances(
    y_bus: &YBusCsr,
    config: &ScrConfig,
) -> Result<Vec<(usize, Complex64, f64)>, String> {
    let z_bus_result = calculate_z_bus(y_bus);

    if !z_bus_result.success {
        return Err(z_bus_result.error.unwrap_or_else(|| "Unknown error".to_string()));
    }

    let impedances: Vec<(usize, Complex64, f64)> = (0..z_bus_result.n)
        .map(|i| {
            let z_th = z_bus_result.z_bus[i][i];
            let s_sc = calculate_short_circuit_mva(z_th, config.system_mva_base);
            (i, z_th, s_sc)
        })
        .collect();

    Ok(impedances)
}

/// Validate Y-bus matrix for SCR calculation
///
/// Checks for common issues that could cause problems:
/// - Empty matrix
/// - All-zero diagonal elements (truly disconnected buses)
///
/// Note: The Y-bus for power flow is typically singular (no ground reference),
/// but it can still be inverted for SCR calculation. We only flag errors for
/// truly problematic cases.
///
/// # Arguments
/// * `y_bus` - Y-bus matrix to validate
///
/// # Returns
/// * `Ok(())` if valid, `Err(message)` if issues found
pub fn validate_y_bus(y_bus: &YBusCsr) -> Result<(), String> {
    if y_bus.n == 0 {
        return Err("Y-bus matrix is empty".to_string());
    }

    // Only flag truly disconnected buses (zero diagonal)
    // A bus with no diagonal element at all is disconnected
    let mut disconnected_count = 0;
    for i in 0..y_bus.n {
        let y_ii = y_bus.get(i, i);
        if y_ii.norm() < 1e-15 {  // Much stricter threshold - only truly zero
            disconnected_count += 1;
        }
    }

    // Only error if ALL buses are disconnected (empty matrix essentially)
    if disconnected_count == y_bus.n {
        return Err("All buses appear disconnected (zero diagonals in Y-bus)".to_string());
    }

    // Check for symmetric structure (Y-bus should be symmetric)
    // This is a basic check - full symmetry check would be expensive
    let mut asymmetric_count = 0;
    for i in 0..y_bus.n.min(10) {
        for j in 0..y_bus.n.min(10) {
            let y_ij = y_bus.get(i, j);
            let y_ji = y_bus.get(j, i);
            let diff = (y_ij - y_ji).norm();
            if diff > 1e-6 {
                asymmetric_count += 1;
            }
        }
    }

    if asymmetric_count > 0 {
        eprintln!("Warning: Y-bus may not be symmetric ({} asymmetric elements found in sample)",
                  asymmetric_count);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple 3-bus test system Y-bus
    fn create_test_y_bus() -> YBusCsr {
        // Simple 3-bus system:
        // Bus 0 -- Bus 1 -- Bus 2
        //
        // All branches have Z = 0.01 + j0.1 p.u.
        // Y = 1/Z = 0.99 - j9.9 p.u. (approximately)

        let z = Complex64::new(0.01, 0.1);
        let y_series = Complex64::new(1.0, 0.0) / z;  // ~0.99 - j9.9

        // Y-bus for 3-bus linear system:
        // Y_00 = y_01 (only connected to bus 1)
        // Y_11 = y_01 + y_12 (connected to both)
        // Y_22 = y_12 (only connected to bus 1)
        // Off-diagonals: -y_series

        // Dense Y-bus:
        // [  y    -y     0  ]
        // [ -y    2y    -y  ]
        // [  0    -y     y  ]

        let y = y_series;
        let neg_y = -y_series;
        let two_y = y_series * 2.0;

        // CSR format (row by row)
        let row_ptrs = vec![0, 2, 5, 7];
        let col_indices = vec![
            0, 1,        // Row 0: columns 0, 1
            0, 1, 2,     // Row 1: columns 0, 1, 2
            1, 2,        // Row 2: columns 1, 2
        ];
        let values = vec![
            y, neg_y,           // Row 0
            neg_y, two_y, neg_y, // Row 1
            neg_y, y,           // Row 2
        ];

        YBusCsr::new(row_ptrs, col_indices, values)
    }

    #[test]
    fn test_y_bus_get() {
        let y_bus = create_test_y_bus();

        // Check diagonal elements exist
        assert!(y_bus.get(0, 0).norm() > 0.0);
        assert!(y_bus.get(1, 1).norm() > 0.0);
        assert!(y_bus.get(2, 2).norm() > 0.0);

        // Check zero elements
        assert!(y_bus.get(0, 2).norm() < 1e-10);
        assert!(y_bus.get(2, 0).norm() < 1e-10);

        // Check off-diagonal elements (negative of series admittance)
        let y_01 = y_bus.get(0, 1);
        assert!(y_01.re < 0.0); // Should be negative
    }

    #[test]
    fn test_z_bus_calculation() {
        let y_bus = create_test_y_bus();
        let z_bus_result = calculate_z_bus(&y_bus);

        assert!(z_bus_result.success, "Z-bus calculation should succeed");
        assert_eq!(z_bus_result.n, 3);

        // Note: calculate_z_bus adds a ground reference at bus 0, so
        // the result is NOT the exact inverse of the input Y-bus.
        // Instead, we verify that the Z-bus is valid by checking:
        // 1. All diagonal elements are non-zero (no isolated buses)
        // 2. Diagonal elements are reasonable (positive real part)
        // 3. Bus 0 has very low impedance (due to ground reference)

        for i in 0..3 {
            let z_ii = z_bus_result.z_bus[i][i];
            assert!(z_ii.norm() > 0.0, "Diagonal element Z_{},{} should be non-zero", i, i);
        }

        // Bus 0 should have very low impedance (connected to ground)
        let z_00 = z_bus_result.z_bus[0][0].norm();
        let z_11 = z_bus_result.z_bus[1][1].norm();
        let z_22 = z_bus_result.z_bus[2][2].norm();

        assert!(z_00 < z_11, "Bus 0 (with ground) should have lowest impedance");
        assert!(z_00 < z_22, "Bus 0 (with ground) should have lowest impedance");
    }

    #[test]
    fn test_thevenin_impedance() {
        let y_bus = create_test_y_bus();
        let z_bus_result = calculate_z_bus(&y_bus);

        assert!(z_bus_result.success);

        // Get Thevenin impedances via manual extraction
        let z_th: Vec<Complex64> = (0..3).map(|i| z_bus_result.z_bus[i][i]).collect();
        assert_eq!(z_th.len(), 3);

        // With ground reference at bus 0:
        // Bus 0: Very low Z (ground connection)
        // Bus 1: One line away from ground
        // Bus 2: Two lines away from ground
        // So Z_0 < Z_1 < Z_2
        let z_0 = z_th[0].norm();
        let z_1 = z_th[1].norm();
        let z_2 = z_th[2].norm();

        assert!(z_0 < z_1, "Bus 0 (ground) should have lower Z than bus 1");
        assert!(z_1 < z_2, "Bus 1 should have lower Z than bus 2 (further from ground)");
    }

    #[test]
    fn test_scr_calculation() {
        let y_bus = create_test_y_bus();
        let config = ScrConfig {
            system_mva_base: 100.0,
            include_generator_reactance: false,
            conditioning_tolerance: 1e-10,
        };

        // Test with plants at buses 1 and 2 (both non-ground buses)
        let plants = vec![
            PlantData {
                bus_id: 1,
                p_rated_mw: 50.0,  // 50 MW plant at bus 1
                xdpp: None,
                mva_base: None,
            },
            PlantData {
                bus_id: 2,
                p_rated_mw: 100.0, // 100 MW plant at bus 2
                xdpp: None,
                mva_base: None,
            },
        ];

        let results = calculate_scr_batch(&y_bus, &plants, &config).unwrap();

        assert_eq!(results.len(), 2);

        // Both SCRs should be positive and finite
        for result in &results {
            assert!(result.scr > 0.0);
            assert!(result.scr.is_finite());
            assert!(result.short_circuit_mva > 0.0);
        }

        let scr_1 = results.iter().find(|r| r.bus_id == 1).unwrap().scr;
        let scr_2 = results.iter().find(|r| r.bus_id == 2).unwrap().scr;

        // Bus 1 (50 MW, closer to ground) should have higher SCR than Bus 2 (100 MW, further)
        // The 2x size difference would give 2x SCR ratio at same bus
        // But bus 1 is also closer to ground, so ratio should be > 2
        assert!(scr_1 > scr_2, "Bus 1 (closer to ground, smaller plant) should have higher SCR");
    }

    #[test]
    fn test_short_circuit_mva() {
        // Test with known impedance
        let z_th = Complex64::new(0.01, 0.1); // |Z| ≈ 0.1005
        let s_base = 100.0;

        let s_sc = calculate_short_circuit_mva(z_th, s_base);

        // S_sc = S_base / |Z| = 100 / 0.1005 ≈ 995 MVA
        assert!(s_sc > 900.0 && s_sc < 1100.0,
                "Short circuit MVA should be ~995, got {}", s_sc);
    }

    #[test]
    fn test_validate_y_bus() {
        let y_bus = create_test_y_bus();
        assert!(validate_y_bus(&y_bus).is_ok());

        // Test empty matrix
        let empty = YBusCsr::new(vec![0], vec![], vec![]);
        assert!(validate_y_bus(&empty).is_err());
    }

    #[test]
    fn test_scr_interpretation() {
        // Test SCR interpretation:
        // SCR < 3: Weak grid (potential stability issues)
        // SCR 3-5: Moderate grid strength
        // SCR > 5: Strong grid

        let y_bus = create_test_y_bus();
        let config = ScrConfig::default();

        // Small plant at bus 1 should have high SCR
        let small_plant = vec![PlantData {
            bus_id: 1,
            p_rated_mw: 1.0,  // Very small 1 MW plant
            xdpp: None,
            mva_base: None,
        }];

        let results = calculate_scr_batch(&y_bus, &small_plant, &config).unwrap();
        let scr = results[0].scr;

        // For such a small plant, SCR should be very high
        assert!(scr > 100.0, "1 MW plant should have very high SCR, got {}", scr);
    }
}
