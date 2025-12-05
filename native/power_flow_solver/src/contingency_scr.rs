//! Contingency Short Circuit Ratio Calculations
//!
//! Provides efficient N-1 contingency analysis for SCR calculations.
//! Used by FRT (Fault Ride Through) assessment to find worst-case
//! grid strength scenarios under single-element outages.
//!
//! # Background
//!
//! During FRT assessment, we need to evaluate grid strength not just
//! under normal conditions, but under credible N-1 contingencies.
//! A transmission line outage can significantly reduce the short circuit
//! capacity at a point of interconnection (POI), potentially causing
//! inverter-based resources to lose synchronization.
//!
//! # Approach
//!
//! For each contingency:
//! 1. Modify Y-bus by removing the branch contribution
//! 2. Invert modified Y-bus to get Z-bus
//! 3. Extract Thevenin impedance (Z_th = Z_ii at POI)
//! 4. Calculate short circuit MVA: S_sc = S_base / |Z_th|
//!
//! The batch calculation parallelizes across contingencies using rayon.

use num_complex::Complex64;
use rayon::prelude::*;

use crate::scr::{calculate_short_circuit_mva, calculate_z_bus, ScrConfig, YBusCsr};

/// Branch data for contingency analysis.
///
/// Represents a transmission element that can be taken out of service.
/// The admittance values should be on the system MVA base.
#[derive(Clone, Debug)]
pub struct BranchData {
    /// Unique identifier for the branch
    pub id: usize,
    /// From bus index (0-based matrix index, not bus ID)
    pub from_bus: usize,
    /// To bus index (0-based matrix index, not bus ID)
    pub to_bus: usize,
    /// Series admittance Y = 1/(R + jX) in per-unit
    pub y_series: Complex64,
    /// Total shunt admittance (line charging B) in per-unit
    /// Half goes to each end of the line
    pub y_shunt: Complex64,
}

/// Result of a single contingency SCR calculation.
#[derive(Clone, Debug)]
pub struct ContingencyScrResult {
    /// Branch ID that was removed
    pub branch_id: usize,
    /// Thevenin impedance at POI with this branch out
    pub z_thevenin: Complex64,
    /// Short circuit MVA at POI with this branch out
    pub short_circuit_mva: f64,
    /// Whether calculation succeeded (false if contingency causes islanding)
    pub success: bool,
    /// Error description if calculation failed
    #[allow(dead_code)]
    pub error: Option<String>,
}

/// Modify Y-bus by removing a branch.
///
/// When a branch is removed from the network:
/// - The series admittance contribution is subtracted from both diagonal
///   elements and added back to off-diagonal elements
/// - The shunt admittance (line charging) is subtracted from diagonals
///
/// # Y-bus Branch Contribution Review
///
/// For a branch between buses i and j with series admittance y_s and
/// total shunt admittance y_sh:
///
/// ```text
/// Y[i,i] += y_s + y_sh/2    (diagonal: series + half shunt)
/// Y[j,j] += y_s + y_sh/2    (diagonal: series + half shunt)
/// Y[i,j] -= y_s             (off-diagonal: negative series)
/// Y[j,i] -= y_s             (off-diagonal: negative series)
/// ```
///
/// To remove the branch, we reverse these operations.
///
/// # Arguments
///
/// * `y_bus` - Original Y-bus matrix (not modified)
/// * `branch` - Branch to remove
///
/// # Returns
///
/// New Y-bus with the branch removed, or error if invalid
pub fn remove_branch_from_ybus(y_bus: &YBusCsr, branch: &BranchData) -> Result<YBusCsr, String> {
    let n = y_bus.n;

    // Validate bus indices
    if branch.from_bus >= n {
        return Err(format!(
            "From bus index {} out of range for {}-bus system",
            branch.from_bus, n
        ));
    }
    if branch.to_bus >= n {
        return Err(format!(
            "To bus index {} out of range for {}-bus system",
            branch.to_bus, n
        ));
    }
    if branch.from_bus == branch.to_bus {
        return Err("Branch cannot connect bus to itself".to_string());
    }

    // Convert to dense for modification
    // (For production with very large systems, could optimize to sparse updates)
    let mut y_dense = y_bus.to_dense();

    let y_s = branch.y_series;
    let y_sh_half = branch.y_shunt * 0.5;

    // Remove branch contribution (reverse of Y-bus building)
    // Diagonal: subtract series admittance and half shunt from each end
    y_dense[branch.from_bus][branch.from_bus] -= y_s + y_sh_half;
    y_dense[branch.to_bus][branch.to_bus] -= y_s + y_sh_half;

    // Off-diagonal: the original Y-bus has -y_s at off-diagonals
    // Removing the branch means adding y_s back (making it less negative or zero)
    y_dense[branch.from_bus][branch.to_bus] += y_s;
    y_dense[branch.to_bus][branch.from_bus] += y_s;

    // Convert back to CSR
    Ok(dense_to_csr(&y_dense))
}

/// Convert dense complex matrix to CSR format.
///
/// Only stores elements with magnitude above threshold to maintain sparsity.
fn dense_to_csr(dense: &[Vec<Complex64>]) -> YBusCsr {
    let mut row_ptrs = vec![0usize];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    const SPARSITY_THRESHOLD: f64 = 1e-15;

    for row in dense.iter() {
        for (col, &val) in row.iter().enumerate() {
            if val.norm() > SPARSITY_THRESHOLD {
                col_indices.push(col);
                values.push(val);
            }
        }
        row_ptrs.push(col_indices.len());
    }

    YBusCsr::new(row_ptrs, col_indices, values)
}

/// Calculate SCR with a single branch contingency.
///
/// # Arguments
///
/// * `y_bus` - Original Y-bus matrix
/// * `branch` - Branch to remove for contingency
/// * `poi_bus` - Bus index where to calculate Thevenin impedance
/// * `config` - SCR calculation configuration
///
/// # Returns
///
/// Contingency result with Z_th and S_sc, or error information
pub fn calculate_contingency_scr(
    y_bus: &YBusCsr,
    branch: &BranchData,
    poi_bus: usize,
    config: &ScrConfig,
) -> ContingencyScrResult {
    // Modify Y-bus by removing the branch
    let modified_ybus = match remove_branch_from_ybus(y_bus, branch) {
        Ok(y) => y,
        Err(e) => {
            return ContingencyScrResult {
                branch_id: branch.id,
                z_thevenin: Complex64::new(f64::NAN, f64::NAN),
                short_circuit_mva: 0.0,
                success: false,
                error: Some(format!("Failed to modify Y-bus: {}", e)),
            };
        }
    };

    // Check if POI bus is valid
    if poi_bus >= modified_ybus.n {
        return ContingencyScrResult {
            branch_id: branch.id,
            z_thevenin: Complex64::new(f64::NAN, f64::NAN),
            short_circuit_mva: 0.0,
            success: false,
            error: Some(format!(
                "POI bus {} out of range for {}-bus system",
                poi_bus, modified_ybus.n
            )),
        };
    }

    // Calculate Z-bus by inverting modified Y-bus
    let z_bus_result = calculate_z_bus(&modified_ybus);

    if !z_bus_result.success {
        return ContingencyScrResult {
            branch_id: branch.id,
            z_thevenin: Complex64::new(f64::NAN, f64::NAN),
            short_circuit_mva: 0.0,
            success: false,
            error: z_bus_result
                .error
                .or_else(|| Some("Z-bus calculation failed (possible islanding)".to_string())),
        };
    }

    // Extract Thevenin impedance at POI (diagonal element of Z-bus)
    let z_th = z_bus_result.z_bus[poi_bus][poi_bus];

    // Check for invalid impedance (can happen with numerical issues)
    if !z_th.re.is_finite() || !z_th.im.is_finite() {
        return ContingencyScrResult {
            branch_id: branch.id,
            z_thevenin: z_th,
            short_circuit_mva: 0.0,
            success: false,
            error: Some("Thevenin impedance is not finite (numerical issue)".to_string()),
        };
    }

    // Calculate short circuit MVA
    let s_sc = calculate_short_circuit_mva(z_th, config.system_mva_base);

    ContingencyScrResult {
        branch_id: branch.id,
        z_thevenin: z_th,
        short_circuit_mva: s_sc,
        success: true,
        error: None,
    }
}

/// Calculate SCR for multiple contingencies in parallel.
///
/// Uses rayon for parallel execution across contingencies.
/// Each contingency is independent, making this embarrassingly parallel.
///
/// # Arguments
///
/// * `y_bus` - Original Y-bus matrix (shared across all contingencies)
/// * `branches` - List of branches to test as contingencies
/// * `poi_bus` - Bus index where to calculate Thevenin impedance
/// * `config` - SCR calculation configuration
///
/// # Returns
///
/// Vector of results, one per branch (in same order as input)
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

/// Find the worst-case (lowest S_sc) contingency at a POI.
///
/// Scans all provided contingencies and returns the one that results
/// in the lowest short circuit capacity, indicating the weakest grid condition.
///
/// # Arguments
///
/// * `y_bus` - Original Y-bus matrix
/// * `branches` - List of branches to test as contingencies
/// * `poi_bus` - Bus index where to calculate Thevenin impedance
/// * `config` - SCR calculation configuration
///
/// # Returns
///
/// The contingency with lowest S_sc, or None if all contingencies fail
#[allow(dead_code)]
pub fn find_worst_contingency(
    y_bus: &YBusCsr,
    branches: &[BranchData],
    poi_bus: usize,
    config: &ScrConfig,
) -> Option<ContingencyScrResult> {
    let results = calculate_contingency_scr_batch(y_bus, branches, poi_bus, config);

    results
        .into_iter()
        .filter(|r| r.success && r.short_circuit_mva.is_finite() && r.short_circuit_mva > 0.0)
        .min_by(|a, b| {
            a.short_circuit_mva
                .partial_cmp(&b.short_circuit_mva)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple 3-bus radial system Y-bus for testing.
    ///
    /// Topology: Slack(0) -- Bus(1) -- Bus(2)
    ///
    /// Both branches have Z = 0.01 + j0.1 pu (typical transmission line)
    /// This gives Y_series = 1/Z ≈ 0.99 - j9.9 pu
    fn create_3bus_radial_ybus() -> YBusCsr {
        let z = Complex64::new(0.01, 0.1);
        let y_series = Complex64::new(1.0, 0.0) / z;

        // Y-bus structure for radial system:
        // Bus 0: connected to bus 1 only
        // Bus 1: connected to buses 0 and 2 (hub)
        // Bus 2: connected to bus 1 only

        let row_ptrs = vec![0, 2, 5, 7];
        let col_indices = vec![
            0, 1, // Row 0: Y[0,0], Y[0,1]
            0, 1, 2, // Row 1: Y[1,0], Y[1,1], Y[1,2]
            1, 2, // Row 2: Y[2,1], Y[2,2]
        ];
        let values = vec![
            y_series,
            -y_series, // Row 0
            -y_series,
            y_series * 2.0,
            -y_series, // Row 1 (hub has 2x on diagonal)
            -y_series,
            y_series, // Row 2
        ];

        YBusCsr::new(row_ptrs, col_indices, values)
    }

    /// Create a 4-bus meshed system for testing parallel paths.
    ///
    /// Topology:
    ///   Slack(0) ---- Bus(1)
    ///      |            |
    ///   Bus(2) ------ Bus(3)
    ///
    /// All branches have same impedance for simplicity.
    fn create_4bus_mesh_ybus() -> (YBusCsr, Vec<BranchData>) {
        let z = Complex64::new(0.01, 0.1);
        let y_series = Complex64::new(1.0, 0.0) / z;
        let neg_y = -y_series;

        // Branches: 0-1, 0-2, 1-3, 2-3
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
                from_bus: 0,
                to_bus: 2,
                y_series,
                y_shunt: Complex64::new(0.0, 0.0),
            },
            BranchData {
                id: 2,
                from_bus: 1,
                to_bus: 3,
                y_series,
                y_shunt: Complex64::new(0.0, 0.0),
            },
            BranchData {
                id: 3,
                from_bus: 2,
                to_bus: 3,
                y_series,
                y_shunt: Complex64::new(0.0, 0.0),
            },
        ];

        // Each bus connects to 2 others, so diagonal = 2 * y_series
        let two_y = y_series * 2.0;

        // Dense Y-bus (for clarity):
        // [  2y   -y   -y    0 ]   Bus 0: connects to 1,2
        // [ -y    2y    0   -y ]   Bus 1: connects to 0,3
        // [ -y     0   2y   -y ]   Bus 2: connects to 0,3
        // [  0    -y   -y   2y ]   Bus 3: connects to 1,2

        let row_ptrs = vec![0, 3, 6, 9, 12];
        let col_indices = vec![
            0, 1, 2, // Row 0
            0, 1, 3, // Row 1
            0, 2, 3, // Row 2
            1, 2, 3, // Row 3
        ];
        let values = vec![
            two_y, neg_y, neg_y, // Row 0
            neg_y, two_y, neg_y, // Row 1
            neg_y, two_y, neg_y, // Row 2
            neg_y, neg_y, two_y, // Row 3
        ];

        (YBusCsr::new(row_ptrs, col_indices, values), branches)
    }

    #[test]
    fn test_remove_branch_basic() {
        let y_bus = create_3bus_radial_ybus();

        let z = Complex64::new(0.01, 0.1);
        let y_series = Complex64::new(1.0, 0.0) / z;

        // Remove branch 1-2 (isolates bus 2)
        let branch = BranchData {
            id: 1,
            from_bus: 1,
            to_bus: 2,
            y_series,
            y_shunt: Complex64::new(0.0, 0.0),
        };

        let modified = remove_branch_from_ybus(&y_bus, &branch).unwrap();

        // After removing 1-2:
        // - Y[1,1] should decrease by y_series (was 2*y, now y)
        // - Y[2,2] should be ~0 (isolated)
        // - Y[1,2] and Y[2,1] should be ~0

        let y_22 = modified.get(2, 2);
        assert!(
            y_22.norm() < 1e-10,
            "Bus 2 diagonal should be ~0 after isolation, got {:?}",
            y_22
        );

        let y_12 = modified.get(1, 2);
        assert!(
            y_12.norm() < 1e-10,
            "Y[1,2] should be ~0 after branch removal, got {:?}",
            y_12
        );
    }

    #[test]
    fn test_remove_branch_with_shunt() {
        let y_bus = create_3bus_radial_ybus();

        let z = Complex64::new(0.01, 0.1);
        let y_series = Complex64::new(1.0, 0.0) / z;
        let y_shunt = Complex64::new(0.0, 0.02); // Line charging

        let branch = BranchData {
            id: 0,
            from_bus: 0,
            to_bus: 1,
            y_series,
            y_shunt,
        };

        // Should succeed without error
        let result = remove_branch_from_ybus(&y_bus, &branch);
        assert!(result.is_ok());
    }

    #[test]
    fn test_remove_branch_invalid_bus() {
        let y_bus = create_3bus_radial_ybus();

        let branch = BranchData {
            id: 0,
            from_bus: 0,
            to_bus: 99, // Invalid
            y_series: Complex64::new(1.0, -10.0),
            y_shunt: Complex64::new(0.0, 0.0),
        };

        let result = remove_branch_from_ybus(&y_bus, &branch);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("out of range"));
    }

    #[test]
    fn test_remove_branch_self_loop() {
        let y_bus = create_3bus_radial_ybus();

        let branch = BranchData {
            id: 0,
            from_bus: 1,
            to_bus: 1, // Self-loop
            y_series: Complex64::new(1.0, -10.0),
            y_shunt: Complex64::new(0.0, 0.0),
        };

        let result = remove_branch_from_ybus(&y_bus, &branch);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("itself"));
    }

    #[test]
    fn test_contingency_scr_radial_causes_islanding() {
        let y_bus = create_3bus_radial_ybus();
        let config = ScrConfig::default();

        let z = Complex64::new(0.01, 0.1);
        let y_series = Complex64::new(1.0, 0.0) / z;

        // In a radial system, removing the only path to a bus causes islanding
        // Remove branch 0-1 and check SCR at bus 2
        let branch = BranchData {
            id: 0,
            from_bus: 0,
            to_bus: 1,
            y_series,
            y_shunt: Complex64::new(0.0, 0.0),
        };

        let result = calculate_contingency_scr(&y_bus, &branch, 2, &config);

        // With branch 0-1 out in a radial system, buses 1 and 2 are
        // disconnected from the slack (bus 0). This should either:
        // 1. Cause matrix singularity (calculation fails)
        // 2. Result in very high impedance (very low S_sc)
        //
        // The Z-bus calculation adds ground reference at bus 0, so
        // the isolated island (buses 1,2) may still invert but with
        // meaningless values. We check for either failure or very low S_sc.

        if result.success {
            // If it "succeeds", the S_sc should be much lower than base case
            // because the isolated portion has no real connection to slack
            assert!(
                result.short_circuit_mva < 100.0,
                "Islanded bus should have very low S_sc"
            );
        }
        // Failure is also acceptable for islanding condition
    }

    #[test]
    fn test_contingency_scr_mesh_reduces_ssc() {
        let (y_bus, branches) = create_4bus_mesh_ybus();
        let config = ScrConfig::default();

        // First, calculate base case S_sc at bus 3 (furthest from slack)
        let z_bus_base = calculate_z_bus(&y_bus);
        assert!(z_bus_base.success);
        let s_sc_base = calculate_short_circuit_mva(z_bus_base.z_bus[3][3], config.system_mva_base);

        // Now test contingency: remove branch 1-3
        // Bus 3 is still connected via 0-2-3 path, but with higher impedance
        let result = calculate_contingency_scr(&y_bus, &branches[2], 3, &config);

        assert!(
            result.success,
            "Contingency should succeed in mesh (alternate path exists)"
        );
        assert!(
            result.short_circuit_mva < s_sc_base,
            "S_sc should decrease with contingency: base={:.1}, contingency={:.1}",
            s_sc_base,
            result.short_circuit_mva
        );
    }

    #[test]
    fn test_contingency_batch_parallel() {
        let (y_bus, branches) = create_4bus_mesh_ybus();
        let config = ScrConfig::default();

        // Test batch calculation at bus 3
        let results = calculate_contingency_scr_batch(&y_bus, &branches, 3, &config);

        assert_eq!(results.len(), 4, "Should have result for each branch");

        // All contingencies should succeed in this mesh system
        for result in &results {
            assert!(
                result.success,
                "Branch {} contingency should succeed",
                result.branch_id
            );
            assert!(
                result.short_circuit_mva > 0.0,
                "S_sc should be positive"
            );
        }

        // Results should be in same order as input branches
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.branch_id, branches[i].id);
        }
    }

    #[test]
    fn test_find_worst_contingency() {
        let (y_bus, branches) = create_4bus_mesh_ybus();
        let config = ScrConfig::default();

        let worst = find_worst_contingency(&y_bus, &branches, 3, &config);

        assert!(worst.is_some(), "Should find a worst contingency");

        let worst = worst.unwrap();
        assert!(worst.success);

        // Verify it actually is the worst by comparing to all results
        let all_results = calculate_contingency_scr_batch(&y_bus, &branches, 3, &config);
        let min_ssc = all_results
            .iter()
            .filter(|r| r.success)
            .map(|r| r.short_circuit_mva)
            .fold(f64::INFINITY, f64::min);

        assert!(
            (worst.short_circuit_mva - min_ssc).abs() < 1e-6,
            "Worst should have minimum S_sc"
        );
    }

    #[test]
    fn test_find_worst_contingency_all_fail() {
        let y_bus = create_3bus_radial_ybus();
        let config = ScrConfig::default();

        let z = Complex64::new(0.01, 0.1);
        let y_series = Complex64::new(1.0, 0.0) / z;

        // In radial system with only one path to bus 2
        // Removing branch 1-2 isolates bus 2
        let branches = vec![BranchData {
            id: 1,
            from_bus: 1,
            to_bus: 2,
            y_series,
            y_shunt: Complex64::new(0.0, 0.0),
        }];

        let worst = find_worst_contingency(&y_bus, &branches, 2, &config);

        // May return None if islanding causes calculation failure
        // Or may return a result with very low S_sc
        // Both are acceptable behaviors for this edge case
        if let Some(w) = worst {
            // If it returns something, it should be the only option
            assert_eq!(w.branch_id, 1);
        }
    }

    #[test]
    fn test_dense_to_csr_preserves_values() {
        let dense = vec![
            vec![
                Complex64::new(10.0, -5.0),
                Complex64::new(-5.0, 2.5),
                Complex64::new(0.0, 0.0),
            ],
            vec![
                Complex64::new(-5.0, 2.5),
                Complex64::new(15.0, -7.5),
                Complex64::new(-10.0, 5.0),
            ],
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(-10.0, 5.0),
                Complex64::new(10.0, -5.0),
            ],
        ];

        let csr = dense_to_csr(&dense);

        // Verify non-zero elements are preserved
        assert_eq!(csr.n, 3);

        // Check specific elements
        let y_00 = csr.get(0, 0);
        assert!((y_00.re - 10.0).abs() < 1e-10);
        assert!((y_00.im - (-5.0)).abs() < 1e-10);

        // Zero element should return zero
        let y_02 = csr.get(0, 2);
        assert!(y_02.norm() < 1e-10);
    }
}
