# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-11-06

### Added
- Initial release of PowerFlowSolver
- Complete Newton-Raphson power flow solver in Rust
- Sparse linear algebra operations using faer library
- Q-limit enforcement for realistic generator modeling
- Parallel Jacobian building using rayon
- Pre-compiled NIFs for major platforms (macOS, Linux)
- Comprehensive documentation and examples

### Features
- Zero boundary crossings during iteration (entire loop in Rust)
- Optimized for large power systems (10,000+ buses)
- Sparse matrix operations in O(nnz) time
- Support for slack, PV, and PQ buses
- Flat start initialization

[0.1.0]: https://github.com/voltageAI/power_flow_solver/releases/tag/v0.1.0
