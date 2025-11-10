# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - 2025-11-10

### Changed
- Fix all Rust compiler warnings for cleaner local builds
- Remove unused imports and variables
- Update rustler::init to modern syntax without explicit NIF list
- Improve code quality for developers compiling from source

## [0.1.4] - 2025-11-06

### Changed
- Simplify GitHub workflow to only build Linux binaries
- Update Rust to 1.85.1
- Remove macOS precompiled builds (developers can build locally)

## [0.1.3] - 2025-11-06

### Fixed
- Fix compilation issues and add precompiled NIF infrastructure
- Improve solution map to use bus IDs instead of array indices
- Handle optional bus fields (q_min, q_max) safely

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
