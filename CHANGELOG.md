# Changelog

All notable changes to PSIQIT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] - 2026-09-04

### Added
- Comprehensive test suite with 123 tests covering all modules
- LSB-first qubit ordering architecture for consistent tensor operations
- Multi-controlled Z gate decomposition using Toffoli gates
- RK4 solver for Lindblad master equation with improved numerical stability
- Feedback protocol with cooldown mechanism for quantum heat-exchange simulations
- Bilingual README (English and Persian)
- GitHub Actions workflows for CI/CD and automated PyPI deployment
- Full API documentation structure with MkDocs

### Fixed
- Qubit ordering inconsistency in multi-qubit gate operations
- Gate parser regex for handling commas within parentheses
- Ancilla qubit handling in Deutsch-Jozsa and Bernstein-Vazirani algorithms
- Toffoli gate implementation using bit-shift logic for LSB-first compatibility
- W-state circuit generation for 3-qubit systems
- Numerical stability issues in open quantum system simulations

### Improved
- Test coverage increased to 100%
- Documentation structure reorganized for better navigation
- Performance optimizations in circuit simulation
- Enhanced error messages and logging

## [1.0.2] - 2026-08-15

### Added
- Quantum machine learning module (QML)
- Quantum kernel methods implementation
- Variational quantum classifier (VQC)
- Advanced visualization tools (Wigner and Husimi functions)

### Fixed
- Memory leak in large circuit simulations
- Phase estimation accuracy for multi-qubit systems

## [1.0.1] - 2026-07-20

### Added
- Optical circuit simulation module
- Polarization utilities
- Symbolic mathematics integration

### Fixed
- Entropy calculation edge cases
- Circuit drawing alignment issues

## [1.0.0] - 2026-06-01

### Added
- Initial release
- Core quantum state and operator classes
- Basic circuit simulation engine
- Quantum algorithms: Grover, Shor, Deutsch-Jozsa, Bernstein-Vazirani, QPE
- Information theory module
- Variational algorithms (VQE, QAOA)
- Error correction codes
- Lindblad equation solver
- Basic visualization tools