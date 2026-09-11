
# PSIQIT

**English** | [فارسی](README.fa.md)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.3-green.svg)](https://github.com/psiqit/psiqit/releases)
[![CI](https://github.com/psiqit/psiqit/actions/workflows/ci.yml/badge.svg)](https://github.com/psiqit/psiqit/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/1365410690.svg)](https://doi.org/10.5281/zenodo.22702331)

**P**ython **S**cientific **Q**uantum **I**nformation **T**oolkit

PSIQIT is a comprehensive, high-performance quantum computing simulation library designed for researchers, students, and developers. It provides a robust framework for simulating quantum circuits, open quantum systems, variational algorithms, and quantum error correction.

## Key Features

- **Circuit Simulation:** Intuitive circuit building with consistent LSB-first qubit ordering.
- **Quantum Algorithms:** Built-in implementations of Grover, Shor, Deutsch-Jozsa, Bernstein-Vazirani, and QPE.
- **Open Quantum Systems:** Solvers for the Lindblad master equation, quantum trajectories, and time evolution.
- **Variational Methods:** Frameworks for VQE, QAOA, and SSVQE with custom ansatz support.
- **Information Theory:** Tools for calculating von Neumann entropy, concurrence, negativity, and mutual information.
- **Error Correction:** Implementations of Bit-Flip, Phase-Flip, Shor, and Steane codes.

## Installation

Install the latest stable release via pip:

```bash
pip install psiqit
```

Or install directly from the repository for the latest features:

```bash
git clone https://github.com/psiqit/psiqit.git
cd psiqit
pip install -e .
```

## Quick Start

### Creating a Bell State

```python
from psiqit.circuits import QuantumCircuit

# Initialize a 2-qubit circuit
circ = QuantumCircuit(2)

# Apply Hadamard and CNOT
circ.h(0).cx(0, 1)

# Run and measure
state = circ.run()
print("State Vector:", state.data)

result = circ.measure(shots=1024)
print("Measurement Counts:", result['counts'])
```

### Running Grover's Search

```python
from psiqit.algorithms import grover_search

# Search for state |101> (5) in a 3-qubit database
result = grover_search(n_qubits=3, target=5, shots=1000)

print(f"Found target: {result['most_likely']}")
print(f"Success probability: {result['probability_target']:.2%}")
```

### Solving Lindblad Equation

```python
from psiqit.quantum import pauli_z, pauli_x
from psiqit.dynamics import LindbladSolver

# Define Hamiltonian and collapse operators
H = pauli_z()
L = [pauli_x()]

# Solve the master equation
solver = LindbladSolver(H, L, gamma=[0.1])
result = solver.solve(t_span=(0, 10), dt=0.01)
```

## Project Structure

```text
psiqit/
├── psiqit/                 # Core library modules
│   ├── quantum/            # States, operators, and measurements
│   ├── circuits/           # Circuit building and execution
│   ├── algorithms/         # Quantum algorithms
│   ├── dynamics/           # Lindblad, Trotter, and adiabatic evolution
│   ├── info/               # Entropy, fidelity, and entanglement
│   ├── variational/        # VQE, QAOA frameworks
│   ├── error_correction/   # QEC codes
│   ├── qml/                # Quantum machine learning
│   └── visualization/      # Bloch sphere, circuit drawing
├── tests/                  # Comprehensive test suite
├── examples/               # Jupyter notebooks and use-cases
├── docs/                   # Project documentation
└── pyproject.toml          # Build and packaging configuration
```

## Documentation

Full documentation is available at [https://psiqit.github.io/psiqit/](https://psiqit.github.io/psiqit/).

The documentation includes:

- **API Reference:** Detailed documentation for all modules and classes.
- **Tutorials:** Step-by-step guides for common quantum computing tasks.
- **Examples:** Real-world applications and research use-cases.

## Testing

PSIQIT comes with a rigorous test suite. To run all tests:

```bash
pytest tests/
# Or run the comprehensive suite
python test_computational.py
```

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use PSIQIT in your research, please cite it using the DOI below. You can also click the **"Cite this repository"** button on the right sidebar of this page for pre-formatted citations in various formats (BibTeX, APA, IEEE, etc.).

[![DOI](https://zenodo.org/badge/1365410690.svg)](https://doi.org/10.5281/zenodo.22702331)

```bibtex
@software{psiqit2026,
  author       = {Azadmarzabadi, Mahdi},
  title        = {PSIQIT: Python Scientific Quantum Information Toolkit},
  version      = {1.0.3},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22702331},
  url          = {https://doi.org/10.5281/zenodo.22702331}
}
```

**APA Style:**
```
Azadmarzabadi, M. (2026). PSIQIT: Python Scientific Quantum Information Toolkit (Version 1.0.3) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22702331
```

## Contact

**Mahdi Azadmarzabadi** - [psiqitofficial@protonmail.com](mailto:psiqitofficial@protonmail.com)  
Project Link: [https://github.com/psiqit/psiqit](https://github.com/psiqit/psiqit)
```

---

### 💡 تغییرات اعمال شده:

1. **اضافه کردن DOI به BibTeX**: خطوط `doi` و `url` اضافه شدند که باعث می‌شود این entry به طور کامل در Google Scholar و دیگر پایگاه‌های علمی نمایه شود.
2. **اضافه کردن publisher**: Zenodo به عنوان ناشر رسمی ثبت شد.
3. **توضیح کاربردی**: یک جمله درباره‌ی دکمه‌ی "Cite this repository" در سایدبار گیت‌هاب اضافه شد.
4. **فرمت APA**: برای کاربرانی که از LaTeX استفاده نمی‌کنند، فرمت APA هم اضافه شد.

---

### 🚀 قدم‌های نهایی برای انتشار:

```bash
# 1. فایل‌های تغییر یافته را اضافه کنید
git add README.md README.fa.md CITATION.cff

# 2. کامیت کنید
git commit -m "Update README with official Zenodo DOI"

# 3. پوش کنید
git push

# 4. مستندات را روی GitHub Pages آپلود کنید
mkdocs gh-deploy
