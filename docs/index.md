# Welcome to PSIQIT

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.3-green.svg)](https://github.com/psiqit/psiqit/releases)

**P**ython **S**cientific **Q**uantum **I**nformation **T**oolkit

PSIQIT is a comprehensive, high-performance quantum computing simulation library designed for researchers, students, and developers. It provides a robust framework for simulating quantum circuits, open quantum systems, variational algorithms, and quantum error correction.

---

## ✨ Key Features

- **🧩 Circuit Simulation:** Intuitive circuit building with consistent LSB-first qubit ordering.
- **⚛️ Quantum Algorithms:** Built-in implementations of Grover, Shor, Deutsch-Jozsa, Bernstein-Vazirani, and QPE.
- **🌊 Open Quantum Systems:** Solvers for the Lindblad master equation, quantum trajectories, and time evolution.
- **📈 Variational Methods:** Frameworks for VQE, QAOA, and SSVQE with custom ansatz support.
- **📊 Information Theory:** Tools for calculating von Neumann entropy, concurrence, negativity, and mutual information.
- **🛡️ Error Correction:** Implementations of Bit-Flip, Phase-Flip, Shor, and Steane codes.

---

## 🚀 Quick Example

!!! tip "Create a Bell State in Seconds"
    Create and measure a maximally entangled Bell state with just a few lines of intuitive, Pythonic code.

    ```python
    from psiqit.circuits import QuantumCircuit

    # 1. Initialize a 2-qubit circuit
    circ = QuantumCircuit(2)

    # 2. Apply Hadamard and CNOT gates (Method chaining supported!)
    circ.h(0).cx(0, 1)

    # 3. Run the simulation and measure
    state = circ.run()
    print("State Vector:\n", state.data)

    result = circ.measure(shots=1024)
    print("\nMeasurement Counts:\n", result['counts'])
    ```

---

## 📚 Where to Go Next?

Choose your path based on your experience level:

- 🛠️ **[Installation Guide](getting-started/installation.md)**: Set up PSIQIT on your machine.
- ⚡ **[Quick Start](getting-started/quick-start.md)**: Run your first quantum circuit in 5 minutes.
- 🎓 **[Tutorials](tutorials/index.md)**: Step-by-step guides for common quantum computing tasks.
- 📖 **[API Reference](api/index.md)**: Detailed documentation for all modules and classes.
- 🔬 **[Advanced Use Cases](examples/index.md)**: Real-world research examples (e.g., Quantum Heat-Exchange).

---

<small>
💡 **Want to contribute?** Please read our [Contributing Guidelines](contributing.md) to get started.
</small>