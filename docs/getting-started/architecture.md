
# Architecture & Design Philosophy

This document explains the core design decisions behind PSIQIT, particularly the **LSB-first qubit ordering** that distinguishes it from other quantum computing frameworks. :)

## Qubit Ordering: LSB-First vs MSB-First

One of the most important architectural decisions in PSIQIT is the use of **LSB-first** (Least Significant Bit first) qubit ordering. This affects how quantum states are represented and how gates are applied.

### What Does LSB-First Mean?

In PSIQIT, **qubit 0 is the least significant bit** in the binary representation of a quantum state.

For a 3-qubit system, the state vector indices map as follows:

| Index | Binary | Qubit State | Meaning |
| :--- | :--- | :--- | :--- |
| 0 | `000` | `|000⟩` | All qubits in `|0⟩` |
| 1 | `001` | `|001⟩` | Qubit 0 = 1, others = 0 |
| 2 | `010` | `|010⟩` | Qubit 1 = 1, others = 0 |
| 3 | `011` | `|011⟩` | Qubits 0 and 1 = 1 |
| 4 | `100` | `|100⟩` | Qubit 2 = 1, others = 0 |
| 5 | `101` | `|101⟩` | Qubits 0 and 2 = 1 |
| 6 | `110` | `|110⟩` | Qubits 1 and 2 = 1 |
| 7 | `111` | `|111⟩` | All qubits in `|1⟩` |

### Why LSB-First?

#### 1. Natural Tensor Product Structure

LSB-first ordering aligns naturally with the **Kronecker product** convention used in linear algebra. When constructing multi-qubit operators, the tensor product `A ⊗ B` places `A` on the left (more significant qubits) and `B` on the right (less significant qubits).

!!! note "Mathematical Consistency"
    In PSIQIT, applying an operator to qubit 0 and qubit 1 corresponds to `np.kron(Operator_Q1, Operator_Q0)`. This makes manual mathematical derivations intuitive and reduces errors.

#### 2. Consistent with Bitwise Operations

PSIQIT's internal implementation uses **bitwise operations** (shifts and XOR) to apply gates efficiently. LSB-first ordering means:

- Qubit `i` corresponds to bit position `i` in the state index.
- Checking if qubit `i` is `|1⟩`: `(state_index >> i) & 1`
- Flipping qubit `i`: `state_index ^ (1 << i)`

This leads to clean, highly optimized C-like performance within Python.

---

## Core Components Overview

### 1. Quantum States (`psiqit.quantum`)

Quantum states are represented by the `Ket` class, which wraps a NumPy complex array:

```python
from psiqit.quantum import zero, one, plus

q0 = zero()  # |0⟩
q1 = one()   # |1⟩
q_plus = plus()  # |+⟩ = (|0⟩ + |1⟩) / √2
```

### 2. Quantum Circuits (`psiqit.circuits`)

The `QuantumCircuit` class provides a high-level, chainable interface for building and executing quantum circuits:

```python
from psiqit.circuits import QuantumCircuit

circ = QuantumCircuit(3)
circ.h(0).cx(0, 1).cx(1, 2)  # Creates a 3-qubit GHZ state

state = circ.run()  # Execute and return final state vector
```

### 3. Operators and Gates (`psiqit.quantum.operator`)

Gates are represented as unitary matrices, allowing for both circuit simulation and direct matrix algebra:

```python
from psiqit.quantum import hadamard, pauli_x, cnot

H = hadamard()  # 2x2 Hadamard matrix
X = pauli_x()   # 2x2 Pauli-X matrix
CNOT = cnot()   # 4x4 CNOT matrix
```

---

## Design Principles

### 1. Simplicity Over Flexibility

PSIQIT prioritizes a clean, intuitive API over supporting every possible quantum computing paradigm. The goal is to make quantum computing accessible without sacrificing mathematical correctness.

### 2. Numerical Stability

All algorithms and solvers are implemented with numerical stability in mind:

- **RK4 integration** for the Lindblad master equation.
- **Automatic normalization** of quantum states after gate application.
- **Careful handling of edge cases** (e.g., division by zero, singular matrices in SVD).

### 3. Educational and Research Value

PSIQIT is designed to be both a research tool and an educational resource. The codebase emphasizes:

- Clear documentation with runnable examples.
- Readable implementations that expose the underlying physics.
- Consistent naming conventions that match standard quantum computing terminology (e.g., Nielsen and Chuang).

---

## Performance Considerations

### State Vector Simulation

PSIQIT uses **exact state vector simulation**, which represents the quantum state as a complex vector of size 2ⁿ for n qubits. This provides:

- **Exact results** (no sampling noise, unless `measure()` is explicitly called).
- **Full state information** (amplitude and phase).
- **Exponential memory scaling** (O(2ⁿ)).

!!! tip "Scalability"
    For systems with more than ~20-25 qubits, consider using tensor network methods or running circuits on actual quantum hardware via future PSIQIT backends.

---

## Next Steps

Now that you understand the core architecture, you are ready to dive deeper:

- **[Core Concepts](../concepts/states-operators.md)**: Deep dive into how states and operators work.
- **[Tutorials](../tutorials/index.md)**: Step-by-step guides for specific algorithms.
- **[API Reference](../api/index.md)**: Detailed documentation for developers.
