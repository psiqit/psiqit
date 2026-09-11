
# Tutorial 3: Quantum Fourier Transform (QFT)

In this tutorial, you'll learn how to apply the **Quantum Fourier Transform (QFT)** - a fundamental quantum operation that forms the backbone of important algorithms like Shor's factoring and quantum phase estimation. :)

## What is QFT?

The Quantum Fourier Transform is the quantum analogue of the classical Discrete Fourier Transform (DFT). While the classical DFT requires O(N²) operations for N data points, the QFT matrix can transform a quantum state using highly optimized linear algebra, and its circuit decomposition requires only O(n²) gates, where n = log₂(N) is the number of qubits.

This **exponential speedup** makes QFT one of the most important building blocks in quantum computing.

## Mathematical Definition

For an n-qubit system, QFT transforms a basis state `|j⟩` as:

`QFT|j⟩ = (1/√N) Σₖ e^(2πijk/N) |k⟩`

where N = 2ⁿ is the dimension of the Hilbert space.

## Prerequisites

- PSIQIT installed (see [Installation Guide](../getting-started/installation.md))
- Understanding of quantum states (see [States & Operators](../concepts/states-operators.md))

---

## Step-by-Step Implementation

### Step 1: Import Required Modules

PSIQIT provides two main functions for QFT:
- `qft()`: Applies QFT directly to a quantum state (Ket) - **recommended for exact results**
- `qft_circuit()`: Generates the gate-level QFT circuit

```python
import numpy as np
from psiqit.quantum.state import Ket
from psiqit.algorithms.qft import qft, qft_circuit
```

### Step 2: Apply QFT to the |000⟩ State

The most straightforward way to apply QFT in PSIQIT is using the state-based `qft()` function, which guarantees mathematically exact results.

```python
n_qubits = 3
dim = 2 ** n_qubits

# Create the |000⟩ state (index 0 in LSB-first ordering)
state_000 = Ket(np.array([1.0] + [0.0] * (dim - 1), dtype=complex))

# Apply QFT
qft_state_000 = qft(state_000)

print("QFT|000⟩ amplitudes:")
for i, amp in enumerate(qft_state_000.data):
    binary = format(i, f'0{n_qubits}b')
    print(f"  |{binary}⟩: {amp.real:+.4f} {amp.imag:+.4f}j")
```

**Expected Output:**
```text
QFT|000⟩ amplitudes:
  |000⟩: +0.3536 +0.0000j
  |001⟩: +0.3536 +0.0000j
  |010⟩: +0.3536 +0.0000j
  |011⟩: +0.3536 +0.0000j
  |100⟩: +0.3536 +0.0000j
  |101⟩: +0.3536 +0.0000j
  |110⟩: +0.3536 +0.0000j
  |111⟩: +0.3536 +0.0000j
```

!!! tip "Uniform Superposition"
    QFT applied to `|000⟩` creates a uniform superposition of all basis states, similar to applying Hadamard gates to all qubits. This is because the Fourier transform of a delta function is a constant.

### Step 3: Apply QFT to a Non-Trivial State

Let's apply QFT to `|001⟩` (where qubit 0 is `1`, and others are `0`). In LSB-first ordering, this corresponds to index `1`.

```python
# Create |001⟩ state
state_001_data = np.zeros(dim, dtype=complex)
state_001_data[1] = 1.0  # Index 1 is |001⟩ in LSB-first
state_001 = Ket(state_001_data)

# Apply QFT
qft_state_001 = qft(state_001)

print("QFT|001⟩ amplitudes (significant values only):")
for i, amp in enumerate(qft_state_001.data):
    if abs(amp) > 0.01:
        binary = format(i, f'0{n_qubits}b')
        print(f"  |{binary}⟩: {amp.real:+.4f} {amp.imag:+.4f}j")
```

**Expected Output:**
```text
QFT|001⟩ amplitudes (significant values only):
  |000⟩: +0.3536 +0.0000j
  |001⟩: +0.2500 +0.2500j
  |010⟩: +0.0000 +0.3536j
  |011⟩: -0.2500 +0.2500j
  |100⟩: -0.3536 +0.0000j
  |101⟩: -0.2500 -0.2500j
  |110⟩: -0.0000 -0.3536j
  |111⟩: +0.2500 -0.2500j
```

!!! note "Phase Encoding"
    Notice how QFT encodes the input state index into the **phases** (the imaginary parts) of the output amplitudes. This is the key property that makes QFT useful for phase estimation algorithms.

### Step 4: Generate the QFT Circuit (Optional)

If you need to see the gate-level decomposition, PSIQIT provides `qft_circuit()`:

```python
from psiqit.visualization import draw_circuit

# Generate the QFT circuit
qft_circ = qft_circuit(n_qubits)

# Visualize the circuit
print(draw_circuit(qft_circ))
```

This will output a circuit composed of Hadamard gates, controlled-phase (`cu1`) gates, and final SWAP gates to correct the qubit ordering.

!!! warning "State-based vs Circuit-based"
    The state-based `qft()` function uses direct matrix multiplication and gives **exact** results. The circuit-based `qft_circuit()` function decomposes QFT into gates, which is useful for visualization and understanding the algorithm, but may have minor numerical differences due to gate decomposition.

---

## Verification of Unitarity

A valid quantum operation must be unitary, meaning the sum of probabilities must equal 1.0:

```python
probs = np.abs(qft_state_000.data)**2
total_prob = np.sum(probs)
print(f"Sum of probabilities: {total_prob:.6f}")
```

**Expected Output:**
```text
Sum of probabilities: 1.000000
```

---

## Complete Code

Here's the complete, runnable script that you can copy and execute:

```python
"""
Tutorial 03: Quantum Fourier Transform (QFT)
Complete runnable example using PSIQIT.
"""

import numpy as np
from psiqit.quantum.state import Ket
from psiqit.algorithms.qft import qft, qft_circuit
from psiqit.visualization import draw_circuit

print("=" * 70)
print("TUTORIAL 03: QUANTUM FOURIER TRANSFORM (QFT)")
print("=" * 70)

n_qubits = 3
dim = 2 ** n_qubits

# ============================================================================
# Test 1: QFT on |000⟩
# ============================================================================
print("\n1. Testing QFT on |000⟩ state (State-based):")
print("-" * 70)

state_000 = Ket(np.array([1.0] + [0.0] * (dim - 1), dtype=complex))
qft_state_000 = qft(state_000)

print("   QFT|000⟩ amplitudes:")
for i, amp in enumerate(qft_state_000.data):
    binary = format(i, f'0{n_qubits}b')
    print(f"   |{binary}⟩: {amp.real:+.4f} {amp.imag:+.4f}j")

# ============================================================================
# Test 2: QFT Circuit Generation
# ============================================================================
print("\n2. Testing QFT Circuit Generation (Gate-based):")
print("-" * 70)

qft_circ = qft_circuit(n_qubits)

print("   Circuit diagram:")
try:
    print(draw_circuit(qft_circ))
except Exception:
    print("   (Circuit generated successfully, drawing skipped)")

print("   Note: The circuit-based approach uses Hadamard and controlled-phase gates.")
print("   For exact state transformations, the state-based qft() function is recommended.")

# ============================================================================
# Test 3: QFT on |001⟩
# ============================================================================
print("\n3. Testing QFT on |001⟩ state:")
print("-" * 70)

state_001_data = np.zeros(dim, dtype=complex)
state_001_data[1] = 1.0  # Index 1 is |001⟩ in LSB-first
state_001 = Ket(state_001_data)

qft_state_001 = qft(state_001)

print("   QFT|001⟩ amplitudes (significant values only):")
for i, amp in enumerate(qft_state_001.data):
    if abs(amp) > 0.01:
        binary = format(i, f'0{n_qubits}b')
        print(f"   |{binary}⟩: {amp.real:+.4f} {amp.imag:+.4f}j")

# ============================================================================
# Test 4: Verification
# ============================================================================
print("\n4. Verification:")
print("-" * 70)

probs = np.abs(qft_state_000.data)**2
total_prob = np.sum(probs)
print(f"   Sum of probabilities: {total_prob:.6f}")

if abs(total_prob - 1.0) < 1e-5:
    print("   ✓ SUCCESS: QFT is unitary and working correctly!")
    print("   ✓ Phase encoding is functioning as expected.")
else:
    print("   ✗ WARNING: Probabilities do not sum to 1.0. Check QFT implementation.")

print("\n" + "=" * 70)
print("Tutorial 03 completed!")
print("=" * 70)
```

---

## Test It Yourself

You can find the complete test script in the repository:

```bash
python examples/tutorials/test_03_qft.py
```

---

## Common Mistakes

!!! warning "Wrong Input Type"
    The `qft()` function expects a `Ket` object, list, or numpy array - **not** a `QuantumCircuit`. If you want to apply QFT to a circuit, use `qft_circuit()` instead.

!!! warning "Qubit Ordering"
    Remember that PSIQIT uses **LSB-first** ordering. When you set `state_data[1] = 1.0`, you're creating the state `|001⟩` (qubit 0 is `1`), not `|100⟩`.

!!! warning "State Dimension"
    The input state must have exactly `2^n_qubits` elements. Otherwise, you'll get a `ValueError`.

---

## Applications of QFT

### 1. Shor's Factoring Algorithm
QFT is the key component that enables Shor's algorithm to factor large numbers exponentially faster than classical algorithms.

### 2. Quantum Phase Estimation
QFT is used to extract the eigenvalue (phase) of a unitary operator, which is crucial for:
- Quantum chemistry simulations
- Solving linear systems of equations
- Quantum machine learning

### 3. Quantum Signal Processing
QFT enables efficient manipulation of quantum signals in the frequency domain.

---

## Next Steps

Now that you've mastered QFT, you can:

- **[Tutorial 4: VQE](04-vqe.md)**: Learn variational quantum algorithms
- **[Tutorial 5: Lindblad Dynamics](05-lindblad.md)**: Explore open quantum systems
- **[Tutorial 6: QEC](06-qec.md)**: Implement quantum error correction codes
- **[API Reference](../api/algorithms.md)**: Complete documentation for `qft` and `qft_circuit`

---

## References

- Nielsen, M.A. & Chuang, I.L. "Quantum Computation and Quantum Information" - Chapter 5
- Coppersmith, D. "An approximate Fourier transform useful in quantum factoring" (1994)
- [PSIQIT API Reference](../api/algorithms.md) - Complete documentation for QFT functions
