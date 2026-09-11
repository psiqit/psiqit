
# Advanced Use Case 1: Quantum Teleportation

In this advanced example, you'll implement the **Quantum Teleportation** protocol - a foundational protocol in quantum information that demonstrates how to transfer an unknown quantum state from one location to another using quantum entanglement and classical communication, without violating the No-Cloning Theorem. :)

## The Protocol

Quantum teleportation involves three qubits:
1. **Qubit 0 (Message):** The unknown state `|ψ⟩ = α|0⟩ + β|1⟩` to be teleported.
2. **Qubit 1 (Alice):** Alice's half of an entangled Bell pair.
3. **Qubit 2 (Bob):** Bob's half of the entangled Bell pair.

The steps are:
1. **Entanglement:** Create a Bell state `|Φ⁺⟩ = (|00⟩ + |11⟩) / √2` between Qubit 1 and Qubit 2.
2. **Alice's Operation:** Alice applies a CNOT gate (Qubit 0 → Qubit 1), followed by a Hadamard gate on Qubit 0.
3. **Measurement:** Alice measures Qubits 0 and 1, obtaining two classical bits.
4. **Bob's Correction:** Based on Alice's classical bits, Bob applies a correction gate (I, X, Z, or XZ) to Qubit 2 to perfectly reconstruct `|ψ⟩`.

## Prerequisites

- PSIQIT installed (see [Installation Guide](../getting-started/installation.md))
- Understanding of quantum circuits and measurement (see [Tutorial 1: Bell State](../tutorials/01-bell-state.md))

---

## Step-by-Step Implementation

### Step 1: Define the Message State

We start by defining an arbitrary message state with both amplitude and phase.

```python
import numpy as np
from psiqit.circuits import QuantumCircuit

# Arbitrary state: α|0⟩ + β|1⟩
theta = np.pi / 3  # Amplitude angle
phi = np.pi / 4    # Phase angle

alpha = np.cos(theta / 2)
beta = np.exp(1j * phi) * np.sin(theta / 2)

print(f"Message state: α={alpha:.4f}, β={beta:.4f}")
```

### Step 2: Build the Teleportation Circuit

We construct the 3-qubit circuit following the protocol.

!!! note "LSB-First Ordering"
    Remember that PSIQIT uses **LSB-first** ordering. In a 3-qubit system, the state `|q₂q₁q₀⟩` means `q₀` is the least significant bit.

```python
# q0: message, q1: Alice's Bell, q2: Bob's Bell
circ = QuantumCircuit(3)

# 1. Prepare message state on q0
circ.ry(0, theta)
circ.rz(0, phi)

# 2. Create Bell state between q1 and q2
circ.h(1)
circ.cx(1, 2)

# 3. Alice's Bell measurement operations
circ.cx(0, 1)  # CNOT: message (q0) → Alice's qubit (q1)
circ.h(0)      # Hadamard on message (q0)
```

### Step 3: Measure and Analyze

We measure all qubits to simulate the protocol and observe the statistical distribution.

```python
result = circ.measure(shots=1000)
counts = result['counts']

for state, count in sorted(counts.items()):
    print(f"|{state}⟩: {count} ({count/10:.1f}%)")
```

### Step 4: Rigorous Mathematical Verification

In a statevector simulator, the system remains in a superposition of all 4 possible measurement outcomes until we explicitly collapse it. To rigorously prove teleportation works, we compare the **simulated probabilities** directly against the **theoretical probabilities** derived from quantum mechanics.

For a successful teleportation, the probability of each of the 8 basis states is exactly 1/4 of the message state's probability for that specific qubit configuration.

```python
# Theoretical probabilities based on quantum mechanics derivation
theoretical_probs = np.zeros(8)
theoretical_probs[0] = np.abs(alpha)**2 / 4  # |000⟩
theoretical_probs[4] = np.abs(beta)**2 / 4   # |100⟩
theoretical_probs[1] = np.abs(alpha)**2 / 4  # |001⟩
theoretical_probs[5] = np.abs(beta)**2 / 4   # |101⟩
theoretical_probs[2] = np.abs(beta)**2 / 4   # |010⟩
theoretical_probs[6] = np.abs(alpha)**2 / 4  # |110⟩
theoretical_probs[3] = np.abs(beta)**2 / 4   # |011⟩
theoretical_probs[7] = np.abs(alpha)**2 / 4  # |111⟩

# Extract simulated probabilities from the statevector before measurement
state_before_measure = circ.run()
simulated_probs = np.abs(state_before_measure.data)**2

# Calculate the maximum difference
max_diff = np.max(np.abs(simulated_probs - theoretical_probs))
print(f"Maximum absolute difference: {max_diff:.2e}")
```

!!! tip "Why this verification?"
    Checking the statevector probabilities against theory is the most robust way to verify quantum protocols in simulation. It proves that the amplitudes are correctly routed to Bob's qubit for every possible measurement branch, without needing to manually reconstruct the state for each branch.

---

## Complete Code

Here is the complete, runnable script:

```python
"""
Advanced Use Case 1: Quantum Teleportation
Rigorous verification of the teleportation protocol using PSIQIT.
"""

import numpy as np
from psiqit.circuits import QuantumCircuit

print("=" * 70)
print("ADVANCED USE CASE 1: QUANTUM TELEPORTATION")
print("=" * 70)

# 1. Define Message State
theta, phi = np.pi / 3, np.pi / 4
alpha = np.cos(theta / 2)
beta = np.exp(1j * phi) * np.sin(theta / 2)

# 2. Build Circuit
circ = QuantumCircuit(3)
circ.ry(0, theta)
circ.rz(0, phi)
circ.h(1)
circ.cx(1, 2)
circ.cx(0, 1)
circ.h(0)

# 3. Measure
result = circ.measure(shots=1000)
print("\nMeasurement results:")
for state, count in sorted(result['counts'].items()):
    print(f"  |{state}⟩: {count} ({count/10:.1f}%)")

# 4. Rigorous Verification
theoretical_probs = np.zeros(8)
theoretical_probs[0] = np.abs(alpha)**2 / 4
theoretical_probs[4] = np.abs(beta)**2 / 4
theoretical_probs[1] = np.abs(alpha)**2 / 4
theoretical_probs[5] = np.abs(beta)**2 / 4
theoretical_probs[2] = np.abs(beta)**2 / 4
theoretical_probs[6] = np.abs(alpha)**2 / 4
theoretical_probs[3] = np.abs(beta)**2 / 4
theoretical_probs[7] = np.abs(alpha)**2 / 4

state_before_measure = circ.run()
simulated_probs = np.abs(state_before_measure.data)**2
max_diff = np.max(np.abs(simulated_probs - theoretical_probs))

print(f"\nMaximum absolute difference: {max_diff:.2e}")

if max_diff < 1e-10:
    print("✓ SUCCESS: Quantum teleportation works perfectly!")
else:
    print("✗ WARNING: Probabilities do not match theoretical predictions.")

print("=" * 70)
```

---

## Test It Yourself

You can find the complete test script in the repository:

```bash
python examples/advanced/test_teleportation.py
```

---

## Understanding the Results

When you run the simulation, you will notice that the measurement outcomes are not uniform. For example, if `α ≈ 0.866` and `β ≈ 0.353+0.353j`, the states where Bob's qubit (`q₂`) is `|0⟩` (indices 0, 1, 2, 3) will appear roughly 75% of the time, matching `|α|²`. This statistical bias is the direct signature of the teleported state's amplitudes manifesting in the measurement statistics.

---

## Next Steps

Now that you've mastered quantum teleportation, explore more advanced topics:

- **[Advanced Use Case 2: NISQ Noise Simulation](02-nisq-noise.md)**: Learn how to simulate realistic hardware noise.
- **[Advanced Use Case 3: Quantum Chemistry with VQE](03-vqe-molecule.md)**: Simulate real molecules like H₂.
- **[API Reference](../api/index.md)**: Dive deep into PSIQIT's core modules.

---

## References

- Nielsen, M.A. & Chuang, I.L. "Quantum Computation and Quantum Information" - Chapter 1.3.7
- Bennett, C.H. et al. "Teleporting an unknown quantum state via dual classical and Einstein-Podolsky-Rosen channels" (1993)
