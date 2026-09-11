
# Advanced Use Case 2: NISQ Simulations & Noise Models

In this advanced example, you'll learn how to simulate **hardware noise** in quantum circuits. Current quantum computers operate in the **NISQ** (Noisy Intermediate-Scale Quantum) era, meaning noise and decoherence are inevitable. Understanding how to model and measure the impact of noise is crucial for developing robust quantum algorithms. :)

## Why Simulate Noise?

In an ideal world, quantum gates are perfect. In reality, every gate has a small probability of error. By simulating noise, we can:
1. **Benchmark Algorithms**: See how well an algorithm performs under realistic conditions.
2. **Optimize Circuits**: Find circuit layouts that are more resilient to specific types of noise.
3. **Develop Error Mitigation**: Test techniques to recover information from noisy results.

## Types of Noise

Common noise models include:
- **Bit-Flip Channel**: Probabilistically applies an $X$ gate (flips $|0\rangle \leftrightarrow |1\rangle$).
- **Phase-Flip Channel**: Probabilistically applies a $Z$ gate (flips the phase).
- **Depolarizing Channel**: Replaces the state with the completely mixed state with some probability.

In this example, we will simulate a **Bit-Flip Channel** and measure its impact on a Bell state using **Fidelity**.

---

## Step-by-Step Implementation

### Step 1: Define the Ideal Target State

We start by creating a perfect, noise-free Bell state $|\Phi^+\rangle = (|00\rangle + |11\rangle) / \sqrt{2}$.

```python
import numpy as np
from psiqit.circuits import QuantumCircuit

# Create an ideal Bell state
ideal_circ = QuantumCircuit(2)
ideal_circ.h(0)
ideal_circ.cx(0, 1)

ideal_state = ideal_circ.run()
ideal_data = ideal_state.data
```

### Step 2: Define a Noise Model

We define a function that simulates a bit-flip error on a specific qubit. For demonstration purposes, we first apply it deterministically to see the worst-case scenario.

```python
def apply_bit_flip_noise(circ, qubit):
    """Simulates a bit-flip noise channel by applying an X gate."""
    noisy_circ = QuantumCircuit(2)
    # Copy the original circuit operations
    noisy_circ.h(0)
    noisy_circ.cx(0, 1)
    
    # Apply the error
    noisy_circ.x(qubit)
    return noisy_circ
```

### Step 3: Simulate Noisy Evolution and Calculate Fidelity

We apply the error to Qubit 1 and calculate the **Fidelity** between the ideal state and the noisy state. Fidelity measures how "close" two quantum states are (1.0 = identical, 0.0 = orthogonal).

```python
# Apply error to qubit 1
noisy_circ = apply_bit_flip_noise(ideal_circ, qubit=1)
noisy_state = noisy_circ.run()
noisy_data = noisy_state.data

# Fidelity between pure states |ψ⟩ and |φ⟩ is |⟨ψ|φ⟩|²
fidelity = np.abs(np.vdot(ideal_data, noisy_data))**2

print(f"Fidelity (Ideal vs Noisy): {fidelity:.6f}")
```

!!! note "Why is Fidelity ~0.0?"
    Applying an $X$ gate to Qubit 1 of $|\Phi^+\rangle$ transforms it into $|\Psi^+\rangle = (|01\rangle + |10\rangle) / \sqrt{2}$. These two Bell states are orthogonal, so their inner product (and thus fidelity) is exactly 0. This demonstrates how a single error can completely destroy the intended quantum correlation.

### Step 4: Monte Carlo Noise Simulation

In reality, noise is probabilistic. To simulate a true noise channel (e.g., 20% error probability), we can use a **Monte Carlo** approach: running the circuit many times and randomly injecting errors.

```python
error_probability = 0.2
shots = 1000
error_count = 0

for _ in range(shots):
    circ_mc = QuantumCircuit(2)
    circ_mc.h(0)
    circ_mc.cx(0, 1)
    
    # Probabilistically apply X error to qubit 1
    if np.random.rand() < error_probability:
        circ_mc.x(1)
        error_count += 1
        
    circ_mc.measure(shots=1) # Execute the circuit

print(f"Errors injected: {error_count} times ({error_count/shots*100:.1f}%)")
```

---

## Complete Code

Here is the complete, runnable script:

```python
"""
Advanced Use Case 2: NISQ Noise Simulation
Demonstrates simulating hardware noise and measuring its impact on fidelity.
"""

import numpy as np
from psiqit.circuits import QuantumCircuit

print("=" * 70)
print("ADVANCED USE CASE 2: NISQ Noise Simulation")
print("=" * 70)

# 1. Ideal Target State
ideal_circ = QuantumCircuit(2)
ideal_circ.h(0)
ideal_circ.cx(0, 1)
ideal_data = ideal_circ.run().data

# 2. Define Noise Model
def apply_bit_flip_noise(qubit):
    noisy_circ = QuantumCircuit(2)
    noisy_circ.h(0)
    noisy_circ.cx(0, 1)
    noisy_circ.x(qubit)
    return noisy_circ

# 3. Simulate Noisy Evolution & Fidelity
noisy_data = apply_bit_flip_noise(qubit=1).run().data
fidelity = np.abs(np.vdot(ideal_data, noisy_data))**2

print(f"\n1. Fidelity (Ideal vs Noisy): {fidelity:.6f}")
if fidelity < 0.01:
    print("   ✓ Noise correctly degraded the state fidelity to ~0.0")

# 4. Monte Carlo Simulation
error_probability = 0.2
shots = 1000
error_count = 0

for _ in range(shots):
    circ_mc = QuantumCircuit(2)
    circ_mc.h(0)
    circ_mc.cx(0, 1)
    if np.random.rand() < error_probability:
        circ_mc.x(1)
        error_count += 1
    circ_mc.measure(shots=1)

print(f"\n2. Monte Carlo Simulation ({shots} runs, {error_probability*100:.1f}% error prob):")
print(f"   Errors injected: {error_count} times ({error_count/shots*100:.1f}%)")
print("   ✓ Monte Carlo simulation completed successfully.")

print("=" * 70)
```

---

## Test It Yourself

You can find the complete test script in the repository:

```bash
python examples/advanced/test_nisq_noise.py
```

---

## Understanding the Results

- **Deterministic Fidelity Drop**: When we force an error, the fidelity drops to 0.0 because the resulting state is orthogonal to the target state. This is the "worst-case" scenario.
- **Monte Carlo Accuracy**: The Monte Carlo simulation shows that over many runs, the empirical error rate converges to the defined probability (e.g., ~20%). This is how real quantum hardware characterization works: by running circuits thousands of times and analyzing the statistical distribution of errors.

---

## Next Steps

Now that you can simulate noise, you are ready to explore:

- **[Advanced Use Case 3: Quantum Chemistry with VQE](03-vqe-molecule.md)**: Simulate real molecules and see how noise affects chemical energy estimates.
- **[Error Mitigation Techniques](../tutorials/06-qec.md)**: Learn how to recover from the noise we just simulated.
- **[API Reference](../api/index.md)**: Dive deep into PSIQIT's core modules for advanced noise modeling.

---

## References

- Preskill, J. "Quantum Computing in the NISQ era and beyond" (2018)
- Nielsen, M.A. & Chuang, I.L. "Quantum Computation and Quantum Information" - Chapter 8 (Quantum Noise)
