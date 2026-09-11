
# Tutorial 4: Variational Quantum Eigensolver (VQE)

In this tutorial, you'll implement the **Variational Quantum Eigensolver (VQE)** - one of the most important hybrid quantum-classical algorithms, widely used for finding the ground state energy of molecules and materials. :)

## What is VQE?

VQE is a hybrid algorithm that uses a quantum computer to prepare a parameterized quantum state (the *ansatz*) and measure its energy, while a classical computer optimizes those parameters to minimize the energy. 

According to the variational principle, the expectation value of the Hamiltonian $H$ for any trial state $|\psi(\theta)\rangle$ is always greater than or equal to the true ground state energy $E_0$:

`⟨ψ(θ)|H|ψ(θ)⟩ ≥ E₀`

By minimizing this expectation value, we can approximate the ground state.

## Prerequisites

- PSIQIT installed (see [Installation Guide](../getting-started/installation.md))
- Understanding of quantum circuits and parameterized gates (see [Building Circuits](../concepts/circuits.md))
- Basic knowledge of expectation values (see [Measurement & Observables](../concepts/measurements.md))

---

## Step-by-Step Implementation

### Step 1: Import Required Modules

We will use `scipy.optimize` for the classical optimization part.

```python
import numpy as np
from scipy.optimize import minimize
from psiqit.circuits import QuantumCircuit
from psiqit.quantum import pauli_z, expectation
```

### Step 2: Define the Problem

Let's find the ground state of a simple 1-qubit Hamiltonian: $H = Z$ (Pauli-Z).
We know analytically that the true ground state is $|1\rangle$ and the ground energy is $-1.0$.

```python
H = pauli_z()
print("Hamiltonian: H = Z")
print("True ground state: |1⟩")
print("True ground energy: -1.0")
```

### Step 3: Define the Ansatz

The ansatz is a parameterized quantum circuit. For a single qubit, a simple rotation gate is sufficient to explore the entire Bloch sphere.

```python
def create_ansatz(theta):
    """Create a parameterized quantum circuit."""
    circ = QuantumCircuit(1)
    circ.ry(0, theta)  # Rotate around Y-axis by theta
    return circ
```

### Step 4: Define the Cost Function

The cost function evaluates the energy of the state produced by the ansatz for a given set of parameters.

```python
def cost_function(theta):
    """Calculate the expectation value ⟨ψ(θ)|H|ψ(θ)⟩."""
    # scipy passes theta as an array, so we extract the first element
    circ = create_ansatz(theta[0])
    state = circ.run()
    
    # PSIQIT's expectation function handles the math cleanly
    energy = expectation(state, H)
    return float(energy)
```

### Step 5: Run the Classical Optimizer

We will use the BFGS algorithm from `scipy` to minimize the cost function.

!!! warning "The Zero-Gradient Trap"
    **Never initialize parameters exactly at 0.0** for this specific Hamiltonian/Ansatz combination. The energy is $E(\theta) = \cos(\theta)$, and its gradient is $-\sin(\theta)$. At $\theta = 0$, the gradient is exactly $0$, which will trick the optimizer into thinking it has already found a minimum (a local maximum, in this case). Always use a small non-zero value like `0.5` or random initialization.

```python
# Start with a small non-zero value to ensure a non-zero initial gradient
initial_theta = 0.5  

print(f"Initial energy: {cost_function([initial_theta]):.4f}")

result = minimize(
    cost_function,
    x0=[initial_theta],
    method='BFGS',
    options={'disp': False}
)

optimal_theta = result.x[0]
optimal_energy = result.fun

print(f"Optimization successful: {result.success}")
print(f"Optimal theta: {optimal_theta:.4f} rad")
print(f"Optimal energy: {optimal_energy:.4f}")
```

### Step 6: Verify the Result

Let's check if the optimized circuit actually produces the true ground state $|1\rangle$.

```python
final_circ = create_ansatz(optimal_theta)
final_state = final_circ.run()

print(f"Final state vector: {final_state.data}")

# Calculate fidelity with the true ground state |1⟩ = [0, 1]
target_state = np.array([0.0, 1.0], dtype=complex)
fidelity = np.abs(np.vdot(target_state, final_state.data))**2

print(f"Fidelity with true ground state |1⟩: {fidelity:.4f}")
```

**Expected Output:**
```text
Optimization successful: True
Optimal theta: 3.1416 rad
Optimal energy: -1.0000
Final state vector: [0.+0.j 1.+0.j]
Fidelity with true ground state |1⟩: 1.0000
```

---

## Complete Code

Here's the complete, runnable script:

```python
"""
Tutorial 04: Variational Quantum Eigensolver (VQE)
Complete runnable example using PSIQIT.
"""

import numpy as np
from scipy.optimize import minimize
from psiqit.circuits import QuantumCircuit
from psiqit.quantum import pauli_z, expectation

print("=" * 70)
print("TUTORIAL 04: VARIATIONAL QUANTUM EIGENSOLVER (VQE)")
print("=" * 70)

# 1. Define the Problem
H = pauli_z()

# 2. Define the Ansatz
def create_ansatz(theta):
    circ = QuantumCircuit(1)
    circ.ry(0, theta)
    return circ

# 3. Define the Cost Function
def cost_function(theta):
    circ = create_ansatz(theta[0])
    state = circ.run()
    energy = expectation(state, H)
    return float(energy)

# 4. Run the Classical Optimizer
print("\nRunning Classical Optimizer (BFGS):")
initial_theta = 0.5  # Crucial: Avoid zero gradient at theta=0
print(f"Initial theta: {initial_theta:.4f} rad")
print(f"Initial energy: {cost_function([initial_theta]):.4f}")

result = minimize(cost_function, x0=[initial_theta], method='BFGS', options={'disp': False})

print(f"\nOptimization successful: {result.success}")
print(f"Optimal theta: {result.x[0]:.4f} rad")
print(f"Optimal energy: {result.fun:.4f}")

# 5. Verify the Result
print("\nVerification:")
final_state = create_ansatz(result.x[0]).run()
target_state = np.array([0.0, 1.0], dtype=complex)
fidelity = np.abs(np.vdot(target_state, final_state.data))**2

print(f"Final state vector: {final_state.data}")
print(f"Fidelity with |1⟩: {fidelity:.4f}")

if abs(result.fun - (-1.0)) < 1e-4 and fidelity > 0.99:
    print("\n✓ SUCCESS: VQE found the correct ground state and energy!")
else:
    print("\n✗ WARNING: VQE did not converge to the expected ground state.")

print("=" * 70)
```

---

## Test It Yourself

You can find the complete test script in the repository:

```bash
python examples/tutorials/test_04_vqe.py
```

---

## Common Mistakes

!!! warning "Initializing at Symmetry Points"
    As demonstrated, initializing parameters at exactly `0.0` or `π` can result in zero gradients, causing the classical optimizer to stall immediately. Use small random values or a fixed offset like `0.5`.

!!! warning "Ignoring the Real Part"
    Expectation values of Hermitian operators (observables) are always real numbers. However, due to floating-point arithmetic, the result might have a tiny imaginary part (e.g., `1e-16j`). PSIQIT's `expectation` function automatically extracts the `.real` part for you, but if you calculate it manually, remember to do so.

---

## Next Steps

Now that you've mastered VQE, you can explore more advanced topics:

- **[Tutorial 5: Lindblad Dynamics](05-lindblad.md)**: Simulate open quantum systems and decoherence.
- **[Tutorial 6: Quantum Error Correction](06-qec.md)**: Protect quantum information from noise.
- **[API Reference](../api/variational.md)**: Explore PSIQIT's built-in VQE and QAOA modules for more complex problems.

---

## References

- Peruzzo, A. et al. "A variational eigenvalue solver on a photonic quantum processor" (2014)
- McClean, J. R. et7. "The theory of variational hybrid quantum-classical algorithms" (2016)
- [PSIQIT API Reference](../api/quantum.md) - Complete documentation for `expectation` and quantum states.
