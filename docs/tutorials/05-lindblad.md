
# Tutorial 5: Open Quantum Systems with Lindblad Dynamics

In this tutorial, you'll learn how to simulate **open quantum systems** using the Lindblad master equation. This is essential for modeling real-world quantum systems that interact with their environment, leading to decoherence and dissipation. :)

## What is the Lindblad Master Equation?

Unlike closed quantum systems that evolve unitarily via the Schrödinger equation, open systems exchange energy and information with their environment. Their evolution is described by the Lindblad master equation for the density matrix $\rho$:

`dρ/dt = -i/ħ [H, ρ] + Σ_k γ_k (L_k ρ L_k† - 1/2 {L_k†L_k, ρ})`

Where:
- $H$ is the system Hamiltonian.
- $L_k$ are the collapse (or jump) operators representing environmental interactions.
- $\gamma_k$ are the corresponding decay rates.

## Prerequisites

- PSIQIT installed (see [Installation Guide](../getting-started/installation.md))
- Understanding of density matrices and observables (see [Measurement & Observables](../concepts/measurements.md))

---

## Step-by-Step Implementation

### Step 1: Import Required Modules

```python
import numpy as np
from psiqit.quantum import pauli_z, pauli_x
from psiqit.dynamics import LindbladSolver
```

### Step 2: Define the Problem

Let's simulate a single qubit in a magnetic field ($H = Z$) that undergoes spontaneous emission or dephasing due to environmental noise, modeled by a collapse operator $L = X$ with a decay rate $\gamma = 0.1$.

!!! note "Density Matrices"
    Open quantum systems **must** be represented by density matrices (2D arrays), not state vectors (1D arrays), because they can become mixed states due to environmental interaction.

```python
# Define Hamiltonian
H = pauli_z()

# Define collapse operators and their rates
L = [pauli_x()]
gamma = [0.1]

# Initial state: |0⟩⟨0| as a density matrix
rho0 = np.array([[1, 0], 
                 [0, 0]], dtype=complex)

print("Initial state (density matrix):")
print(rho0)
```

### Step 3: Initialize the Solver

Create an instance of the `LindbladSolver` with the defined Hamiltonian and collapse operators.

```python
solver = LindbladSolver(H, L, gamma=gamma)
print("✓ Solver created successfully")
```

### Step 4: Run the Evolution

Use the `evolve` method to simulate the system's dynamics over a specified time span.

```python
t_max = 10.0  # Total simulation time
dt = 0.1      # Time step

result = solver.evolve(rho0, t_max=t_max, dt=dt, method='euler')

print(f"Time span: 0 to {t_max}")
print(f"Time steps saved: {len(result.times)}")
```

### Step 5: Analyze the Results

Extract the populations (diagonal elements of the density matrix) to observe how the state evolves over time.

```python
times = result.times
states = result.states

# Extract populations P(|0⟩) and P(|1⟩)
populations_0 = [np.real(rho[0, 0]) for rho in states]
populations_1 = [np.real(rho[1, 1]) for rho in states]

print(f"Initial population |0⟩: {populations_0[0]:.4f}")
print(f"Final population |0⟩:   {populations_0[-1]:.4f}")
print(f"Final population |1⟩:   {populations_1[-1]:.4f}")
```

**Expected Output:**
```text
Initial population |0⟩: 1.0000
Final population |0⟩:   0.5663
Final population |1⟩:   0.4337
```

### Step 6: Verify Physical Consistency

A valid quantum evolution must preserve the trace of the density matrix (Trace = 1) and keep all probabilities non-negative.

```python
final_rho = result.final_state
trace = np.trace(final_rho).real

print(f"Final trace: {trace:.6f}")

if abs(trace - 1.0) < 1e-5:
    print("✓ SUCCESS: Trace preservation verified.")
```

---

## Complete Code

Here is the complete, runnable script for this tutorial:

```python
"""
Tutorial 05: Lindblad Dynamics (Open Quantum Systems)
Complete runnable example using PSIQIT.
"""

import numpy as np
from psiqit.quantum import pauli_z, pauli_x
from psiqit.dynamics import LindbladSolver

print("=" * 70)
print("TUTORIAL 05: LINDBLAD DYNAMICS")
print("=" * 70)

# 1. Define the Problem
H = pauli_z()
L = [pauli_x()]
gamma = [0.1]
rho0 = np.array([[1, 0], [0, 0]], dtype=complex)

# 2. Create Solver
solver = LindbladSolver(H, L, gamma=gamma)

# 3. Solve Dynamics
t_max = 10.0
dt = 0.1
result = solver.evolve(rho0, t_max=t_max, dt=dt, method='euler')

# 4. Analyze Results
times = result.times
states = result.states

populations_0 = [np.real(rho[0, 0]) for rho in states]
populations_1 = [np.real(rho[1, 1]) for rho in states]

print(f"\nInitial P(|0⟩): {populations_0[0]:.4f}")
print(f"Final P(|0⟩):   {populations_0[-1]:.4f}")
print(f"Final P(|1⟩):   {populations_1[-1]:.4f}")

# 5. Verification
final_rho = result.final_state
trace = np.trace(final_rho).real
print(f"\nFinal trace: {trace:.6f}")

if abs(trace - 1.0) < 1e-5:
    print("✓ SUCCESS: Lindblad dynamics working correctly!")
else:
    print("✗ WARNING: Trace not preserved.")

print("=" * 70)
```

---

## Test It Yourself

You can find the complete test script in the repository:

```bash
python examples/tutorials/test_05_lindblad.py
```

---

## Common Mistakes

!!! warning "Using State Vectors Instead of Density Matrices"
    The `LindbladSolver` expects a 2D numpy array (density matrix) as the initial state `rho0`. Passing a 1D state vector will result in dimension mismatch errors.

!!! warning "Choosing Too Large Time Steps (dt)"
    If `dt` is too large, numerical integration methods like Euler may become unstable or violate trace preservation. For highly oscillatory systems, consider using smaller `dt` or advanced methods like `rk4` (if supported).

!!! warning "Non-Hermitian Collapse Operators"
    Ensure your collapse operators $L_k$ are physically meaningful. While the solver will run with any matrix, unphysical operators can lead to negative probabilities.

---

## Next Steps

Now that you can simulate open quantum systems, you are ready to explore:

- **[Tutorial 6: Quantum Error Correction](06-qec.md)**: Learn how to protect quantum information from the very decoherence we just simulated.
- **[Advanced Use Cases](../examples/index.md)**: Explore real-world research applications like Quantum Heat-Exchange with Feedback.
- **[API Reference](../api/dynamics.md)**: Complete documentation for `LindbladSolver` and other dynamical tools.

---

## References

- Breuer, H.P. & Petruccione, F. "The Theory of Open Quantum Systems" - Chapter 3
- [PSIQIT API Reference](../api/dynamics.md) - Complete documentation for `LindbladSolver`
