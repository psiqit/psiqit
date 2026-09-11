
# Advanced Use Case 3: Quantum Chemistry with VQE (H₂ Molecule)

In this advanced example, you'll simulate a real molecule — **Molecular Hydrogen (H₂)** — using the **Variational Quantum Eigensolver (VQE)**. This is one of the most promising near-term applications of quantum computing, with potential breakthroughs in materials science, drug discovery, and energy research. :)

## The Problem

Finding the ground state energy of a molecule is fundamentally a quantum mechanical problem. Classically, the computational cost grows **exponentially** with the number of electrons. Quantum computers, by naturally representing quantum states, can potentially solve this problem much more efficiently.

For H₂ in the minimal **STO-3G basis**, the electronic structure can be mapped to a **2-qubit system** using transformations like Jordan-Wigner or Bravyi-Kitaev.

## The Hamiltonian

The molecular Hamiltonian for H₂ (at equilibrium bond length) can be expressed as a sum of Pauli operators:

`H = g₀·I + g₁·Z₀ + g₂·Z₁ + g₃·Z₀Z₁ + g₄·X₀X₁ + g₅·Y₀Y₁`

Where the coefficients (in Hartree units) are approximately:
- `g₀ = -1.0523` (constant energy offset)
- `g₁ = 0.3979`, `g₂ = -0.3979` (single-qubit Z terms)
- `g₃ = -0.0112` (two-qubit ZZ interaction)
- `g₄ = g₅ = 0.1809` (XX and YY coupling terms)

## Prerequisites

- PSIQIT installed (see [Installation Guide](../getting-started/installation.md))
- Understanding of VQE (see [Tutorial 4: VQE](../tutorials/04-vqe.md))
- Basic quantum chemistry knowledge

---

## Step-by-Step Implementation

### Step 1: Build the Molecular Hamiltonian

We construct the 4×4 Hamiltonian matrix by combining Pauli operators using the Kronecker product.

```python
import numpy as np
from scipy.optimize import minimize
from psiqit.circuits import QuantumCircuit
from psiqit.quantum import pauli_z, pauli_x, pauli_y, identity

# Hamiltonian coefficients for H2 (STO-3G, equilibrium geometry)
g0, g1, g2 = -1.0523, 0.3979, -0.3979
g3, g4, g5 = -0.0112, 0.1809, 0.1809

# Build 2-qubit operators (4×4 matrices)
I = identity(4).data   # 4×4 identity for 2-qubit system
Z0 = np.kron(pauli_z().data, np.eye(2))
Z1 = np.kron(np.eye(2), pauli_z().data)
Z0Z1 = np.kron(pauli_z().data, pauli_z().data)
X0X1 = np.kron(pauli_x().data, pauli_x().data)
Y0Y1 = np.kron(pauli_y().data, pauli_y().data)

# Combine to form the full Hamiltonian
H_molecule = (g0 * I + g1 * Z0 + g2 * Z1 + 
              g3 * Z0Z1 + g4 * X0X1 + g5 * Y0Y1)
```

!!! note "Why `identity(4)`?"
    For a 2-qubit system, the Hilbert space has dimension `2² = 4`. We use `identity(4)` to create a 4×4 identity matrix that matches the tensor product dimensions of our other operators.

### Step 2: Define the Variational Ansatz

We design a parameterized quantum circuit that can explore the relevant part of the Hilbert space. This **Hardware-Efficient Ansatz** uses single-qubit rotations and entangling gates.

```python
def create_ansatz(theta):
    """2-qubit variational circuit for H2."""
    circ = QuantumCircuit(2)
    
    # Hartree-Fock initialization: |10⟩
    circ.x(0)
    
    # Variational layers
    circ.ry(0, theta[0])
    circ.ry(1, theta[1])
    circ.cx(0, 1)              # Entanglement
    circ.ry(0, theta[2])
    circ.ry(1, theta[3])
    
    return circ
```

!!! tip "Hartree-Fock Initialization"
    We start from the Hartree-Fock state `|10⟩` (one electron per atomic orbital in minimal basis) rather than `|00⟩`. This provides a physically meaningful starting point that's already close to the true ground state, helping the optimizer converge faster.

### Step 3: Define the Cost Function

The cost function evaluates the molecular energy for a given set of parameters using the quantum expectation value.

```python
def cost_function(theta):
    """Calculate E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩."""
    circ = create_ansatz(theta)
    state = circ.run()
    
    state_vector = state.data
    energy = np.real(np.vdot(state_vector, H_molecule @ state_vector))
    return float(energy)
```

### Step 4: Run the Optimization

We use the classical **BFGS optimizer** to find the parameters that minimize the energy.

```python
# Initial parameters (non-zero to avoid flat gradient regions)
initial_theta = np.array([0.5, 0.5, 0.5, 0.5])

result = minimize(
    cost_function,
    x0=initial_theta,
    method='BFGS',
    options={'maxiter': 100}
)

optimal_theta = result.x
optimal_energy = result.fun
```

### Step 5: Verify Against Exact Solution

We can verify VQE's result by comparing it to the exact ground state energy obtained via classical diagonalization.

```python
eigenvalues = np.linalg.eigvalsh(H_molecule)
exact_ground_energy = np.min(eigenvalues)

error = abs(optimal_energy - exact_ground_energy)
print(f"VQE energy: {optimal_energy:.6f} Ha")
print(f"Exact energy: {exact_ground_energy:.6f} Ha")
print(f"Error: {error:.2e} Ha")
```

!!! success "Chemical Accuracy"
    In quantum chemistry, an error below **1.6 × 10⁻³ Ha** (1 kcal/mol) is considered "chemical accuracy." Our VQE typically achieves errors on the order of **10⁻¹¹ Ha**, far exceeding this threshold.

---

## Complete Code

Here is the complete, runnable script:

```python
"""
Advanced Use Case 3: Quantum Chemistry with VQE (H₂ Molecule)
Complete runnable example using PSIQIT.
"""

import numpy as np
from scipy.optimize import minimize
from psiqit.circuits import QuantumCircuit
from psiqit.quantum import pauli_z, pauli_x, pauli_y, identity

print("=" * 70)
print("ADVANCED USE CASE 3: Quantum Chemistry with VQE (H₂)")
print("=" * 70)

# 1. Build Molecular Hamiltonian
g0, g1, g2 = -1.0523, 0.3979, -0.3979
g3, g4, g5 = -0.0112, 0.1809, 0.1809

I = identity(4).data
Z0 = np.kron(pauli_z().data, np.eye(2))
Z1 = np.kron(np.eye(2), pauli_z().data)
Z0Z1 = np.kron(pauli_z().data, pauli_z().data)
X0X1 = np.kron(pauli_x().data, pauli_x().data)
Y0Y1 = np.kron(pauli_y().data, pauli_y().data)

H_molecule = (g0 * I + g1 * Z0 + g2 * Z1 + 
              g3 * Z0Z1 + g4 * X0X1 + g5 * Y0Y1)

# 2. Define Ansatz
def create_ansatz(theta):
    circ = QuantumCircuit(2)
    circ.x(0)
    circ.ry(0, theta[0])
    circ.ry(1, theta[1])
    circ.cx(0, 1)
    circ.ry(0, theta[2])
    circ.ry(1, theta[3])
    return circ

# 3. Cost Function
def cost_function(theta):
    state = create_ansatz(theta).run()
    return float(np.real(np.vdot(state.data, H_molecule @ state.data)))

# 4. Optimize
initial_theta = np.array([0.5, 0.5, 0.5, 0.5])
result = minimize(cost_function, x0=initial_theta, method='BFGS')

print(f"\nOptimal energy: {result.fun:.6f} Ha")

# 5. Verify
exact_energy = np.min(np.linalg.eigvalsh(H_molecule))
print(f"Exact energy:   {exact_energy:.6f} Ha")
print(f"Error: {abs(result.fun - exact_energy):.2e} Ha")

if abs(result.fun - exact_energy) < 1e-3:
    print("\n✓ SUCCESS: Chemical accuracy achieved!")

# 6. Analyze final state
final_state = create_ansatz(result.x).run()
probs = np.abs(final_state.data)**2
print("\nFinal state probabilities:")
for i, p in enumerate(probs):
    if p > 0.01:
        print(f"  |{format(i, '02b')}⟩: {p:.4f}")

print("=" * 70)
```

---

## Test It Yourself

You can find the complete test script in the repository:

```bash
python examples/advanced/test_vqe_molecule.py
```

---

## Understanding the Results

### Why does the final state look like |10⟩?

The dominant probability `P(|10⟩) ≈ 95.5%` corresponds to the **electronic configuration** of H₂ in the minimal basis. In the Jordan-Wigner mapping:
- `|10⟩` represents the Hartree-Fock-like ground state where the bonding orbital is occupied
- The small admixture of `|01⟩` (≈4.5%) captures the **electron correlation** effects that classical Hartree-Fock theory misses

### The Power of VQE

This example demonstrates several key advantages of VQE:
1. **No deep circuits**: Works with short, near-term friendly circuits
2. **Robust to noise**: Variational nature provides some error resilience
3. **Classical-quantum synergy**: Leverages the best of both worlds
4. **Chemical accuracy**: Achieves results meaningful to real chemists

---

## Next Steps

Now that you can simulate molecules, explore more:

- **Larger molecules**: Scale up to LiH, BeH₂ with more qubits
- **Potential Energy Surfaces**: Compute energy at different bond lengths
- **Excited states**: Use VQE variants like SSVQE or MC-VQE
- **[API Reference](../api/index.md)**: Explore PSIQIT's variational module

---

## References

- Peruzzo, A. et al. "A variational eigenvalue solver on a photonic quantum processor" (2014)
- Kandala, A. et al. "Hardware-efficient variational quantum eigensolver for small molecules" (2017)
- Tilly, J. et al. "The Variational Quantum Eigensolver: A review of methods and best practices" (2022)
- [PSIQIT API Reference](../api/variational.md) - Documentation for variational algorithms
