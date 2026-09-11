"""
Test script for Advanced Use Case 3: Quantum Chemistry with VQE
This script demonstrates simulating the H2 molecule using VQE.
"""

import numpy as np
from scipy.optimize import minimize
from psiqit.circuits import QuantumCircuit
from psiqit.quantum import pauli_z, pauli_x, pauli_y, identity

print("=" * 70)
print("ADVANCED USE CASE 3: Quantum Chemistry with VQE (H2 Molecule)")
print("=" * 70)

# ==============================================================================
# 1. Define the Molecular Hamiltonian
# ==============================================================================
print("\n1. Defining H2 Molecular Hamiltonian:")
print("-" * 70)

# Simplified H2 Hamiltonian in the STO-3G basis (2 qubits)
# H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1 + g5*Y0Y1
# These coefficients are approximate values for demonstration

g0 = -1.0523  # Constant term
g1 = 0.3979   # Z0 coefficient
g2 = -0.3979  # Z1 coefficient
g3 = -0.0112  # Z0Z1 coefficient
g4 = 0.1809   # X0X1 coefficient
g5 = 0.1809   # Y0Y1 coefficient

print("   H2 Hamiltonian (simplified STO-3G basis):")
print(f"   H = {g0:.4f}*I + {g1:.4f}*Z₀ + {g2:.4f}*Z₁ + {g3:.4f}*Z₀Z₁ + {g4:.4f}*X₀X₁ + {g5:.4f}*Y₀Y₁")

# Build the Hamiltonian matrix
I = identity(4).data
Z0 = np.kron(pauli_z().data, np.eye(2))
Z1 = np.kron(np.eye(2), pauli_z().data)
Z0Z1 = np.kron(pauli_z().data, pauli_z().data)
X0X1 = np.kron(pauli_x().data, pauli_x().data)
Y0Y1 = np.kron(pauli_y().data, pauli_y().data)

H_molecule = (g0 * I + g1 * Z0 + g2 * Z1 + g3 * Z0Z1 + 
              g4 * X0X1 + g5 * Y0Y1)

print(f"   Hamiltonian matrix shape: {H_molecule.shape}")

# ==============================================================================
# 2. Define the Ansatz (Parameterized Circuit)
# ==============================================================================
print("\n2. Defing Variational Ansatz:")
print("-" * 70)

def create_ansatz(theta):
    """Create a 2-qubit parameterized circuit for H2."""
    circ = QuantumCircuit(2)
    
    # Initial excitation (Hartree-Fock state preparation)
    circ.x(0)
    
    # Variational layers
    circ.ry(0, theta[0])
    circ.ry(1, theta[1])
    circ.cx(0, 1)
    circ.ry(0, theta[2])
    circ.ry(1, theta[3])
    
    return circ

print("   Ansatz structure:")
print("   - X gate on qubit 0 (Hartree-Fock initialization)")
print("   - Ry rotations on both qubits")
print("   - CNOT entanglement")
print("   - Additional Ry rotations")
print("   - Total parameters: 4")

# ==============================================================================
# 3. Define the Cost Function
# ==============================================================================
print("\n3. Defining Cost Function:")
print("-" * 70)

def cost_function(theta):
    """Calculate the energy expectation value <ψ(θ)|H|ψ(θ)>."""
    circ = create_ansatz(theta)
    state = circ.run()
    
    # Calculate <ψ|H|ψ>
    state_vector = state.data
    energy = np.real(np.vdot(state_vector, H_molecule @ state_vector))
    
    return float(energy)

print("   Cost function: E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩")
print("   This represents the molecular energy for given parameters θ")

# ==============================================================================
# 4. Run the Classical Optimizer
# ==============================================================================
print("\n4. Running Classical Optimizer (BFGS):")
print("-" * 70)

# Initial parameters (small random values to avoid zero gradient)
initial_theta = np.array([0.5, 0.5, 0.5, 0.5])

print(f"   Initial parameters: {initial_theta}")
print(f"   Initial energy: {cost_function(initial_theta):.6f} Ha")

# Optimize
result = minimize(
    cost_function,
    x0=initial_theta,
    method='BFGS',
    options={'disp': False, 'maxiter': 100}
)

optimal_theta = result.x
optimal_energy = result.fun

print(f"\n   Optimization successful: {result.success}")
print(f"   Optimal parameters: {optimal_theta}")
print(f"   Optimal energy: {optimal_energy:.6f} Ha")

# ==============================================================================
# 5. Compare with Exact Diagonalization
# ==============================================================================
print("\n5. Verification (Exact Diagonalization):")
print("-" * 70)

# Calculate exact ground state energy
eigenvalues = np.linalg.eigvalsh(H_molecule)
exact_ground_energy = np.min(eigenvalues)

print(f"   Exact ground state energy: {exact_ground_energy:.6f} Ha")
print(f"   VQE optimized energy:      {optimal_energy:.6f} Ha")
print(f"   Error: {abs(optimal_energy - exact_ground_energy):.2e} Ha")

if abs(optimal_energy - exact_ground_energy) < 1e-3:
    print("\n   ✓ SUCCESS: VQE found the ground state energy with high accuracy!")
    print("   ✓ This demonstrates quantum chemistry simulation capability.")
else:
    print("\n   ℹ Note: Error is larger than expected. May need more optimization iterations.")

# ==============================================================================
# 6. Analyze the Final State
# ==============================================================================
print("\n6. Final State Analysis:")
print("-" * 70)

final_circ = create_ansatz(optimal_theta)
final_state = final_circ.run()

print(f"   Final statevector: {final_state.data}")

# Calculate probabilities
probs = np.abs(final_state.data)**2
print(f"\n   State probabilities:")
for i, prob in enumerate(probs):
    if prob > 0.01:
        binary = format(i, '02b')
        print(f"      |{binary}⟩: {prob:.4f} ({prob*100:.2f}%)")

print("\n" + "=" * 70)
print("Advanced Use Case 3 simulation completed!")
print("=" * 70)