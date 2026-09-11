"""
Test script for Tutorial 04: Variational Quantum Eigensolver (VQE)
This script demonstrates finding the ground state energy of a simple Hamiltonian.
"""

import numpy as np
from scipy.optimize import minimize
from psiqit.circuits import QuantumCircuit
from psiqit.quantum import pauli_z, expectation

print("=" * 70)
print("TUTORIAL 04: VARIATIONAL QUANTUM EIGENSOLVER (VQE)")
print("=" * 70)

# ==============================================================================
# 1. Define the Problem
# ==============================================================================
print("\n1. Problem Definition:")
print("-" * 70)
print("   Hamiltonian: H = Z (1-qubit Pauli-Z)")
print("   True ground state: |1>")
print("   True ground energy: -1.0")

H = pauli_z()

# ==============================================================================
# 2. Define the Ansatz (Parameterized Circuit)
# ==============================================================================
def create_ansatz(theta):
    """Create a parameterized quantum circuit."""
    circ = QuantumCircuit(1)
    circ.ry(0, theta)  # Rotate around Y-axis by theta
    return circ

# ==============================================================================
# 3. Define the Cost Function (Expectation Value)
# ==============================================================================
def cost_function(theta):
    """Calculate the expectation value <psi(theta)|H|psi(theta)>."""
    circ = create_ansatz(theta[0])  # scipy passes theta as an array
    state = circ.run()
    
    # Now this works perfectly! The library handles the argument order smartly.
    energy = expectation(state, H)
    return float(energy)

# ==============================================================================
# 4. Run the Classical Optimizer
# ==============================================================================
print("\n2. Running Classical Optimizer (BFGS):")
print("-" * 70)

initial_theta = 0.5  # Start at |0> state
print(f"   Initial theta: {initial_theta:.4f} rad")
print(f"   Initial energy: {cost_function([initial_theta]):.4f}")

result = minimize(
    cost_function,
    x0=[initial_theta],
    method='BFGS',
    options={'disp': False}
)

optimal_theta = result.x[0]
optimal_energy = result.fun

print(f"\n   Optimization successful: {result.success}")
print(f"   Optimal theta: {optimal_theta:.4f} rad (approx {optimal_theta/np.pi:.4f} * pi)")
print(f"   Optimal energy: {optimal_energy:.4f}")

# ==============================================================================
# 5. Verify the Result
# ==============================================================================
print("\n3. Verification:")
print("-" * 70)

final_circ = create_ansatz(optimal_theta)
final_state = final_circ.run()

print(f"   Final state vector: {final_state.data}")

# Check if it's close to |1> (which is [0, 1] in LSB-first)
target_state = np.array([0.0, 1.0], dtype=complex)
fidelity = np.abs(np.vdot(target_state, final_state.data))**2

print(f"   Fidelity with true ground state |1>: {fidelity:.4f}")

if abs(optimal_energy - (-1.0)) < 1e-4 and fidelity > 0.99:
    print("\n   ✓ SUCCESS: VQE found the correct ground state and energy!")
else:
    print("\n   ✗ WARNING: VQE did not converge to the expected ground state.")

print("\n" + "=" * 70)
print("Tutorial 04 completed!")
print("=" * 70)