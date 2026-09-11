"""
Test script for Tutorial 05: Lindblad Dynamics
This script demonstrates simulating open quantum systems and decoherence.
"""

import numpy as np
from psiqit.quantum import pauli_z, pauli_x
from psiqit.dynamics import LindbladSolver

print("=" * 70)
print("TUTORIAL 05: LINDBLAD DYNAMICS (Open Quantum Systems)")
print("=" * 70)

# ==============================================================================
# 1. Define the Problem
# ==============================================================================
print("\n1. Problem Definition:")
print("-" * 70)
print("   Simulating a qubit undergoing decoherence")
print("   Hamiltonian: H = Z (qubit in magnetic field)")
print("   Collapse operator: L = X (spontaneous emission)")
print("   Decay rate: gamma = 0.1")

# Define Hamiltonian
H = pauli_z()

# Define collapse operators
L = [pauli_x()]
gamma = [0.1]

# Initial state: |0⟩⟨0| as density matrix
rho0 = np.array([[1, 0], [0, 0]], dtype=complex)

print(f"   Initial state (density matrix):")
print(f"   {rho0}")

# ==============================================================================
# 2. Create the Solver
# ==============================================================================
print("\n2. Creating Lindblad Solver:")
print("-" * 70)

solver = LindbladSolver(H, L, gamma=gamma)
print("   ✓ Solver created successfully")

# ==============================================================================
# 3. Solve the Dynamics
# ==============================================================================
print("\n3. Solving Lindblad Master Equation:")
print("-" * 70)

t_max = 10.0
dt = 0.1

result = solver.evolve(rho0, t_max=t_max, dt=dt, method='euler')

print(f"   Time span: 0 to {t_max}")
print(f"   Time steps: {len(result.times)}")
print("   ✓ Evolution completed successfully")

# ==============================================================================
# 4. Analyze Results
# ==============================================================================
print("\n4. Analyzing Results:")
print("-" * 70)

times = result.times
states = result.states

# Extract populations (diagonal elements of density matrix)
populations_0 = []
populations_1 = []

for rho in states:
    pop_0 = np.real(rho[0, 0])
    pop_1 = np.real(rho[1, 1])
    populations_0.append(pop_0)
    populations_1.append(pop_1)

print(f"   Initial population |0⟩: {populations_0[0]:.4f}")
print(f"   Final population |0⟩: {populations_0[-1]:.4f}")
print(f"   Initial population |1⟩: {populations_1[0]:.4f}")
print(f"   Final population |1⟩: {populations_1[-1]:.4f}")

# Show evolution at key time points
print("\n   Evolution at key time points:")
key_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
for idx in key_indices:
    t = times[idx]
    p0 = populations_0[idx]
    p1 = populations_1[idx]
    print(f"   t={t:5.1f}: P(|0⟩)={p0:.4f}, P(|1⟩)={p1:.4f}")

# ==============================================================================
# 5. Verification
# ==============================================================================
print("\n5. Verification:")
print("-" * 70)

# Check if trace is preserved (should be 1.0)
final_rho = result.final_state
trace = np.trace(final_rho).real

print(f"   Final trace: {trace:.6f}")

if abs(trace - 1.0) < 1e-5:
    print("   ✓ SUCCESS: Lindblad dynamics working correctly!")
    print("   ✓ Trace preservation verified.")
else:
    print("   ✗ WARNING: Trace not preserved. Check implementation.")

# Check physical consistency (populations should be non-negative)
if all(p >= -1e-10 for p in populations_0) and all(p >= -1e-10 for p in populations_1):
    print("   ✓ All populations are non-negative (physically valid).")
else:
    print("   ✗ WARNING: Some populations are negative (unphysical).")

print("\n" + "=" * 70)
print("Tutorial 05 completed!")
print("=" * 70)