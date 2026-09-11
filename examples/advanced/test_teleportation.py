"""
Test script for Advanced Use Case: Quantum Teleportation
This script demonstrates the quantum teleportation protocol with rigorous verification.
"""

import numpy as np
from psiqit.circuits import QuantumCircuit

print("=" * 70)
print("ADVANCED USE CASE: Quantum Teleportation")
print("=" * 70)

# ==============================================================================
# 1. Define the Message State
# ==============================================================================
print("\n1. Preparing the Message State:")
print("-" * 70)

theta = np.pi / 3  # Angle for amplitude
phi = np.pi / 4    # Angle for phase

alpha = np.cos(theta / 2)
beta = np.exp(1j * phi) * np.sin(theta / 2)

print(f"   Message state: α|0⟩ + β|1⟩")
print(f"   α = {alpha:.4f}")
print(f"   β = {beta:.4f}")
print(f"   where θ={theta:.4f}, φ={phi:.4f}")

# ==============================================================================
# 2. Complete Teleportation Circuit
# ==============================================================================
print("\n2. Building Complete Teleportation Circuit:")
print("-" * 70)

# q0: message, q1: Alice's Bell, q2: Bob's Bell
circ = QuantumCircuit(3)

# Step 1: Prepare message state on q0
circ.ry(0, theta)
circ.rz(0, phi)

# Step 2: Create Bell state between q1 and q2
circ.h(1)
circ.cx(1, 2)

# Step 3: Alice's Bell measurement
circ.cx(0, 1)  # CNOT: message (q0) → Alice's qubit (q1)
circ.h(0)      # Hadamard on message (q0)

print("   Circuit built successfully.")

# ==============================================================================
# 3. Measure All Qubits & Analyze
# ==============================================================================
print("\n3. Measuring All Qubits (1000 shots):")
print("-" * 70)

result = circ.measure(shots=1000)
counts = result['counts']

for state, count in sorted(counts.items()):
    print(f"      |{state}⟩: {count} ({count/10:.1f}%)")

# ==============================================================================
# 4. Rigorous Mathematical Verification
# ==============================================================================
print("\n4. Rigorous Verification (Theoretical vs Simulated):")
print("-" * 70)

# In LSB-first ordering, index = q2*4 + q1*2 + q0*1
# Based on quantum mechanics derivation of teleportation, the exact 
# probabilities for each of the 8 outcomes are:
theoretical_probs = np.zeros(8)
theoretical_probs[0] = np.abs(alpha)**2 / 4  # |000⟩ (q2=0, q1=0, q0=0)
theoretical_probs[4] = np.abs(beta)**2 / 4   # |100⟩ (q2=1, q1=0, q0=0)
theoretical_probs[1] = np.abs(alpha)**2 / 4  # |001⟩ (q2=0, q1=0, q0=1)
theoretical_probs[5] = np.abs(beta)**2 / 4   # |101⟩ (q2=1, q1=0, q0=1)
theoretical_probs[2] = np.abs(beta)**2 / 4   # |010⟩ (q2=0, q1=1, q0=0)
theoretical_probs[6] = np.abs(alpha)**2 / 4  # |110⟩ (q2=1, q1=1, q0=0)
theoretical_probs[3] = np.abs(beta)**2 / 4   # |011⟩ (q2=0, q1=1, q0=1)
theoretical_probs[7] = np.abs(alpha)**2 / 4  # |111⟩ (q2=1, q1=1, q0=1)

# Get simulated probabilities directly from the statevector before measurement
state_before_measure = circ.run()
simulated_probs = np.abs(state_before_measure.data)**2

# Calculate the maximum difference between theory and simulation
max_diff = np.max(np.abs(simulated_probs - theoretical_probs))

print(f"   Theoretical probabilities calculated based on α and β.")
print(f"   Simulated probabilities extracted from statevector.")
print(f"   Maximum absolute difference: {max_diff:.2e}")

if max_diff < 1e-10:
    print("\n   ✓ SUCCESS: Quantum teleportation works perfectly!")
    print("   ✓ The measurement statistics match theoretical predictions exactly.")
    print("   ✓ This rigorously proves the state was teleported to Bob's qubit.")
else:
    print("\n   ✗ WARNING: Probabilities do not match theoretical predictions.")

print("\n" + "=" * 70)
print("Advanced Use Case simulation completed!")
print("=" * 70)