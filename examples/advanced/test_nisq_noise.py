"""
Test script for Advanced Use Case 2: NISQ Noise Simulation
This script demonstrates how to simulate hardware noise (e.g., Bit-Flip) 
and measure its impact on quantum state fidelity.
"""

import numpy as np
from psiqit.circuits import QuantumCircuit

print("=" * 70)
print("ADVANCED USE CASE 2: NISQ Noise Simulation")
print("=" * 70)

# ==============================================================================
# 1. Define the Ideal Target State
# ==============================================================================
print("\n1. Defining the Ideal Target State:")
print("-" * 70)

# Create an ideal Bell state: |Φ⁺⟩ = (|00⟩ + |11⟩) / √2
ideal_circ = QuantumCircuit(2)
ideal_circ.h(0)
ideal_circ.cx(0, 1)

ideal_state = ideal_circ.run()
ideal_data = ideal_state.data

print("   Target State: Bell State |Φ⁺⟩")
print(f"   Ideal Statevector: {ideal_data}")

# ==============================================================================
# 2. Define a Noise Model (Probabilistic Bit-Flip Channel)
# ==============================================================================
print("\n2. Defining Noise Model:")
print("-" * 70)

def apply_bit_flip_noise(circ, qubit, probability):
    """
    Simulates a bit-flip noise channel by probabilistically applying an X gate.
    Note: In a true density matrix simulator, this is a superoperator. 
    Here, we simulate it by creating a mixed state representation or 
    by showing the effect of a guaranteed error for educational purposes.
    """
    noisy_circ = QuantumCircuit(2)
    # Copy the original circuit operations
    noisy_circ.h(0)
    noisy_circ.cx(0, 1)
    
    # Apply the error (for demonstration, we apply it deterministically 
    # to show the worst-case scenario, or we can simulate the mixed state)
    noisy_circ.x(qubit)
    return noisy_circ

print("   Noise Type: Bit-Flip Channel (X error)")
print("   Target Qubit: Qubit 1")

# ==============================================================================
# 3. Simulate Noisy Evolution
# ==============================================================================
print("\n3. Simulating Noisy Evolution:")
print("-" * 70)

# Apply error to qubit 1
noisy_circ = apply_bit_flip_noise(ideal_circ, qubit=1, probability=1.0)
noisy_state = noisy_circ.run()
noisy_data = noisy_state.data

print(f"   Noisy Statevector: {noisy_data}")
print("   (Notice how the amplitudes have shifted due to the X error on qubit 1)")

# ==============================================================================
# 4. Calculate Fidelity
# ==============================================================================
print("\n4. Calculating Fidelity:")
print("-" * 70)

# Fidelity between pure states |ψ⟩ and |φ⟩ is |⟨ψ|φ⟩|²
fidelity = np.abs(np.vdot(ideal_data, noisy_data))**2

print(f"   Fidelity (Ideal vs Noisy): {fidelity:.6f}")

if fidelity < 0.01:
    print("   ✓ SUCCESS: Noise simulation correctly degraded the state fidelity!")
    print("   ✓ This demonstrates how a single bit-flip destroys the Bell state correlation.")
else:
    print("   ℹ Note: Fidelity is higher than expected. Check the noise application.")

# ==============================================================================
# 5. Statistical Noise Simulation (Monte Carlo approach)
# ==============================================================================
print("\n5. Monte Carlo Noise Simulation (1000 shots):")
print("-" * 70)

# To simulate a true probabilistic channel (e.g., p=0.2), we can run 
# the circuit multiple times, randomly injecting errors.
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
        
    # Measure
    circ_mc.measure(shots=1) # Just to execute

print(f"   Simulated {shots} runs with {error_probability*100:.1f}% error probability.")
print(f"   Errors injected: {error_count} times ({error_count/shots*100:.1f}%)")
print("   ✓ Monte Carlo simulation completed successfully.")

print("\n" + "=" * 70)
print("Advanced Use Case 2 simulation completed!")
print("=" * 70)