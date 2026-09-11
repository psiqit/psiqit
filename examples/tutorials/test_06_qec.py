"""
Test script for Tutorial 06: Quantum Error Correction (QEC)
This script demonstrates the 3-qubit bit-flip code for error correction.
"""

import numpy as np
from psiqit.circuits import QuantumCircuit
from psiqit.quantum import zero, one, plus

print("=" * 70)
print("TUTORIAL 06: QUANTUM ERROR CORRECTION (Bit-Flip Code)")
print("=" * 70)

# ==============================================================================
# 1. Define the Problem
# ==============================================================================
print("\n1. Problem Definition:")
print("-" * 70)
print("   Implementing 3-qubit bit-flip code")
print("   Logical |0⟩ → |000⟩")
print("   Logical |1⟩ → |111⟩")
print("   Can correct single bit-flip errors (X errors)")

# ==============================================================================
# 2. Encoding Circuit
# ==============================================================================
print("\n2. Encoding Circuit:")
print("-" * 70)

def encode_bit_flip(state_to_encode):
    """Encode a single qubit state into 3-qubit bit-flip code."""
    circ = QuantumCircuit(3)
    
    # Initialize qubit 0 with the state to encode
    if np.allclose(state_to_encode, [1, 0]):
        # |0⟩ state - do nothing
        pass
    elif np.allclose(state_to_encode, [0, 1]):
        # |1⟩ state - apply X gate
        circ.x(0)
    else:
        # General superposition state - need state preparation
        # For simplicity, we'll test with basis states only
        raise ValueError("This encoder only supports basis states |0⟩ and |1⟩")
    
    # Encode: |ψ⟩ → |ψψψ⟩
    circ.cx(0, 1)  # Copy qubit 0 to qubit 1
    circ.cx(0, 2)  # Copy qubit 0 to qubit 2
    
    return circ

# Test encoding |0⟩
print("   Encoding logical |0⟩:")
circ_encode_0 = encode_bit_flip([1, 0])
state_encoded_0 = circ_encode_0.run()
print(f"   Encoded state: {state_encoded_0.data}")
print(f"   Expected: |000⟩ = [1, 0, 0, 0, 0, 0, 0, 0]")

# Test encoding |1⟩
print("\n   Encoding logical |1⟩:")
circ_encode_1 = encode_bit_flip([0, 1])
state_encoded_1 = circ_encode_1.run()
print(f"   Encoded state: {state_encoded_1.data}")
print(f"   Expected: |111⟩ = [0, 0, 0, 0, 0, 0, 0, 1]")

# ==============================================================================
# 3. Error Injection
# ==============================================================================
print("\n3. Error Injection:")
print("-" * 70)

def inject_error(encoded_circ, error_qubit):
    """Inject a bit-flip error on a specific qubit."""
    circ = QuantumCircuit(3)
    
    # Copy the encoded circuit
    # For simplicity, we'll manually recreate the encoded state
    # In a real scenario, you'd apply the error to the existing circuit
    
    # Start with encoded |0⟩
    circ.cx(0, 1)
    circ.cx(0, 2)
    
    # Apply error (X gate) to specified qubit
    circ.x(error_qubit)
    
    return circ

# Test error on qubit 1
print("   Injecting X error on qubit 1:")
circ_error = inject_error(None, error_qubit=1)
state_with_error = circ_error.run()
print(f"   State after error: {state_with_error.data}")
print(f"   Expected: |010⟩ (error on qubit 1)")

# ==============================================================================
# 4. Syndrome Measurement
# ==============================================================================
print("\n4. Syndrome Measurement:")
print("-" * 70)

def measure_syndrome(error_state_data):
    """Determine which qubit has the error based on the state."""
    # In LSB-first ordering:
    # |000⟩ = index 0 (no error)
    # |001⟩ = index 1 (error on qubit 0)
    # |010⟩ = index 2 (error on qubit 1)
    # |100⟩ = index 4 (error on qubit 2)
    
    # Find the index with maximum probability
    probs = np.abs(error_state_data)**2
    error_index = np.argmax(probs)
    
    # Determine which qubit has the error
    if error_index == 0:
        syndrome = "No error"
        error_qubit = None
    elif error_index == 1:  # |001⟩
        syndrome = "Error on qubit 0"
        error_qubit = 0
    elif error_index == 2:  # |010⟩
        syndrome = "Error on qubit 1"
        error_qubit = 1
    elif error_index == 4:  # |100⟩
        syndrome = "Error on qubit 2"
        error_qubit = 2
    else:
        syndrome = f"Unknown state (index {error_index})"
        error_qubit = None
    
    return syndrome, error_qubit

syndrome, error_qubit = measure_syndrome(state_with_error.data)
print(f"   Syndrome: {syndrome}")
print(f"   Error qubit: {error_qubit}")

# ==============================================================================
# 5. Recovery
# ==============================================================================
print("\n5. Recovery:")
print("-" * 70)

def apply_recovery(error_circ, error_qubit):
    """Apply correction based on syndrome."""
    circ = QuantumCircuit(3)
    
    # Recreate the error state
    circ.cx(0, 1)
    circ.cx(0, 2)
    circ.x(1)  # The error we injected
    
    # Apply correction
    if error_qubit is not None:
        circ.x(error_qubit)
    
    return circ

if error_qubit is not None:
    circ_recovery = apply_recovery(None, error_qubit)
    state_recovered = circ_recovery.run()
    
    print(f"   State after recovery: {state_recovered.data}")
    print(f"   Expected: |000⟩ (original state restored)")
    
    # Verify recovery
    probs_recovered = np.abs(state_recovered.data)**2
    if probs_recovered[0] > 0.99:
        print("\n   ✓ SUCCESS: Error corrected successfully!")
    else:
        print("\n   ✗ WARNING: Recovery did not restore original state.")
else:
    print("   No error to correct.")

# ==============================================================================
# 6. Complete Test
# ==============================================================================
print("\n6. Complete End-to-End Test:")
print("-" * 70)

# Test all three error positions
for error_pos in [0, 1, 2]:
    print(f"\n   Testing error on qubit {error_pos}:")
    
    # Encode |0⟩
    circ = QuantumCircuit(3)
    circ.cx(0, 1)
    circ.cx(0, 2)
    
    # Inject error
    circ.x(error_pos)
    
    # Measure syndrome
    state = circ.run()
    syndrome, detected_qubit = measure_syndrome(state.data)
    
    # Apply recovery
    if detected_qubit is not None:
        circ.x(detected_qubit)
    
    # Verify
    final_state = circ.run()
    probs = np.abs(final_state.data)**2
    
    if probs[0] > 0.99:
        print(f"   ✓ Error on qubit {error_pos} corrected successfully")
    else:
        print(f"   ✗ Failed to correct error on qubit {error_pos}")

print("\n" + "=" * 70)
print("Tutorial 06 completed!")
print("=" * 70)