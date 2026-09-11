"""
Test script for Tutorial 01: Bell State
This script demonstrates creating and measuring a Bell state using PSIQIT.
"""

from psiqit.circuits import QuantumCircuit
from psiqit.visualization import draw_circuit
import numpy as np

print("=" * 70)
print("TUTORIAL 01: BELL STATE")
print("=" * 70)

# Step 1: Create a 2-qubit circuit
print("\n1. Creating a 2-qubit circuit...")
circ = QuantumCircuit(2)
print(f"   Circuit initialized with {circ.n_qubits} qubits")

# Step 2: Apply Hadamard gate to qubit 0
print("\n2. Applying Hadamard gate to qubit 0...")
circ.h(0)
print("   H gate applied")

# Step 3: Apply CNOT gate (control=0, target=1)
print("\n3. Applying CNOT gate (control=0, target=1)...")
circ.cx(0, 1)
print("   CNOT gate applied")

# Step 4: Draw the circuit
print("\n4. Circuit diagram:")
print("-" * 70)
print(draw_circuit(circ))

# Step 5: Run the circuit to get state vector
print("\n5. Running the circuit...")
state = circ.run()
print(f"   State vector: {state.data}")
print(f"   Type: {type(state)}")

# Step 6: Calculate probabilities
print("\n6. Probability distribution:")
print("-" * 70)
probabilities = np.abs(state.data)**2
for i, prob in enumerate(probabilities):
    if prob > 0.01:  # Only show non-negligible probabilities
        binary = format(i, '02b')
        print(f"   |{binary}⟩: {prob:.4f} ({prob*100:.2f}%)")

# Step 7: Measure the circuit
print("\n7. Measuring with 1024 shots...")
result = circ.measure(shots=1024)
print(f"   Counts: {result['counts']}")

# Step 8: Verify entanglement
print("\n8. Verifying entanglement:")
print("-" * 70)
counts = result['counts']
total = sum(counts.values())
p_00 = counts.get('00', 0) / total
p_11 = counts.get('11', 0) / total
p_01 = counts.get('01', 0) / total
p_10 = counts.get('10', 0) / total

print(f"   P(00) = {p_00:.4f}")
print(f"   P(11) = {p_11:.4f}")
print(f"   P(01) = {p_01:.4f}")
print(f"   P(10) = {p_10:.4f}")

if p_00 > 0.4 and p_11 > 0.4 and p_01 < 0.05 and p_10 < 0.05:
    print("\n   ✓ SUCCESS: Bell state created successfully!")
    print("   The qubits are entangled - they always measure the same.")
else:
    print("\n   ✗ WARNING: Results don't match expected Bell state")

print("\n" + "=" * 70)
print("Tutorial 01 completed!")
print("=" * 70)