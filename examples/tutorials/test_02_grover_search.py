"""
Test script for Tutorial 02: Grover's Search Algorithm
This script demonstrates and verifies Grover's search using PSIQIT.
"""

import numpy as np
from psiqit.algorithms import grover_search
from psiqit.circuits import QuantumCircuit
from psiqit.visualization import draw_circuit

print("=" * 70)
print("TUTORIAL 02: GROVER'S SEARCH ALGORITHM")
print("=" * 70)

# 1. Define the problem
n_qubits = 3
target = 5  # Binary: 101 (LSB-first: qubit 0=1, qubit 1=0, qubit 2=1)
shots = 1000

print(f"\n1. Problem Definition:")
print(f"   Number of qubits: {n_qubits}")
print(f"   Database size: {2**n_qubits} items")
print(f"   Target state: |{format(target, f'0{n_qubits}b')}⟩ (decimal: {target})")

# 2. Calculate optimal iterations
optimal_iterations = int(np.pi / 4 * np.sqrt(2**n_qubits))
print(f"\n2. Optimal Iterations: {optimal_iterations}")

# 3. Run built-in Grover's Search
print(f"\n3. Running built-in grover_search() with {shots} shots...")
result = grover_search(n_qubits=n_qubits, target=target, shots=shots)

print("\n4. Results:")
print("-" * 70)
print(f"   Most likely state (decimal): {result['most_likely']}")
print(f"   Most likely state (binary) : {result['most_likely_binary']}")
print(f"   Success probability        : {result['probability_target']:.2%}")
print(f"   All counts                 : {result['counts']}")

# 5. Verification
print("\n5. Verification:")
print("-" * 70)
success_prob = result['probability_target']
is_correct = result['most_likely'] == target

if is_correct and success_prob > 0.90:
    print("   ✓ SUCCESS: Grover's algorithm found the target with high probability!")
    print("   ✓ Quadratic speedup demonstrated.")
else:
    print(f"   ✗ WARNING: Expected target {target}, got {result['most_likely']}")
    print(f"   ✗ Probability was {success_prob:.2%} (expected > 90%)")

print("\n" + "=" * 70)
print("Tutorial 02 completed!")
print("=" * 70)