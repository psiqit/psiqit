
# Measurement & Observables

This guide covers quantum measurement theory and how to extract information from quantum states in PSIQIT. :)

## Quantum Measurement Basics

In quantum mechanics, measurement is a fundamental operation that collapses a quantum state into one of the basis states. PSIQIT provides flexible measurement capabilities for both simulation and analysis.

## Computational Basis Measurement

The most common type of measurement is in the computational basis `{|0⟩, |1⟩}`.

### Measuring a Single Qubit

```python
from psiqit.circuits import QuantumCircuit

# Create a superposition state
circ = QuantumCircuit(1)
circ.h(0)  # |+⟩ = (|0⟩ + |1⟩) / √2

# Measure with 1000 shots
result = circ.measure(shots=1000)

print("Counts:", result['counts'])
print("Probabilities:", result['probabilities'])
```

**Output:**
```text
Counts: {'0': 503, '1': 497}
Probabilities: {'0': 0.503, '1': 0.497}
```

!!! note "Statistical Nature"
    Quantum measurement is probabilistic. Running the same circuit multiple times will give slightly different results due to the statistical nature of quantum mechanics.

### Measuring Multiple Qubits

```python
from psiqit.circuits import QuantumCircuit

# Create a Bell state
circ = QuantumCircuit(2)
circ.h(0).cx(0, 1)

# Measure all qubits
result = circ.measure(shots=1024)

print("Counts:", result['counts'])
```

**Output:**
```text
Counts: {'00': 512, '11': 512}
```

!!! warning "Qubit Ordering"
    Remember that PSIQIT uses **LSB-first** ordering. The measurement result `'01'` means qubit 0 is in state `|1⟩` and qubit 1 is in state `|0⟩`.

### Measuring Specific Qubits

You can measure only a subset of qubits:

```python
circ = QuantumCircuit(3)
circ.h(0).cx(0, 1).h(2)

# Measure only qubits 0 and 1
result = circ.measure(qubits=[0, 1], shots=1000)

print("Counts:", result['counts'])
```

---

## Expectation Values

The expectation value of an observable `O` for a state `|ψ⟩` is defined as:

`⟨O⟩ = ⟨ψ|O|ψ⟩`

### Computing Expectation Values

```python
from psiqit.quantum import zero, plus, pauli_z, pauli_x, expectation

# State |0⟩
q0 = zero()

# Expectation value of Z for |0⟩
# ⟨0|Z|0⟩ = 1
Z = pauli_z()
exp_Z = expectation(q0, Z)
print("⟨0|Z|0⟩:", exp_Z)

# Expectation value of X for |0⟩
# ⟨0|X|0⟩ = 0
X = pauli_x()
exp_X = expectation(q0, X)
print("⟨0|X|0⟩:", exp_X)

# State |+⟩
q_plus = plus()

# Expectation value of X for |+⟩
# ⟨+|X|+⟩ = 1
exp_X_plus = expectation(q_plus, X)
print("⟨+|X|+⟩:", exp_X_plus)
```

**Output:**
```text
⟨0|Z|0⟩: 1.0
⟨0|X|0⟩: 0.0
⟨+|X|+⟩: 1.0
```

### Multi-Qubit Expectation Values

For multi-qubit systems, you need to construct the full observable:

```python
from psiqit.quantum import (
    tensor_product, identity, pauli_z, pauli_x, 
    bell_phi_plus, expectation
)

# Bell state: (|00⟩ + |11⟩) / √2
bell = bell_phi_plus()

# Observable: Z ⊗ I (measure Z on first qubit)
I = identity(2)
Z_I = tensor_product(pauli_z(), I)

exp_Z1 = expectation(bell, Z_I)
print("⟨Z⊗I⟩:", exp_Z1)

# Observable: Z ⊗ Z (correlation between qubits)
Z_Z = tensor_product(pauli_z(), pauli_z())

exp_ZZ = expectation(bell, Z_Z)
print("⟨Z⊗Z⟩:", exp_ZZ)
```

**Output:**
```text
⟨Z⊗I⟩: 0.0
⟨Z⊗Z⟩: 1.0
```

!!! tip "Quantum Correlations"
    The result `⟨Z⊗Z⟩ = 1.0` for the Bell state shows perfect correlation: when you measure both qubits, you always get the same result (both `0` or both `1`).

---

## Variance and Uncertainty

The variance of an observable `O` is:

`Var(O) = ⟨O²⟩ - ⟨O⟩²`

### Computing Variance

```python
from psiqit.quantum import plus, pauli_z, pauli_x, expectation
import numpy as np

# State |+⟩
q_plus = plus()

# Observable Z
Z = pauli_z()

# ⟨Z⟩ for |+⟩
exp_Z = expectation(q_plus, Z)

# ⟨Z²⟩ for |+⟩ (Z² = I)
Z_squared = Z @ Z
exp_Z2 = expectation(q_plus, Z_squared)

# Variance
variance = exp_Z2 - exp_Z**2
std_dev = np.sqrt(variance)

print("⟨Z⟩:", exp_Z)
print("⟨Z²⟩:", exp_Z2)
print("Variance:", variance)
print("Standard Deviation:", std_dev)
```

**Output:**
```text
⟨Z⟩: 0.0
⟨Z²⟩: 1.0
Variance: 1.0
Standard Deviation: 1.0
```

---

## Measurement on Circuits

### Getting State Vector Before Measurement

```python
from psiqit.circuits import QuantumCircuit

circ = QuantumCircuit(2)
circ.h(0).cx(0, 1)

# Get the state vector (no measurement)
state = circ.run()
print("State vector:", state.data)

# Now measure
result = circ.measure(shots=1024)
print("Counts:", result['counts'])
```

### Measurement with Different Shot Counts

```python
circ = QuantumCircuit(2)
circ.h(0).cx(0, 1)

# Small number of shots
result_100 = circ.measure(shots=100)
print("100 shots:", result_100['counts'])

# Large number of shots
result_10000 = circ.measure(shots=10000)
print("10000 shots:", result_10000['counts'])
```

!!! note "Shot Count"
    More shots give better statistics and more accurate probability estimates, but take longer to compute. For quick tests, 100-1000 shots are sufficient. For publication-quality results, use 10000+ shots.

---

## Observables from Circuits

You can compute expectation values directly from circuits:

```python
from psiqit.circuits import QuantumCircuit
from psiqit.quantum import pauli_z, tensor_product, identity

# Create a circuit
circ = QuantumCircuit(2)
circ.h(0).cx(0, 1)

# Get the final state
state = circ.run()

# Define observable
I = identity(2)
Z_I = tensor_product(pauli_z(), I)

# Compute expectation value
from psiqit.quantum import expectation
exp_val = expectation(state, Z_I)
print("⟨Z⊗I⟩:", exp_val)
```

---

## Probabilities from State Vector

You can extract probabilities directly from the state vector:

```python
from psiqit.circuits import QuantumCircuit
import numpy as np

circ = QuantumCircuit(2)
circ.h(0).cx(0, 1)

state = circ.run()

# Get probabilities
probabilities = np.abs(state.data)**2

print("Probabilities:")
for i, prob in enumerate(probabilities):
    binary = format(i, '02b')
    print(f"  |{binary}⟩: {prob:.4f}")
```

**Output:**
```text
Probabilities:
  |00⟩: 0.5000
  |01⟩: 0.0000
  |10⟩: 0.0000
  |11⟩: 0.5000
```

---

## Complete Example

Here's a comprehensive example demonstrating measurement concepts:

```python
from psiqit.circuits import QuantumCircuit
from psiqit.quantum import (
    pauli_z, pauli_x, pauli_y, 
    tensor_product, identity, expectation
)
import numpy as np

print("=" * 60)
print("Quantum Measurement Demo")
print("=" * 60)

# 1. Create a Bell state
print("\n1. Bell State Measurement")
print("-" * 60)
circ = QuantumCircuit(2)
circ.h(0).cx(0, 1)

result = circ.measure(shots=1000)
print("Counts:", result['counts'])

# 2. Expectation values
print("\n2. Expectation Values")
print("-" * 60)
state = circ.run()

I = identity(2)
Z_I = tensor_product(pauli_z(), I)
I_Z = tensor_product(I, pauli_z())
Z_Z = tensor_product(pauli_z(), pauli_z())

print("⟨Z⊗I⟩:", expectation(state, Z_I))
print("⟨I⊗Z⟩:", expectation(state, I_Z))
print("⟨Z⊗Z⟩:", expectation(state, Z_Z))

# 3. GHZ state
print("\n3. GHZ State (3 qubits)")
print("-" * 60)
circ_ghz = QuantumCircuit(3)
circ_ghz.h(0).cx(0, 1).cx(1, 2)

result_ghz = circ_ghz.measure(shots=1000)
print("Counts:", result_ghz['counts'])

# 4. Probability distribution
print("\n4. Probability Distribution")
print("-" * 60)
state_ghz = circ_ghz.run()
probs = np.abs(state_ghz.data)**2

for i, prob in enumerate(probs):
    if prob > 0.01:  # Only show non-zero probabilities
        binary = format(i, '03b')
        print(f"  |{binary}⟩: {prob:.4f}")

print("\n" + "=" * 60)
```

---

## Next Steps

Now that you understand measurement and observables, you're ready to:

- **[Tutorials](../tutorials/index.md)**: Apply these concepts in step-by-step algorithm implementations
- **[API Reference](../api/quantum.md)**: Complete API documentation for measurement functions
- **[Advanced Use Cases](../examples/index.md)**: Real-world applications of quantum measurement
