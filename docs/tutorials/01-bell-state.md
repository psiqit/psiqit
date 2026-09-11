
# Tutorial 1: Creating a Bell State

In this tutorial, you'll learn how to create and measure a **Bell state** - one of the most fundamental entangled states in quantum computing. :)

## What is a Bell State?

A Bell state is a maximally entangled two-qubit state. The most common Bell state is:

`|Φ⁺⟩ = (|00⟩ + |11⟩) / √2`

This means when you measure both qubits, you'll always get the same result: either both `0` or both `1`, each with 50% probability.

## Prerequisites

- PSIQIT installed (see [Installation Guide](../getting-started/installation.md))
- Basic understanding of quantum gates (see [States & Operators](../concepts/states-operators.md))

---

## Step-by-Step Guide

### Step 1: Import Required Modules

```python
from psiqit.circuits import QuantumCircuit
from psiqit.visualization import draw_circuit
import numpy as np
```

### Step 2: Create a Quantum Circuit

Initialize a circuit with 2 qubits:

```python
circ = QuantumCircuit(2)
print(f"Circuit with {circ.n_qubits} qubits created")
```

### Step 3: Apply Hadamard Gate

The Hadamard gate puts qubit 0 into a superposition state:

```python
circ.h(0)  # Apply H to qubit 0
```

After this step, qubit 0 is in the state `|+⟩ = (|0⟩ + |1⟩) / √2`.

### Step 4: Apply CNOT Gate

The CNOT (Controlled-NOT) gate entangles the two qubits:

```python
circ.cx(0, 1)  # Control: qubit 0, Target: qubit 1
```

This creates the Bell state `|Φ⁺⟩`.

!!! tip "Understanding CNOT"
    The CNOT gate flips the target qubit (qubit 1) if and only if the control qubit (qubit 0) is in state `|1⟩`. Combined with the superposition from the Hadamard gate, this creates entanglement.

### Step 5: Visualize the Circuit

```python
print(draw_circuit(circ))
```

**Output:**
```text
q0: ── H ── ● ──
q1: ──────── X ──
```

### Step 6: Get the State Vector

Run the circuit to obtain the final quantum state:

```python
state = circ.run()
print("State vector:", state.data)
```

**Output:**
```text
State vector: [0.70710678+0.j 0.+0.j 0.+0.j 0.70710678+0.j]
```

This represents `(1/√2)|00⟩ + (1/√2)|11⟩`.

### Step 7: Calculate Probabilities

```python
probabilities = np.abs(state.data)**2
for i, prob in enumerate(probabilities):
    if prob > 0.01:
        binary = format(i, '02b')
        print(f"|{binary}⟩: {prob:.4f}")
```

**Output:**
```text
|00⟩: 0.5000
|11⟩: 0.5000
```

### Step 8: Measure the Circuit

Simulate measurements to see the statistical distribution:

```python
result = circ.measure(shots=1024)
print("Counts:", result['counts'])
```

**Typical Output:**
```text
Counts: {'00': 512, '11': 512}
```

!!! note "Statistical Variation"
    The exact numbers will vary each time you run the measurement due to the probabilistic nature of quantum mechanics. However, you should always see approximately 50% `00` and 50% `11`, with almost no `01` or `10`.

---

## Complete Code

Here's the complete script:

```python
from psiqit.circuits import QuantumCircuit
from psiqit.visualization import draw_circuit
import numpy as np

# Create circuit
circ = QuantumCircuit(2)

# Build Bell state
circ.h(0)
circ.cx(0, 1)

# Visualize
print("Circuit:")
print(draw_circuit(circ))

# Get state vector
state = circ.run()
print("\nState vector:", state.data)

# Calculate probabilities
probabilities = np.abs(state.data)**2
print("\nProbabilities:")
for i, prob in enumerate(probabilities):
    if prob > 0.01:
        binary = format(i, '02b')
        print(f"  |{binary}⟩: {prob:.4f}")

# Measure
result = circ.measure(shots=1024)
print("\nMeasurement counts:", result['counts'])
```

---

## Test It Yourself

You can find the complete test script in the repository:

```bash
python examples/tutorials/test_01_bell_state.py
```

This will run the tutorial and verify that the Bell state is created correctly.

---

## Understanding the Results

### Why is this important?

1. **Entanglement**: The Bell state demonstrates quantum entanglement - the qubits are correlated in a way that's impossible classically.

2. **Perfect Correlation**: When you measure both qubits, they always give the same result. If qubit 0 is `0`, qubit 1 is guaranteed to be `0`. If qubit 0 is `1`, qubit 1 is guaranteed to be `1`.

3. **Foundation for Quantum Computing**: Bell states are the building blocks for:
   - Quantum teleportation
   - Superdense coding
   - Quantum error correction
   - Quantum cryptography

---

## Common Mistakes

!!! warning "Wrong Qubit Order"
    Remember that PSIQIT uses **LSB-first** ordering. The state `|01⟩` means qubit 0 is `1` and qubit 1 is `0`.

!!! warning "Forgetting to Run"
    Don't forget to call `circ.run()` before measuring. The circuit needs to be executed to produce the final state.

---

## Next Steps

Now that you've created a Bell state, you can:

- **[Tutorial 2: Grover's Search](02-grover-search.md)**: Learn how to use entanglement in quantum algorithms
- **[Tutorial 3: QFT](03-qft.md)**: Explore the Quantum Fourier Transform
- **[Core Concepts](../concepts/index.md)**: Deepen your understanding of quantum states and operators

---

## References

- Nielsen, M.A. & Chuang, I.L. "Quantum Computation and Quantum Information" - Chapter 2
- [PSIQIT API Reference](../api/circuits.md) - Complete documentation for `QuantumCircuit`
