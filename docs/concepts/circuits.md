
# Building Circuits

This guide shows you how to build and execute quantum circuits using PSIQIT's `QuantumCircuit` class. :)

## Creating a Quantum Circuit

The `QuantumCircuit` class is the main interface for building quantum circuits in PSIQIT.

### Basic Initialization

```python
from psiqit.circuits import QuantumCircuit

# Create a circuit with 3 qubits
circ = QuantumCircuit(3)

print(f"Number of qubits: {circ.n_qubits}")
print(f"Number of gates: {circ.n_gates}")
```

**Output:**
```text
Number of qubits: 3
Number of gates: 0
```

---

## Applying Gates

PSIQIT provides a clean, intuitive API for applying quantum gates to circuits.

### Single-Qubit Gates

```python
from psiqit.circuits import QuantumCircuit

circ = QuantumCircuit(2)

# Apply gates to individual qubits
circ.h(0)      # Hadamard on qubit 0
circ.x(1)      # Pauli-X on qubit 1
circ.y(0)      # Pauli-Y on qubit 0
circ.z(1)      # Pauli-Z on qubit 1
circ.s(0)      # S gate on qubit 0
circ.t(1)      # T gate on qubit 1
```

### Method Chaining

PSIQIT supports method chaining for more concise code:

```python
circ = QuantumCircuit(2)

# Chain multiple gates
circ.h(0).x(1).y(0).z(1)

# Or all in one line
circ.h(0).cx(0, 1).measure_all()
```

!!! tip "Readability"
    Method chaining makes your code more readable and Pythonic. Use it when applying multiple gates in sequence.

### Two-Qubit Gates

```python
circ = QuantumCircuit(3)

# CNOT gate: control=0, target=1
circ.cx(0, 1)

# Controlled-Z gate: control=1, target=2
circ.cz(1, 2)

# SWAP gate: swap qubits 0 and 2
circ.swap(0, 2)
```

!!! warning "Qubit Ordering"
    Remember that PSIQIT uses **LSB-first** ordering. In `circ.cx(control, target)`, the control qubit is the first argument and the target is the second.

### Rotation Gates

```python
import numpy as np
from psiqit.circuits import QuantumCircuit

circ = QuantumCircuit(2)

# Rotation gates with angles
circ.rx(0, np.pi / 4)    # Rotate qubit 0 around X by π/4
circ.ry(1, np.pi / 2)    # Rotate qubit 1 around Y by π/2
circ.rz(0, np.pi)        # Rotate qubit 0 around Z by π
```

### Multi-Qubit Gates

```python
circ = QuantumCircuit(3)

# Toffoli gate (CCNOT): controls=[0,1], target=2
circ.toffoli(0, 1, 2)

# Fredkin gate (CSWAP): control=0, targets=[1,2]
circ.fredkin(0, 1, 2)
```

---

## Running Circuits

After building your circuit, you can execute it to get the final quantum state.

### Getting the State Vector

```python
from psiqit.circuits import QuantumCircuit

# Create a Bell state circuit
circ = QuantumCircuit(2)
circ.h(0).cx(0, 1)

# Run the circuit
state = circ.run()

print("State vector:", state.data)
print("Type:", type(state))
```

**Output:**
```text
State vector: [0.70710678+0.j 0.+0.j 0.+0.j 0.70710678+0.j]
Type: <class 'psiqit.quantum.state.Ket'>
```

!!! note "Return Type"
    The `run()` method returns a `Ket` object, which you can use for further quantum operations or analysis.

### Circuit with Parameters

For parameterized circuits (useful in variational algorithms):

```python
import numpy as np
from psiqit.circuits import QuantumCircuit

circ = QuantumCircuit(2)

# Apply rotation gates with parameters
theta = np.pi / 4
phi = np.pi / 2

circ.ry(0, theta)
circ.ry(1, phi)
circ.cx(0, 1)

state = circ.run()
print("Final state:", state.data)
```

---

## Measurement

PSIQIT provides flexible measurement capabilities.

### Measuring All Qubits

```python
from psiqit.circuits import QuantumCircuit

circ = QuantumCircuit(2)
circ.h(0).cx(0, 1)  # Bell state

# Measure all qubits with 1024 shots
result = circ.measure(shots=1024)

print("Counts:", result['counts'])
print("Probabilities:", result['probabilities'])
```

**Output:**
```text
Counts: {'00': 512, '11': 512}
Probabilities: {'00': 0.5, '11': 0.5}
```

### Measuring Specific Qubits

```python
circ = QuantumCircuit(3)
circ.h(0).cx(0, 1).h(2)

# Measure only qubits 0 and 1
result = circ.measure(qubits=[0, 1], shots=1000)

print("Counts:", result['counts'])
```

### Adding Measurement Gates to Circuit

You can also add measurement gates explicitly:

```python
circ = QuantumCircuit(2)
circ.h(0).cx(0, 1)

# Add measurement gates
circ.measure_qubit(0)
circ.measure_qubit(1)

# Or measure all at once
circ.measure_all()

# Then run
result = circ.run_with_measurement(shots=1024)
```

---

## Circuit Visualization

PSIQIT can draw your circuits as ASCII diagrams.

### Drawing a Circuit

```python
from psiqit.circuits import QuantumCircuit
from psiqit.visualization import draw_circuit

circ = QuantumCircuit(3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)

# Draw the circuit
print(draw_circuit(circ))
```

**Output:**
```text
q0: ── H ── ● ────────
q1: ──────── X ── ● ──
q2: ──────────── X ───
```

### Circuit Statistics

```python
from psiqit.visualization import circuit_statistics

circ = QuantumCircuit(3)
circ.h(0).cx(0, 1).cx(1, 2).h(2)

# Get circuit statistics
stats = circuit_statistics(circ)
print("Number of qubits:", stats['n_qubits'])
print("Number of gates:", stats['n_gates'])
print("Circuit depth:", stats['depth'])
print("Gate counts:", stats['gate_counts'])
```

**Output:**
```text
Number of qubits: 3
Number of gates: 4
Circuit depth: 4
Gate counts: {'H': 2, 'CX': 2}
```

---

## Common Circuit Patterns

### Bell State

```python
circ = QuantumCircuit(2)
circ.h(0).cx(0, 1)
state = circ.run()
```

### GHZ State

```python
circ = QuantumCircuit(3)
circ.h(0).cx(0, 1).cx(1, 2)
state = circ.run()
```

### Quantum Fourier Transform (QFT)

```python
from psiqit.algorithms import qft

circ = QuantumCircuit(3)
qft(circ, [0, 1, 2])
state = circ.run()
```

---

## Advanced Features

### Circuit Composition

You can combine multiple circuits:

```python
circ1 = QuantumCircuit(2)
circ1.h(0).cx(0, 1)

circ2 = QuantumCircuit(2)
circ2.x(0).y(1)

# Append circ2 to circ1
combined = circ1.compose(circ2)

state = combined.run()
```

### Circuit Inverse

Get the inverse (adjoint) of a circuit:

```python
circ = QuantumCircuit(2)
circ.h(0).cx(0, 1).s(1)

# Get inverse circuit
circ_inv = circ.inverse()

# Verify: circ @ circ_inv should give identity
combined = circ.compose(circ_inv)
state = combined.run()
```

### Barrier

Add barriers for visualization and optimization:

```python
circ = QuantumCircuit(3)
circ.h(0).cx(0, 1)
circ.barrier()  # Visual separator
circ.cx(1, 2).h(2)
```

---

## Complete Example

Here's a complete example that demonstrates circuit building:

```python
from psiqit.circuits import QuantumCircuit
from psiqit.visualization import draw_circuit, circuit_statistics
import numpy as np

# 1. Create a 3-qubit circuit
circ = QuantumCircuit(3)

# 2. Build a GHZ state
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)

# 3. Add some rotations
circ.ry(0, np.pi / 4)
circ.rz(1, np.pi / 2)

# 4. Draw the circuit
print("Circuit Diagram:")
print(draw_circuit(circ))

# 5. Get statistics
stats = circuit_statistics(circ)
print(f"\nCircuit Statistics:")
print(f"  Qubits: {stats['n_qubits']}")
print(f"  Gates: {stats['n_gates']}")
print(f"  Depth: {stats['depth']}")

# 6. Run and measure
state = circ.run()
print(f"\nFinal state: {state.data}")

result = circ.measure(shots=1000)
print(f"\nMeasurement counts: {result['counts']}")
```

---

## Next Steps

Now that you can build circuits, you're ready to:

- **[Measurement & Observables](measurements.md)**: Deep dive into quantum measurement theory
- **[Tutorials](../tutorials/index.md)**: Apply circuits in algorithm implementations
- **[API Reference](../api/circuits.md)**: Complete API documentation for `QuantumCircuit`
