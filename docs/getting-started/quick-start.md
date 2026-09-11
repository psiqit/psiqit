

# Quick Start

Get up and running with PSIQIT in 5 minutes! This guide walks you through creating your first quantum circuit, running an algorithm, and visualizing the results. :)

## Prerequisites

Before starting, make sure you have:

- Python 3.8+ installed
- PSIQIT installed (see the [Installation Guide](installation.md))

You can verify your installation:

```bash
python -c "import psiqit; print(psiqit.__version__)"
```


## Your First Quantum Circuit

Let's create a simple 2-qubit circuit that generates a **Bell state** (a maximally entangled state).

### Step 1: Import PSIQIT

```python
from psiqit.circuits import QuantumCircuit
```

### Step 2: Create a Circuit

Initialize a quantum circuit with 2 qubits:

```python
circ = QuantumCircuit(2)
```

### Step 3: Apply Gates

Apply a Hadamard gate to qubit 0, then a CNOT gate between qubits 0 and 1:

```python
circ.h(0)      # Put qubit 0 in superposition
circ.cx(0, 1)  # Entangle qubit 0 with qubit 1
```

!!! tip "Method Chaining"
    PSIQIT supports method chaining for a more concise syntax:
    ```python
    circ.h(0).cx(0, 1)
    ```

### Step 4: Run the Circuit

Execute the circuit to obtain the final quantum state:

```python
state = circ.run()
print("State Vector:")
print(state.data)
```

**Expected Output:**
```text
State Vector:
[0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
```

This represents the Bell state:

$$
|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)
$$

### Step 5: Measure the Circuit

Simulate measurements to see the probability distribution:

```python
result = circ.measure(shots=1024)
print("\nMeasurement Counts:")
print(result['counts'])
```

**Expected Output:**
```text
Measurement Counts:
{'00': 512, '11': 512}
```

You'll see approximately 50% of measurements yield `00` and 50% yield `11`, demonstrating quantum entanglement! :)

---

## Running Your First Algorithm

PSIQIT includes built-in implementations of famous quantum algorithms. Let's run **Grover's search algorithm**.

### Problem Setup

Search for the state $|101\rangle$ (decimal: 5) in a 3-qubit database:

```python
from psiqit.algorithms import grover_search

# Search for target state 5 (binary: 101) in 3-qubit space
result = grover_search(n_qubits=3, target=5, shots=1000)

print(f"Found target: {result['most_likely']}")
print(f"Binary: {result['most_likely_binary']}")
print(f"Success probability: {result['probability_target']:.2%}")
```

**Expected Output:**
```text
Found target: 5
Binary: 101
Success probability: 94.53%
```

!!! note "Quadratic Speedup"
    Grover's algorithm finds the target state with ~94.5% probability using only 2 iterations, compared to classical search which would require checking ~4 states on average.

---

## Exploring Open Quantum Systems

PSIQIT can simulate **open quantum systems** using the Lindblad master equation. Let's model a qubit undergoing decoherence.

### Setup

```python
from psiqit.quantum import pauli_z, pauli_x
from psiqit.dynamics import LindbladSolver

# Define Hamiltonian (qubit in a magnetic field)
H = pauli_z()

# Define collapse operator (spontaneous emission)
L = [pauli_x()]

# Create solver with decay rate gamma = 0.1
solver = LindbladSolver(H, L, gamma=[0.1])
```

### Solve the Dynamics

```python
# Evolve the system from t=0 to t=10
result = solver.solve(t_span=(0, 10), dt=0.01)

# Extract final state
final_state = result['states'][-1]
print("Final density matrix:")
print(final_state.data)
```

This simulates how a qubit loses coherence over time due to interaction with its environment.

---

## Visualizing Results

PSIQIT includes visualization tools for quantum states and circuits.

### Draw a Circuit

```python
from psiqit.circuits import QuantumCircuit
from psiqit.visualization import draw_circuit

circ = QuantumCircuit(2)
circ.h(0).cx(0, 1)

# Generate ASCII circuit diagram
print(draw_circuit(circ))
```

**Expected Output:**
```text
q0: ── H ── ● ──
q1: ──────── X ──
```

---

## Complete Example

Here's the complete code from this quick start guide in a single script:

```python
from psiqit.circuits import QuantumCircuit
from psiqit.algorithms import grover_search
from psiqit.visualization import draw_circuit

# 1. Bell State Circuit
print("=== Bell State ===")
circ = QuantumCircuit(2)
circ.h(0).cx(0, 1)
state = circ.run()
print("State:", state.data)

result = circ.measure(shots=1024)
print("Counts:", result['counts'])
print("\nCircuit Diagram:")
print(draw_circuit(circ))

# 2. Grover's Algorithm
print("\n=== Grover's Search ===")
grover_result = grover_search(n_qubits=3, target=5, shots=1000)
print(f"Found: {grover_result['most_likely']} ({grover_result['most_likely_binary']})")
print(f"Probability: {grover_result['probability_target']:.2%}")
```

Run this script to see all the examples in action! :)

---

## What's Next?

Now that you've run your first circuits and algorithms, explore:

- **[Architecture & Design](architecture.md)**: Learn about PSIQIT's LSB-first qubit ordering
- **[Core Concepts](../concepts/states-operators.md)**: Deep dive into states and operators
- **[Tutorials](../tutorials/index.md)**: Step-by-step guides for advanced topics
- **[API Reference](../api/index.md)**: Detailed documentation of all modules
