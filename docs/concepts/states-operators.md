
# States & Operators

This guide introduces the fundamental building blocks of quantum computing in PSIQIT: quantum states (`Ket`) and quantum operators (`Operator`). :)

## Quantum States (Ket)

In PSIQIT, quantum states are represented by the `Ket` class. A `Ket` wraps a NumPy complex array and provides methods for quantum operations.

### Creating Basic States

PSIQIT provides convenient functions for creating common quantum states:

```python
from psiqit.quantum import zero, one, plus, minus

# Computational basis states
q0 = zero()  # |0⟩
q1 = one()   # |1⟩

# Superposition states
q_plus = plus()   # |+⟩ = (|0⟩ + |1⟩) / √2
q_minus = minus() # |-⟩ = (|0⟩ - |1⟩) / √2
```

### Accessing State Data

You can access the underlying NumPy array using the `.data` attribute:

```python
print("State vector:", q0.data)
print("Shape:", q0.data.shape)
print("Type:", q0.data.dtype)
```

**Output:**
```text
State vector: [1.+0.j 0.+0.j]
Shape: (2,)
Type: complex128
```

### Creating Custom States

You can create custom quantum states by directly initializing a `Ket`:

```python
import numpy as np
from psiqit.quantum import Ket

# Create a custom state: (|0⟩ + i|1⟩) / √2
custom_state = Ket(np.array([1, 1j]) / np.sqrt(2))
print("Custom state:", custom_state.data)
```

### Multi-Qubit States

For multi-qubit systems, you can use tensor products:

```python
from psiqit.quantum import zero, one, tensor_product

# Create |00⟩
q00 = tensor_product(zero(), zero())

# Create |01⟩
q01 = tensor_product(zero(), one())

# Create |11⟩
q11 = tensor_product(one(), one())

print("|00⟩:", q00.data)
print("|01⟩:", q01.data)
print("|11⟩:", q11.data)
```

!!! tip "Entangled States"
    PSIQIT also provides built-in functions for creating entangled states like Bell states and GHZ states:
    ```python
    from psiqit.quantum import bell_phi_plus, ghz
    
    # Bell state: (|00⟩ + |11⟩) / √2
    bell = bell_phi_plus()
    
    # 3-qubit GHZ state: (|000⟩ + |111⟩) / √2
    ghz_state = ghz(3)
    ```

---

## Quantum Operators

Quantum operators (gates) are represented by the `Operator` class, which wraps a unitary matrix.

### Basic Gates

PSIQIT provides all standard quantum gates:

```python
from psiqit.quantum import (
    pauli_x, pauli_y, pauli_z,
    hadamard, s_gate, t_gate,
    cnot, cz, swap
)

# Single-qubit gates
X = pauli_x()    # Pauli-X (NOT gate)
Y = pauli_y()    # Pauli-Y
Z = pauli_z()    # Pauli-Z
H = hadamard()   # Hadamard gate
S = s_gate()     # S gate (√Z)
T = t_gate()     # T gate (⁴√Z)

# Two-qubit gates
CNOT = cnot()    # Controlled-NOT
CZ = cz()        # Controlled-Z
SWAP = swap()    # SWAP gate
```

### Accessing Operator Data

Like `Ket`, you can access the underlying matrix:

```python
print("Hadamard matrix:")
print(H.data)
print("\nShape:", H.data.shape)
```

**Output:**
```text
Hadamard matrix:
[[ 0.70710678  0.70710678]
 [ 0.70710678 -0.70710678]]

Shape: (2, 2)
```

### Rotation Gates

PSIQIT supports parameterized rotation gates:

```python
from psiqit.quantum import rx, ry, rz
import numpy as np

# Rotation around X-axis by π/4
Rx = rx(np.pi / 4)

# Rotation around Y-axis by π/2
Ry = ry(np.pi / 2)

# Rotation around Z-axis by π
Rz = rz(np.pi)
```

---

## Applying Operators to States

The most common operation in quantum computing is applying an operator (gate) to a state.

### Single-Qubit Operations

```python
from psiqit.quantum import zero, hadamard

# Start with |0⟩
q0 = zero()

# Apply Hadamard gate
H = hadamard()
q_plus = H @ q0  # or H.apply(q0)

print("After Hadamard:", q_plus.data)
# Output: [0.70710678+0.j 0.70710678+0.j]
```

!!! note "Operator Overloading"
    PSIQIT supports the `@` operator for applying gates to states, making the code more readable and Pythonic.

### Multi-Qubit Operations

For multi-qubit systems, you need to construct the full operator using tensor products:

```python
from psiqit.quantum import zero, one, hadamard, cnot, tensor_product, identity

# Create |00⟩
q00 = tensor_product(zero(), zero())

# Apply H to first qubit, I to second
H = hadamard()
I = identity(2)
H_on_first = tensor_product(H, I)

# Apply CNOT
CNOT = cnot()

# Build Bell state: CNOT @ (H ⊗ I) @ |00⟩
bell_state = CNOT @ H_on_first @ q00

print("Bell state:", bell_state.data)
# Output: [0.70710678+0.j 0.+0.j 0.+0.j 0.70710678+0.j]
```

!!! warning "Qubit Ordering"
    Remember that PSIQIT uses **LSB-first** ordering. When you write `tensor_product(A, B)`, operator `A` acts on the more significant qubit (left) and `B` acts on the less significant qubit (right).

---

## Inner Product and Fidelity

PSIQIT provides functions for computing inner products and fidelity between states.

### Inner Product

```python
from psiqit.quantum import zero, one, plus, inner_product

# ⟨0|0⟩ = 1
print("⟨0|0⟩:", inner_product(zero(), zero()))

# ⟨0|1⟩ = 0
print("⟨0|1⟩:", inner_product(zero(), one()))

# ⟨+|0⟩ = 1/√2
print("⟨+|0⟩:", inner_product(plus(), zero()))
```

### Fidelity

Fidelity measures how close two quantum states are (1.0 means identical):

```python
from psiqit.quantum import zero, plus, fidelity

# Fidelity between |0⟩ and |+⟩
fid = fidelity(zero(), plus())
print("Fidelity(|0⟩, |+⟩):", fid)
# Output: 0.5
```

---

## Expectation Values

The expectation value of an operator `O` for a state `|ψ⟩` is `⟨ψ|O|ψ⟩`:

```python
from psiqit.quantum import zero, pauli_z, expectation

# ⟨0|Z|0⟩ = 1
q0 = zero()
Z = pauli_z()
exp_val = expectation(q0, Z)
print("⟨0|Z|0⟩:", exp_val)
# Output: 1.0
```

---

## Complete Example

Here's a complete example that demonstrates states and operators:

```python
from psiqit.quantum import (
    zero, one, hadamard, cnot, 
    tensor_product, identity,
    inner_product, expectation, pauli_z
)

# 1. Create |00⟩
q00 = tensor_product(zero(), zero())
print("Initial state |00⟩:", q00.data)

# 2. Apply H to first qubit
H = hadamard()
I = identity(2)
H_I = tensor_product(H, I)
state1 = H_I @ q00
print("After H⊗I:", state1.data)

# 3. Apply CNOT
CNOT = cnot()
bell_state = CNOT @ state1
print("Bell state:", bell_state.data)

# 4. Compute expectation value of Z⊗I
Z_I = tensor_product(pauli_z(), I)
exp_val = expectation(bell_state, Z_I)
print("⟨Z⊗I⟩:", exp_val)

# 5. Check orthogonality
orthogonality = inner_product(bell_state, tensor_product(one(), one()))
print("⟨Bell|11⟩:", orthogonality)
```

---

## Next Steps

Now that you understand states and operators, you're ready to:

- **[Building Circuits](circuits.md)**: Learn how to use `QuantumCircuit` for easier gate application
- **[Measurement & Observables](measurements.md)**: Understand quantum measurement and POVMs
- **[Tutorials](../tutorials/index.md)**: Apply these concepts in step-by-step guides
