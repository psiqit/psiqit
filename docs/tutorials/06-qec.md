
# Tutorial 6: Quantum Error Correction (QEC)

In this tutorial, you'll implement the **3-qubit bit-flip code** - the simplest quantum error correction code that can detect and correct single bit-flip errors. :)

## Why Quantum Error Correction?

Quantum information is extremely fragile. Interactions with the environment cause errors that can destroy quantum computations. Unlike classical bits that can be protected by simple redundancy (copying), quantum states cannot be cloned due to the **No-Cloning Theorem**.

Quantum error correction solves this by encoding logical qubits into entangled states of multiple physical qubits, allowing errors to be detected and corrected without measuring (and thus destroying) the quantum information.

## The 3-Qubit Bit-Flip Code

The bit-flip code encodes one logical qubit into three physical qubits:

- Logical `|0⟩` → `|000⟩`
- Logical `|1⟩` → `|111⟩`

This code can correct any single bit-flip error (X gate applied to one qubit).

## Prerequisites

- PSIQIT installed (see [Installation Guide](../getting-started/installation.md))
- Understanding of quantum circuits (see [Building Circuits](../concepts/circuits.md))
- Familiarity with measurement (see [Measurement & Observables](../concepts/measurements.md))

---

## Step-by-Step Implementation

### Step 1: Import Required Modules

```python
import numpy as np
from psiqit.circuits import QuantumCircuit
```

### Step 2: Encoding

The encoding circuit copies the logical qubit state onto three physical qubits using CNOT gates.

```python
def encode_bit_flip(state_to_encode):
    """Encode a single qubit state into 3-qubit bit-flip code."""
    circ = QuantumCircuit(3)
    
    # Initialize qubit 0 with the state to encode
    if np.allclose(state_to_encode, [1, 0]):
        pass  # |0⟩ state - do nothing
    elif np.allclose(state_to_encode, [0, 1]):
        circ.x(0)  # |1⟩ state - apply X gate
    
    # Encode: |ψ⟩ → |ψψψ⟩
    circ.cx(0, 1)  # Copy qubit 0 to qubit 1
    circ.cx(0, 2)  # Copy qubit 0 to qubit 2
    
    return circ

# Test encoding |0⟩
circ_0 = encode_bit_flip([1, 0])
state_0 = circ_0.run()
print("Encoded |0⟩:", state_0.data)
# Output: [1, 0, 0, 0, 0, 0, 0, 0] = |000⟩

# Test encoding |1⟩
circ_1 = encode_bit_flip([0, 1])
state_1 = circ_1.run()
print("Encoded |1⟩:", state_1.data)
# Output: [0, 0, 0, 0, 0, 0, 0, 1] = |111⟩
```

!!! tip "No-Cloning Theorem"
    Notice that we're not "cloning" the state. The CNOT gates create an entangled state. For a superposition `α|0⟩ + β|1⟩`, the encoded state becomes `α|000⟩ + β|111⟩`, which is an entangled GHZ-like state, not three independent copies.

### Step 3: Error Injection

To test our code, we intentionally inject a bit-flip error (X gate) on one of the three qubits.

```python
def inject_error(error_qubit):
    """Create a circuit that encodes |0⟩ and then applies an X error."""
    circ = QuantumCircuit(3)
    
    # Encode |0⟩ → |000⟩
    circ.cx(0, 1)
    circ.cx(0, 2)
    
    # Apply error (X gate) to specified qubit
    circ.x(error_qubit)
    
    return circ

# Inject error on qubit 1
circ_error = inject_error(error_qubit=1)
state_error = circ_error.run()
print("State after error on qubit 1:", state_error.data)
# Output: [0, 0, 1, 0, 0, 0, 0, 0] = |010⟩
```

### Step 4: Syndrome Measurement

The syndrome measurement determines which qubit (if any) has an error, without revealing the encoded quantum information.

```python
def measure_syndrome(state_data):
    """Determine which qubit has the error based on the state."""
    # In LSB-first ordering:
    # |000⟩ = index 0 (no error)
    # |001⟩ = index 1 (error on qubit 0)
    # |010⟩ = index 2 (error on qubit 1)
    # |100⟩ = index 4 (error on qubit 2)
    
    probs = np.abs(state_data)**2
    error_index = np.argmax(probs)
    
    syndrome_map = {
        0: ("No error", None),
        1: ("Error on qubit 0", 0),
        2: ("Error on qubit 1", 1),
        4: ("Error on qubit 2", 2)
    }
    
    if error_index in syndrome_map:
        return syndrome_map[error_index]
    else:
        return (f"Unknown state (index {error_index})", None)

syndrome, error_qubit = measure_syndrome(state_error.data)
print(f"Syndrome: {syndrome}")
# Output: Syndrome: Error on qubit 1
```

!!! note "Syndrome Extraction"
    In a real quantum computer, the syndrome is extracted using ancilla qubits and parity-check measurements. Here, we use the state vector directly for educational clarity.

### Step 5: Recovery

Once the syndrome identifies the error location, we apply an X gate to the affected qubit to flip it back.

```python
def apply_recovery(error_qubit):
    """Create a circuit that encodes |0⟩, applies error, then corrects it."""
    circ = QuantumCircuit(3)
    
    # Encode
    circ.cx(0, 1)
    circ.cx(0, 2)
    
    # Inject error on qubit 1
    circ.x(1)
    
    # Apply correction
    if error_qubit is not None:
        circ.x(error_qubit)
    
    return circ

circ_recovered = apply_recovery(error_qubit=1)
state_recovered = circ_recovered.run()
print("State after recovery:", state_recovered.data)
# Output: [1, 0, 0, 0, 0, 0, 0, 0] = |000⟩ (original state restored!)
```

---

## Complete Code

Here is the complete, runnable script:

```python
"""
Tutorial 06: Quantum Error Correction (Bit-Flip Code)
Complete runnable example using PSIQIT.
"""

import numpy as np
from psiqit.circuits import QuantumCircuit

print("=" * 70)
print("TUTORIAL 06: QUANTUM ERROR CORRECTION (Bit-Flip Code)")
print("=" * 70)

# Syndrome measurement function
def measure_syndrome(state_data):
    probs = np.abs(state_data)**2
    error_index = np.argmax(probs)
    syndrome_map = {
        0: ("No error", None),
        1: ("Error on qubit 0", 0),
        2: ("Error on qubit 1", 1),
        4: ("Error on qubit 2", 2)
    }
    return syndrome_map.get(error_index, (f"Unknown (index {error_index})", None))

# Test all three error positions
print("\nEnd-to-End Error Correction Test:")
print("-" * 70)

for error_pos in [0, 1, 2]:
    # Encode |0⟩ → |000⟩
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
    
    status = "✓" if probs[0] > 0.99 else "✗"
    print(f"   {status} Error on qubit {error_pos}: detected as '{syndrome}' -> corrected")

print("\n" + "=" * 70)
print("Tutorial 06 completed!")
print("=" * 70)
```

---

## Test It Yourself

You can find the complete test script in the repository:

```bash
python examples/tutorials/test_06_qec.py
```

---

## Understanding the Results

### What the Bit-Flip Code Can Do

| Error Type | Correctable? | Example |
| :--- | :--- | :--- |
| Single bit-flip (X) | ✓ Yes | X on qubit 1 |
| No error | ✓ Yes | Identity |
| Two bit-flips | ✗ No | X on qubits 0 and 1 |
| Phase-flip (Z) | ✗ No | Z on qubit 0 |

!!! warning "Limitations"
    The 3-qubit bit-flip code can only correct **single bit-flip errors**. It cannot correct phase-flip errors (Z errors) or multiple simultaneous errors. For full error correction, more advanced codes like the **Shor code** (9 qubits) or **Steane code** (7 qubits) are needed.

---

## Common Mistakes

!!! warning "Qubit Ordering"
    Remember that PSIQIT uses **LSB-first** ordering. The state `|010⟩` corresponds to index `2` (not `4`), because qubit 1 is the second-least significant bit.

!!! warning "Measuring the Encoded State"
    Never directly measure the encoded qubits to detect errors - this would collapse the quantum information. In real QEC, syndrome extraction uses ancilla qubits and parity measurements.

---

## Next Steps

Now that you understand basic quantum error correction, you can explore:

- **[Advanced Use Cases](../examples/index.md)**: Real-world research applications
- **[API Reference](../api/error_correction.md)**: Complete documentation for QEC codes in PSIQIT
- **Shor Code**: A 9-qubit code that can correct both bit-flip and phase-flip errors

---

## References

- Nielsen, M.A. & Chuang, I.L. "Quantum Computation and Quantum Information" - Chapter 10
- Shor, P.W. "Scheme for reducing decoherence in quantum computer memory" (1995)
- [PSIQIT API Reference](../api/error_correction.md) - Complete documentation for QEC
