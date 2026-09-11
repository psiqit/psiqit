
# Tutorial 2: Grover's Search Algorithm

In this tutorial, you'll implement **Grover's search algorithm** - a quantum algorithm that provides a quadratic speedup for unstructured search problems. :)

## What is Grover's Algorithm?

Grover's algorithm searches for a specific item in an unsorted database of N items using only O(√N) queries, compared to O(N) for classical algorithms.

For a 3-qubit system (N=8 items), Grover's algorithm finds the target in just **2 iterations**, while a classical search would need ~4 queries on average.

## The Algorithm

Grover's algorithm consists of:

1. **Initialization**: Create a uniform superposition of all states
2. **Oracle**: Mark the target state by flipping its phase
3. **Diffusion**: Amplify the amplitude of the marked state
4. **Repeat**: Steps 2-3 for approximately √N iterations
5. **Measure**: The target state now has the highest probability

## Prerequisites

- PSIQIT installed (see [Installation Guide](../getting-started/installation.md))
- Understanding of quantum gates (see [States & Operators](../concepts/states-operators.md))
- Basic circuit building skills (see [Building Circuits](../concepts/circuits.md))

---

## Step-by-Step Implementation

### Step 1: Import Required Modules

```python
from psiqit.algorithms import grover_search
import numpy as np
```

### Step 2: Define the Problem

Let's search for the state `|101⟩` (decimal: 5) in a 3-qubit database:

```python
n_qubits = 3
target = 5  # Binary: 101
shots = 1000

print(f"Searching for state |{format(target, f'0{n_qubits}b')}⟩ (decimal: {target})")
print(f"Database size: {2**n_qubits} items")
```

### Step 3: Calculate Optimal Iterations

The optimal number of iterations is approximately:

`iterations ≈ (π/4) * √N`

For N=8: `iterations ≈ (π/4) * √8 ≈ 2`

```python
optimal_iterations = int(np.pi / 4 * np.sqrt(2**n_qubits))
print(f"Optimal iterations: {optimal_iterations}")
```

### Step 4: Run Grover's Algorithm

PSIQIT provides a convenient built-in function for Grover's search:

```python
# Run Grover's algorithm
result = grover_search(n_qubits=n_qubits, target=target, shots=shots)

# Display results
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

!!! tip "Success Probability"
    With the optimal number of iterations, Grover's algorithm finds the target with approximately 94.5% probability. This is much better than the classical 12.5% (1/8) for a single random guess.

---

## Understanding the Results

### Why 94.5% and not 100%?

Grover's algorithm doesn't guarantee 100% success because the optimal number of iterations is not always an integer. The amplitude oscillates as we apply more iterations:

| Iterations | Success Probability |
| :--- | :--- |
| 0 | 12.5% (random guess) |
| 1 | 78.5% |
| 2 | 94.5% (optimal) |
| 3 | 86.2% (overshot) |
| 4 | 59.3% |

!!! warning "Too Many Iterations"
    Applying more iterations than optimal actually **decreases** the success probability. This is because the amplitude starts to "overshoot" and move away from the target state.

### Quadratic Speedup

For a database of N items:

| N (items) | Classical Queries | Grover Queries | Speedup |
| :--- | :--- | :--- | :--- |
| 8 | ~4 | 2 | 2x |
| 64 | ~32 | 6 | 5x |
| 1024 | ~512 | 25 | 20x |
| 1,000,000 | ~500,000 | 785 | 637x |

---

## Complete Code

Here's the complete script:

```python
from psiqit.algorithms import grover_search
import numpy as np

print("=" * 70)
print("GROVER'S SEARCH ALGORITHM")
print("=" * 70)

# Define problem
n_qubits = 3
target = 5  # Binary: 101

print(f"\nSearching for state |{format(target, f'0{n_qubits}b')}⟩ (decimal: {target})")
print(f"Database size: {2**n_qubits} items")

# Calculate optimal iterations
optimal_iterations = int(np.pi / 4 * np.sqrt(2**n_qubits))
print(f"Optimal iterations: {optimal_iterations}")

# Run Grover's algorithm
result = grover_search(n_qubits=n_qubits, target=target, shots=1000)

# Display results
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Found target: {result['most_likely']}")
print(f"Binary: {result['most_likely_binary']}")
print(f"Success probability: {result['probability_target']:.2%}")

# Verify success
if result['most_likely'] == target:
    print("\n✓ SUCCESS: Target found!")
else:
    print(f"\n✗ FAILED: Expected {target}, got {result['most_likely']}")

print("=" * 70)
```

---

## Test It Yourself

You can find the complete test script in the repository:

```bash
python examples/tutorials/test_02_grover_search.py
```

This will run the tutorial and verify that Grover's algorithm works correctly.

---

## Common Mistakes

!!! warning "Wrong Number of Iterations"
    Always calculate the optimal number of iterations using `(π/4) * √N`. Too few or too many iterations will reduce the success probability.

!!! warning "Oracle Implementation"
    The oracle must flip the phase of the target state **only**. A common mistake is to flip the phase of multiple states.

!!! warning "Qubit Ordering"
    Remember that PSIQIT uses **LSB-first** ordering. When you specify `target=5`, it corresponds to binary `101` where qubit 0 is the rightmost bit.

---

## Next Steps

Now that you've implemented Grover's search, you can:

- **[Tutorial 3: QFT](03-qft.md)**: Learn the Quantum Fourier Transform
- **[Tutorial 4: VQE](04-vqe.md)**: Explore variational quantum algorithms
- **[API Reference](../api/algorithms.md)**: Complete documentation for quantum algorithms

---

## References

- Grover, L.K. "A fast quantum mechanical algorithm for database search" (1996)
- Nielsen, M.A. & Chuang, I.L. "Quantum Computation and Quantum Information" - Chapter 6
- [PSIQIT API Reference](../api/algorithms.md) - Complete documentation for `grover_search`
