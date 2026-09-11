
# Tutorials Overview

Welcome to the PSIQIT tutorials! This section provides step-by-step guides to help you master quantum computing concepts using PSIQIT. :)

## Learning Path

We recommend following the tutorials in order, as each builds upon concepts from the previous ones:

1. **[Tutorial 01: Bell State](01-bell-state.md)** → 2. **[Tutorial 02: Grover](02-grover-search.md)** → 3. **[Tutorial 03: QFT](03-qft.md)** → 4. **[Tutorial 04: VQE](04-vqe.md)** → 5. **[Tutorial 05: Lindblad](05-lindblad.md)** → 6. **[Tutorial 06: QEC](06-qec.md)**

## Available Tutorials

| # | Tutorial | Level | Topics | Time |
| :--- | :--- | :--- | :--- | :--- |
| 01 | [Creating a Bell State](01-bell-state.md) | Beginner | Entanglement, Superposition, Measurement | 10 min |
| 02 | [Grover's Search Algorithm](02-grover-search.md) | Intermediate | Quantum Algorithms, Oracle, Amplitude Amplification | 20 min |
| 03 | [Quantum Fourier Transform](03-qft.md) | Intermediate | QFT, Phase Estimation, Quantum Circuits | 25 min |
| 04 | [Variational Quantum Eigensolver](04-vqe.md) | Advanced | Hybrid Algorithms, Optimization, Quantum Chemistry | 30 min |
| 05 | [Lindblad Dynamics](05-lindblad.md) | Advanced | Open Systems, Decoherence, Density Matrices | 25 min |
| 06 | [Quantum Error Correction](06-qec.md) | Advanced | QEC, Syndrome Measurement, Fault Tolerance | 30 min |

## Prerequisites

Before starting the tutorials, make sure you have:

!!! note "Required Setup"
    - Python 3.8 or higher installed
    - PSIQIT installed (see [Installation Guide](../getting-started/installation.md))
    - Basic understanding of linear algebra (vectors, matrices)
    - Familiarity with quantum computing concepts (see [Core Concepts](../concepts/states-operators.md))

## Tutorial Summaries

### 01. Creating a Bell State

**What you'll learn:**
- How to create and manipulate quantum circuits
- Understanding quantum entanglement
- Measuring quantum states and interpreting results

**Key concepts:** Hadamard gate, CNOT gate, Bell states, quantum measurement

**Why it matters:** Bell states are the foundation of quantum communication, teleportation, and many quantum algorithms.

👉 [Start Tutorial 01 →](01-bell-state.md)

---

### 02. Grover's Search Algorithm

**What you'll learn:**
- Implementing quantum algorithms from scratch
- Understanding quantum speedup
- Working with oracles and diffusion operators

**Key concepts:** Unstructured search, quadratic speedup, amplitude amplification, oracle design

**Why it matters:** Grover's algorithm demonstrates quantum advantage for search problems and is a building block for more complex algorithms.

👉 [Start Tutorial 02 →](02-grover-search.md)

---

### 03. Quantum Fourier Transform (QFT)

**What you'll learn:**
- Implementing QFT circuits
- Understanding phase encoding in quantum states
- Applications in quantum algorithms

**Key concepts:** Fourier transform, controlled rotations, phase estimation, Shor's algorithm foundation

**Why it matters:** QFT is essential for Shor's factoring algorithm, quantum phase estimation, and many other quantum algorithms.

👉 [Start Tutorial 03 →](03-qft.md)

---

### 04. Variational Quantum Eigensolver (VQE)

**What you'll learn:**
- Building hybrid quantum-classical algorithms
- Parameterized quantum circuits (ansatz)
- Classical optimization of quantum circuits

**Key concepts:** Variational principle, cost functions, gradient-based optimization, quantum chemistry applications

**Why it matters:** VQE is one of the most promising near-term quantum algorithms for solving real-world optimization and chemistry problems.

👉 [Start Tutorial 04 →](04-vqe.md)

---

### 05. Lindblad Dynamics

**What you'll learn:**
- Simulating open quantum systems
- Modeling decoherence and dissipation
- Working with density matrices

**Key concepts:** Lindblad master equation, collapse operators, decoherence, quantum trajectories

**Why it matters:** Real quantum systems interact with their environment. Understanding open systems is crucial for designing robust quantum algorithms and error correction schemes.

👉 [Start Tutorial 05 →](05-lindblad.md)

---

### 06. Quantum Error Correction (QEC)

**What you'll learn:**
- Implementing basic error correction codes
- Syndrome measurement and recovery
- Understanding fault-tolerant quantum computing

**Key concepts:** Bit-flip code, syndrome extraction, logical qubits, fault tolerance

**Why it matters:** Error correction is essential for building scalable, reliable quantum computers. This tutorial introduces the fundamental concepts behind protecting quantum information.

👉 [Start Tutorial 06 →](06-qec.md)

---

## Learning Tracks

Depending on your goals, you can follow different learning paths:

### 🎓 Academic Track
Focus on fundamental concepts and algorithms:

**Path:** 01 → 02 → 03 → 06

**Best for:** Students, researchers, and anyone wanting to understand quantum computing theory.

### 💼 Industry Track
Focus on practical applications and near-term algorithms:

**Path:** 01 → 04 → 05

**Best for:** Developers, engineers, and professionals working on quantum applications.

### 🔬 Research Track
Comprehensive coverage for advanced research:

**Path:** All tutorials in order (01 → 06)

**Best for:** PhD students, postdocs, and researchers pushing the boundaries of quantum computing.

---

## Additional Resources

!!! tip "Need More Help?"
    - Check the [API Reference](../api/index.md) for detailed documentation
    - Explore [Advanced Use Cases](../examples/index.md) for real-world applications
    - Join our [Community](../contributing.md) to ask questions and share your work

!!! note "Prerequisite Knowledge"
    If you're new to quantum computing, we recommend reading the [Core Concepts](../concepts/states-operators.md) section first to build a solid foundation.

---

## What's Next?

Ready to start? Begin with [Tutorial 01: Creating a Bell State](01-bell-state.md) and work your way through the learning path!

!!! success "Pro Tip"
    Each tutorial includes complete, runnable code. Don't just read - execute the code, modify it, and experiment!
