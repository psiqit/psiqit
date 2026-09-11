# psiqit/qml/quantum_kernel.py 

"""
Quantum Kernel Module
Quantum kernel methods for machine learning
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, zero, one, plus, minus, basis
from ..quantum.operator import Operator, identity, pauli_x, pauli_y, pauli_z, hadamard, rx, ry, rz, cnot, cz, swap
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# KERNEL RESULT CLASS
# ============================================================================

@dataclass
class KernelResult:
    """
    Result container for kernel evaluation
    
    Attributes:
        value: Kernel value
        fidelity: Fidelity between quantum states
        distance: Distance between feature vectors
        shots: Number of shots used
        n_qubits: Number of qubits
        method: Method used for evaluation
    """
    value: float = 0.0
    fidelity: float = 0.0
    distance: float = 0.0
    shots: int = 1024
    n_qubits: int = 0
    method: str = ""
    
    def __repr__(self) -> str:
        return f"KernelResult(value={self.value:.6f}, fidelity={self.fidelity:.6f})"
    
    def __str__(self) -> str:
        lines = [
            "Kernel Evaluation Results:",
            f"  Value: {self.value:.6f}",
            f"  Fidelity: {self.fidelity:.6f}",
            f"  Distance: {self.distance:.6f}",
            f"  Shots: {self.shots}",
            f"  Qubits: {self.n_qubits}",
            f"  Method: {self.method}",
        ]
        return "\n".join(lines)


# ============================================================================
# QUANTUM KERNEL
# ============================================================================

class QuantumKernel:
    """
    Quantum Kernel for machine learning
    
    Implements quantum feature maps and kernel functions for use in
    quantum machine learning algorithms.
    
    Example:
        >>> from psiqit.qml import QuantumKernel
        >>> # Create quantum kernel
        >>> kernel = QuantumKernel(n_qubits=2, feature_map='zz', n_layers=2)
        >>> # Evaluate kernel between two data points
        >>> x1 = np.array([0.5, 0.3])
        >>> x2 = np.array([0.2, 0.7])
        >>> result = kernel.evaluate(x1, x2)
        >>> print(result.value)
    """
    
    def __init__(
        self,
        n_qubits: int,
        feature_map: str = 'zz',
        n_layers: int = 2
    ):
        """
        Initialize Quantum Kernel
        
        Args:
            n_qubits: Number of qubits
            feature_map: Type of feature map ('zz', 'xy', 'custom')
            n_layers: Number of layers for the feature map
        """
        validate_qubits(n_qubits)
        
        self.n_qubits = n_qubits
        self.feature_map = feature_map
        self.n_layers = n_layers
        
        # Build feature map function
        self._feature_map_fn = self._build_feature_map()
        
        logger.info(f"QuantumKernel initialized: {n_qubits} qubits, feature_map={feature_map}, layers={n_layers}")
    
    def _build_feature_map(self) -> Callable:
        """
        Build the feature map function
        
        Returns:
            Callable: Feature map function that takes a circuit and data
            
        Example:
            >>> feature_map = kernel._build_feature_map()
            >>> circ = QuantumCircuit(2)
            >>> feature_map(circ, np.array([0.5, 0.3]))
        """
        if self.feature_map == 'zz':
            return self._zz_feature_map
        elif self.feature_map == 'xy':
            return self._xy_feature_map
        elif self.feature_map == 'custom':
            return self._custom_feature_map
        else:
            raise ValueError(f"Unknown feature_map: {self.feature_map}")
    
    def _zz_feature_map(self, circuit: QuantumCircuit, x: np.ndarray):
        """
        ZZ Feature Map
        
        Encodes data using ZZ interactions:
        U_Φ(x) = exp(i Σ φ(x_i) Z_i + i Σ φ(x_i, x_j) Z_i Z_j)
        
        Args:
            circuit: QuantumCircuit object
            x: Feature vector
        """
        n_features = min(len(x), self.n_qubits)
        
        # Single-qubit rotations
        for i in range(n_features):
            circuit.ry(i, x[i] * np.pi)
        
        # ZZ interactions
        for layer in range(self.n_layers):
            for i in range(n_features - 1):
                angle = x[i] * x[i + 1] * np.pi
                circuit.cz(i, i + 1)
                circuit.rz(i + 1, angle)
                circuit.cz(i, i + 1)
            
            # Additional rotations after each layer
            for i in range(n_features):
                circuit.ry(i, x[i] * np.pi / 2)
    
    def _xy_feature_map(self, circuit: QuantumCircuit, x: np.ndarray):
        """
        XY Feature Map
        
        Encodes data using XY interactions:
        U_Φ(x) = exp(i Σ φ(x_i) X_i + i Σ φ(x_i, x_j) X_i X_j + ...)
        
        Args:
            circuit: QuantumCircuit object
            x: Feature vector
        """
        n_features = min(len(x), self.n_qubits)
        
        # Single-qubit rotations
        for i in range(n_features):
            circuit.rx(i, x[i] * np.pi)
            circuit.ry(i, x[i] * np.pi)
        
        # XY interactions
        for layer in range(self.n_layers):
            for i in range(n_features - 1):
                angle = x[i] * x[i + 1] * np.pi
                circuit.cx(i, i + 1)
                circuit.ry(i + 1, angle)
                circuit.cx(i, i + 1)
    
    def _custom_feature_map(self, circuit: QuantumCircuit, x: np.ndarray):
        """
        Custom Feature Map (placeholder for user-defined)
        
        Args:
            circuit: QuantumCircuit object
            x: Feature vector
        """
        # Default: simple angle encoding
        n_features = min(len(x), self.n_qubits)
        for i in range(n_features):
            circuit.ry(i, x[i] * np.pi)
    
    def _encode_data(self, x: np.ndarray) -> QuantumCircuit:
        """
        Encode data into a quantum circuit using the feature map
        
        Args:
            x: Feature vector
            
        Returns:
            QuantumCircuit: Encoded quantum circuit
            
        Example:
            >>> circ = kernel._encode_data(np.array([0.5, 0.3]))
            >>> print(circ.draw())
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        circuit = QuantumCircuit(self.n_qubits)
        
        # Apply feature map
        self._feature_map_fn(circuit, x[0])
        
        return circuit
    
    def evaluate(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        shots: int = 1024
    ) -> KernelResult:
        """
        Evaluate the quantum kernel between two data points
        
        K(x1, x2) = |⟨ψ(x1)|ψ(x2)⟩|²
        
        Args:
            x1: First feature vector
            x2: Second feature vector
            shots: Number of shots for estimation
            
        Returns:
            KernelResult: Kernel evaluation results
            
        Example:
            >>> result = kernel.evaluate(x1, x2)
            >>> print(result.value)
        """
        # Encode data into quantum circuits
        circ1 = self._encode_data(x1)
        circ2 = self._encode_data(x2)
        
        # Get states
        state1 = circ1.run()
        state2 = circ2.run()
        
        # Compute fidelity
        fidelity = abs(np.vdot(state1.data, state2.data)) ** 2
        
        # For quantum kernel, we use the fidelity
        # In practice, this would be estimated using the SWAP test
        kernel_value = float(fidelity)
        
        # Compute distance
        distance = np.linalg.norm(x1 - x2)
        
        logger.debug(f"Kernel evaluated: value={kernel_value:.4f}, fidelity={fidelity:.4f}")
        
        return KernelResult(
            value=kernel_value,
            fidelity=fidelity,
            distance=distance,
            shots=shots,
            n_qubits=self.n_qubits,
            method=f"quantum_{self.feature_map}"
        )
    
    def kernel_matrix(
        self,
        X: np.ndarray,
        shots: int = 1024
    ) -> List[List[float]]:
        """
        Compute the full kernel matrix for a dataset
        
        Args:
            X: Data matrix (n_samples x n_features)
            shots: Number of shots for estimation
            
        Returns:
            List[List[float]]: Kernel matrix
            
        Example:
            >>> K = kernel.kernel_matrix(X)
            >>> print(K[0][1])  # Kernel between sample 0 and 1
        """
        n_samples = len(X)
        K = np.zeros((n_samples, n_samples))
        
        # Pre-compute all states
        states = []
        for i in range(n_samples):
            circ = self._encode_data(X[i])
            state = circ.run()
            states.append(state)
        
        # Compute kernel matrix
        for i in range(n_samples):
            for j in range(i, n_samples):
                fidelity = abs(np.vdot(states[i].data, states[j].data)) ** 2
                K[i, j] = fidelity
                K[j, i] = fidelity
        
        logger.info(f"Kernel matrix computed: {n_samples}x{n_samples}")
        
        return K.tolist()
    
    def kernel_alignment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shots: int = 1024
    ) -> float:
        """
        Compute kernel alignment with labels
        
        Alignment = ⟨K, yyᵀ⟩ / (||K|| * ||yyᵀ||)
        
        Args:
            X: Data matrix
            y: Labels
            shots: Number of shots for estimation
            
        Returns:
            float: Kernel alignment (0-1)
            
        Example:
            >>> alignment = kernel.kernel_alignment(X, y)
            >>> print(alignment)
        """
        # Compute kernel matrix
        K = np.array(self.kernel_matrix(X, shots))
        
        # Compute label matrix
        y = np.array(y)
        y_matrix = np.outer(y, y)
        
        # Compute alignment
        numerator = np.sum(K * y_matrix)
        denominator = np.linalg.norm(K) * np.linalg.norm(y_matrix)
        
        if denominator == 0:
            return 0.0
        
        alignment = numerator / denominator
        
        logger.info(f"Kernel alignment: {alignment:.4f}")
        
        return float(alignment)
    
    def get_feature_map_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """
        Get the feature map circuit for a data point
        
        Args:
            x: Feature vector
            
        Returns:
            QuantumCircuit: Feature map circuit
            
        Example:
            >>> circ = kernel.get_feature_map_circuit(x)
            >>> print(circ.draw())
        """
        return self._encode_data(x)
    
    def get_state(self, x: np.ndarray) -> Ket:
        """
        Get the quantum state for a data point
        
        Args:
            x: Feature vector
            
        Returns:
            Ket: Quantum state
            
        Example:
            >>> state = kernel.get_state(x)
            >>> print(state)
        """
        circ = self._encode_data(x)
        return circ.run()
    
    def __repr__(self) -> str:
        return f"QuantumKernel(n_qubits={self.n_qubits}, feature_map={self.feature_map}, layers={self.n_layers})"
    
    def __str__(self) -> str:
        lines = [
            "Quantum Kernel:",
            f"  Qubits: {self.n_qubits}",
            f"  Feature Map: {self.feature_map}",
            f"  Layers: {self.n_layers}",
        ]
        return "\n".join(lines)


# ============================================================================
# QUANTUM KERNEL ESTIMATOR
# ============================================================================

class QuantumKernelEstimator:
    """
    Quantum Kernel Estimator using the SWAP test
    
    Estimates the kernel (fidelity) between two quantum states
    using the SWAP test circuit.
    
    Example:
        >>> from psiqit.qml import QuantumKernelEstimator
        >>> estimator = QuantumKernelEstimator(n_qubits=2)
        >>> state1 = Ket([1, 0, 0, 0])
        >>> state2 = Ket([0, 1, 0, 0])
        >>> fidelity = estimator.swap_test(state1, state2)
        >>> print(fidelity)
    """
    
    def __init__(self, n_qubits: int):
        """
        Initialize Quantum Kernel Estimator
        
        Args:
            n_qubits: Number of qubits
        """
        validate_qubits(n_qubits)
        self.n_qubits = n_qubits
        
        logger.info(f"QuantumKernelEstimator initialized: {n_qubits} qubits")
    
    def swap_test(
        self,
        state1: Union[Ket, np.ndarray],
        state2: Union[Ket, np.ndarray],
        shots: int = 1024
    ) -> float:
        """
        Estimate fidelity between two states using the SWAP test
        
        The SWAP test estimates |⟨ψ|φ⟩|² by measuring the probability
        of the ancilla qubit being in the |0⟩ state.
        
        P(ancilla=0) = (1 + |⟨ψ|φ⟩|²) / 2
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            shots: Number of measurements
            
        Returns:
            float: Estimated fidelity (0-1)
            
        Example:
            >>> state1 = Ket([1, 0])
            >>> state2 = Ket([1/np.sqrt(2), 1/np.sqrt(2)])
            >>> fidelity = estimator.swap_test(state1, state2)
            >>> print(fidelity)  # 0.5
        """
        # Convert to Ket if needed
        if isinstance(state1, np.ndarray):
            state1 = Ket(state1)
        if isinstance(state2, np.ndarray):
            state2 = Ket(state2)
        
        # Check dimensions
        if state1.dim != state2.dim:
            raise ValueError(f"State dimensions mismatch: {state1.dim} vs {state2.dim}")
        
        if state1.dim != 2 ** self.n_qubits:
            raise ValueError(f"State dimension {state1.dim} does not match 2^{self.n_qubits}")
        
        # Build SWAP test circuit
        # The SWAP test requires 2n + 1 qubits (ancilla + two registers)
        total_qubits = 2 * self.n_qubits + 1
        circuit = QuantumCircuit(total_qubits)
        
        # Prepare states
        # This is a simplified version - in practice, we would need to
        # prepare the states using state preparation circuits
        
        # For simulation, we compute the fidelity directly
        # In a real quantum computer, we would use the SWAP test circuit
        
        # Compute fidelity directly (for simulation)
        fidelity = abs(np.vdot(state1.data, state2.data)) ** 2
        
        # Simulate SWAP test measurement
        # P(ancilla=0) = (1 + fidelity) / 2
        prob_zero = (1 + fidelity) / 2
        
        # Sample measurements
        outcomes = np.random.choice([0, 1], size=shots, p=[prob_zero, 1 - prob_zero])
        count_zero = np.sum(outcomes == 0)
        
        # Estimate fidelity from measurements
        estimated_fidelity = 2 * (count_zero / shots) - 1
        
        logger.debug(f"SWAP test: true={fidelity:.4f}, estimated={estimated_fidelity:.4f}, shots={shots}")
        
        return float(max(0, min(1, estimated_fidelity)))
    
    def kernel_matrix_swap(
        self,
        states: List[Ket],
        shots: int = 1024
    ) -> List[List[float]]:
        """
        Compute kernel matrix using SWAP test
        
        Args:
            states: List of quantum states
            shots: Number of shots for each pair
            
        Returns:
            List[List[float]]: Kernel matrix
            
        Example:
            >>> states = [state1, state2, state3]
            >>> K = estimator.kernel_matrix_swap(states)
            >>> print(K)
        """
        n_states = len(states)
        K = np.zeros((n_states, n_states))
        
        for i in range(n_states):
            for j in range(i, n_states):
                fidelity = self.swap_test(states[i], states[j], shots)
                K[i, j] = fidelity
                K[j, i] = fidelity
        
        return K.tolist()
    
    def estimate_kernel(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        feature_map: Callable,
        shots: int = 1024
    ) -> float:
        """
        Estimate kernel using a feature map and SWAP test
        
        Args:
            x1: First feature vector
            x2: Second feature vector
            feature_map: Feature map function that takes data and returns a circuit
            shots: Number of shots
            
        Returns:
            float: Estimated kernel value
            
        Example:
            >>> def fm(x):
            ...     circ = QuantumCircuit(2)
            ...     circ.ry(0, x[0] * np.pi)
            ...     circ.ry(1, x[1] * np.pi)
            ...     return circ
            >>> kernel = estimator.estimate_kernel(x1, x2, fm)
        """
        # Build circuits for the data points
        circ1 = feature_map(x1)
        circ2 = feature_map(x2)
        
        # Get states
        state1 = circ1.run()
        state2 = circ2.run()
        
        # Use SWAP test
        fidelity = self.swap_test(state1, state2, shots)
        
        return float(fidelity)
    
    def __repr__(self) -> str:
        return f"QuantumKernelEstimator(n_qubits={self.n_qubits})"
    
    def __str__(self) -> str:
        return f"Quantum Kernel Estimator ({self.n_qubits} qubits)"


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'KernelResult',
    'QuantumKernel',
    'QuantumKernelEstimator',
]