# psiqit/qml/qsvm.py

"""
Quantum Support Vector Machine (QSVM)
Quantum kernel methods for classification
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, zero, one, plus, minus, basis
from ..quantum.operator import Operator, identity, pauli_x, pauli_y, pauli_z, hadamard, rx, ry, rz, cnot, cz
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# SVM RESULT CLASS
# ============================================================================

@dataclass
class SVMResult:
    """
    Result container for SVM training
    
    Attributes:
        support_vectors: Support vectors
        alphas: Lagrange multipliers
        bias: Bias term
        n_support: Number of support vectors
        accuracy: Training accuracy
        classes: Unique classes
        n_iterations: Number of iterations
        success: Whether training was successful
    """
    support_vectors: Optional[np.ndarray] = None
    alphas: Optional[np.ndarray] = None
    bias: float = 0.0
    n_support: int = 0
    accuracy: float = 0.0
    classes: Optional[np.ndarray] = None
    n_iterations: int = 0
    success: bool = True
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"SVMResult(n_support={self.n_support}, status={status}, accuracy={self.accuracy:.4f})"
    
    def __str__(self) -> str:
        lines = [
            "SVM Training Results:",
            f"  Success: {self.success}",
            f"  Support Vectors: {self.n_support}",
            f"  Bias: {self.bias:.6f}",
            f"  Accuracy: {self.accuracy:.4f}",
            f"  Iterations: {self.n_iterations}",
        ]
        if self.classes is not None:
            lines.append(f"  Classes: {self.classes}")
        return "\n".join(lines)


# ============================================================================
# QUANTUM SUPPORT VECTOR MACHINE
# ============================================================================

class QSVM:
    """
    Quantum Support Vector Machine
    
    A support vector machine that uses quantum kernels for classification.
    Supports both classical kernels (linear, RBF) and quantum kernels.
    
    Example:
        >>> from psiqit.qml import QSVM
        >>> # Create a QSVM with quantum kernel
        >>> qsvm = QSVM(n_qubits=2, kernel_type='quantum', C=1.0)
        >>> # Train on synthetic data
        >>> X = np.random.randn(100, 2)
        >>> y = np.random.randint(0, 2, 100)
        >>> result = qsvm.fit(X, y)
        >>> print(result.accuracy)
    """
    
    def __init__(
        self,
        n_qubits: int = 2,
        kernel_type: str = 'quantum',
        C: float = 1.0,
        feature_map: str = 'zz',
        gamma: float = 1.0
    ):
        """
        Initialize Quantum SVM
        
        Args:
            n_qubits: Number of qubits for quantum kernel
            kernel_type: 'linear', 'rbf', or 'quantum'
            C: Regularization parameter
            feature_map: Type of feature map for quantum kernel ('zz', 'xy', 'custom')
            gamma: Gamma parameter for RBF kernel
        """
        validate_qubits(n_qubits)
        
        self.n_qubits = n_qubits
        self.kernel_type = kernel_type
        self.C = C
        self.feature_map = feature_map
        self.gamma = gamma
        
        # Model parameters
        self._support_vectors = None
        self._alphas = None
        self._bias = 0.0
        self._classes = None
        
        logger.info(f"QSVM initialized: n_qubits={n_qubits}, kernel_type={kernel_type}, C={C}")
    
    def _linear_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Linear kernel: K(x1, x2) = x1 · x2
        
        Args:
            x1: First feature vector
            x2: Second feature vector
            
        Returns:
            float: Kernel value
        """
        return float(np.dot(x1, x2))
    
    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        RBF (Gaussian) kernel: K(x1, x2) = exp(-γ||x1 - x2||²)
        
        Args:
            x1: First feature vector
            x2: Second feature vector
            
        Returns:
            float: Kernel value
        """
        diff = x1 - x2
        return float(np.exp(-self.gamma * np.dot(diff, diff)))
    
    def _quantum_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Quantum kernel: K(x1, x2) = |⟨ψ(x1)|ψ(x2)⟩|²
        
        Args:
            x1: First feature vector
            x2: Second feature vector
            
        Returns:
            float: Quantum kernel value
        """
        # Encode features into quantum states
        psi1 = self._encode_features(x1)
        psi2 = self._encode_features(x2)
        
        # Compute fidelity
        fidelity = abs(np.vdot(psi1.data, psi2.data)) ** 2
        return float(fidelity)
    
    def _encode_features(self, x: np.ndarray) -> Ket:
        """
        Encode classical features into a quantum state
        
        Args:
            x: Feature vector
            
        Returns:
            Ket: Encoded quantum state
            
        Example:
            >>> state = qsvm._encode_features(np.array([0.5, 0.3]))
        """
        # Reshape if needed
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        n_features = min(x.shape[-1], self.n_qubits)
        
        # Create circuit
        circuit = QuantumCircuit(self.n_qubits)
        
        # Apply feature map
        if self.feature_map == 'zz':
            # ZZ Feature Map
            for i in range(n_features):
                # Encode feature as rotation
                circuit.ry(i, x[0, i] * np.pi)
            
            # Entangling gates
            for i in range(n_features - 1):
                angle = x[0, i] * x[0, i + 1]
                circuit.cz(i, i + 1)
                circuit.rz(i + 1, angle * np.pi)
                circuit.cz(i, i + 1)
        
        elif self.feature_map == 'xy':
            # XY Feature Map
            for i in range(n_features):
                circuit.rx(i, x[0, i] * np.pi)
                circuit.ry(i, x[0, i] * np.pi)
            
            for i in range(n_features - 1):
                circuit.cx(i, i + 1)
                circuit.ry(i + 1, x[0, i] * x[0, i + 1] * np.pi)
                circuit.cx(i, i + 1)
        
        else:
            # Default: simple angle encoding
            for i in range(n_features):
                circuit.ry(i, x[0, i] * np.pi)
        
        # Run circuit
        state = circuit.run()
        return state
    
    def _compute_kernel(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the kernel matrix for all pairs of data points
        
        Args:
            X: Data matrix (n_samples x n_features)
            
        Returns:
            np.ndarray: Kernel matrix (n_samples x n_samples)
        """
        n_samples = len(X)
        K = np.zeros((n_samples, n_samples))
        
        # Select kernel function
        if self.kernel_type == 'linear':
            kernel_fn = self._linear_kernel
        elif self.kernel_type == 'rbf':
            kernel_fn = self._rbf_kernel
        elif self.kernel_type == 'quantum':
            kernel_fn = self._quantum_kernel
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")
        
        # Compute kernel matrix
        for i in range(n_samples):
            for j in range(i, n_samples):
                k_val = kernel_fn(X[i], X[j])
                K[i, j] = k_val
                K[j, i] = k_val
        
        return K
    
    def _solve_svm(self, K: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Solve the SVM dual problem
        
        maximize: Σ α_i - 0.5 Σ α_i α_j y_i y_j K(x_i, x_j)
        subject to: 0 ≤ α_i ≤ C, Σ α_i y_i = 0
        
        Args:
            K: Kernel matrix
            y: Labels (±1)
            
        Returns:
            Tuple: (alphas, bias)
        """
        n_samples = len(y)
        
        # Use a simplified SMO-like algorithm
        # This is a placeholder - in practice, use a proper QP solver
        
        # For simplicity, we use a small iterative method
        # Initialize alphas
        alphas = np.zeros(n_samples)
        bias = 0.0
        
        # Simple iterative update
        max_iter = 100
        for iteration in range(max_iter):
            alphas_old = alphas.copy()
            
            for i in range(n_samples):
                # Compute error
                pred = np.sum(alphas * y * K[i, :]) + bias
                error = pred - y[i]
                
                # Update alpha
                if y[i] * error < 0:
                    alphas[i] = min(alphas[i] + 0.01, self.C)
                elif y[i] * error > 0:
                    alphas[i] = max(alphas[i] - 0.01, 0)
            
            # Update bias
            support_indices = np.where(alphas > 1e-6)[0]
            if len(support_indices) > 0:
                bias = np.mean([
                    y[i] - np.sum(alphas * y * K[i, :])
                    for i in support_indices
                ])
            
            # Check convergence
            if np.max(np.abs(alphas - alphas_old)) < 1e-6:
                break
        
        return alphas, bias
    
    def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int = 100) -> SVMResult:
        """
        Train the QSVM
        
        Args:
            X: Training data (n_samples x n_features)
            y: Labels (0 or 1)
            max_iter: Maximum iterations for solver
            
        Returns:
            SVMResult: Training results
            
        Example:
            >>> result = qsvm.fit(X_train, y_train)
            >>> print(result.accuracy)
        """
        logger.info(f"Training QSVM: {len(X)} samples, {X.shape[1]} features")
        
        # Convert labels to ±1
        self._classes = np.unique(y)
        if len(self._classes) != 2:
            raise ValueError("QSVM only supports binary classification")
        
        y_svm = np.where(y == self._classes[0], -1, 1)
        
        # Compute kernel matrix
        K = self._compute_kernel(X)
        
        # Solve SVM
        alphas, bias = self._solve_svm(K, y_svm)
        
        # Find support vectors
        support_mask = alphas > 1e-6
        support_vectors = X[support_mask]
        support_alphas = alphas[support_mask]
        support_labels = y_svm[support_mask]
        
        # Store model parameters
        self._support_vectors = support_vectors
        self._alphas = support_alphas
        self._bias = bias
        self._support_labels = support_labels
        self._X = X
        self._y = y_svm
        self._K = K
        
        # Compute training accuracy
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        
        logger.info(f"QSVM training completed: {len(support_vectors)} support vectors, accuracy={accuracy:.4f}")
        
        return SVMResult(
            support_vectors=support_vectors,
            alphas=support_alphas,
            bias=bias,
            n_support=len(support_vectors),
            accuracy=accuracy,
            classes=self._classes,
            n_iterations=max_iter,
            success=True
        )
    
    def _predict_single(self, x: np.ndarray) -> int:
        """
        Predict label for a single sample
        
        Args:
            x: Feature vector
            
        Returns:
            int: Predicted label (0 or 1)
        """
        if self._support_vectors is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Select kernel function
        if self.kernel_type == 'linear':
            kernel_fn = self._linear_kernel
        elif self.kernel_type == 'rbf':
            kernel_fn = self._rbf_kernel
        elif self.kernel_type == 'quantum':
            kernel_fn = self._quantum_kernel
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")
        
        # Compute decision function
        decision = self._bias
        for alpha, y, sv in zip(self._alphas, self._support_labels, self._support_vectors):
            decision += alpha * y * kernel_fn(x, sv)
        
        # Convert to class label
        pred_label = self._classes[0] if decision < 0 else self._classes[1]
        return int(pred_label)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for multiple samples
        
        Args:
            X: Test data (n_samples x n_features)
            
        Returns:
            np.ndarray: Predicted labels
            
        Example:
            >>> predictions = qsvm.predict(X_test)
        """
        predictions = np.array([self._predict_single(x) for x in X])
        return predictions
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values for multiple samples
        
        Args:
            X: Test data (n_samples x n_features)
            
        Returns:
            np.ndarray: Decision values
            
        Example:
            >>> decisions = qsvm.decision_function(X_test)
        """
        if self._support_vectors is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Select kernel function
        if self.kernel_type == 'linear':
            kernel_fn = self._linear_kernel
        elif self.kernel_type == 'rbf':
            kernel_fn = self._rbf_kernel
        elif self.kernel_type == 'quantum':
            kernel_fn = self._quantum_kernel
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")
        
        decisions = []
        for x in X:
            decision = self._bias
            for alpha, y, sv in zip(self._alphas, self._support_labels, self._support_vectors):
                decision += alpha * y * kernel_fn(x, sv)
            decisions.append(decision)
        
        return np.array(decisions)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy on test data
        
        Args:
            X: Test data
            y: True labels
            
        Returns:
            float: Accuracy (0-1)
            
        Example:
            >>> accuracy = qsvm.score(X_test, y_test)
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return float(accuracy)
    
    def get_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the kernel matrix for the given data
        
        Args:
            X: Data matrix
            
        Returns:
            np.ndarray: Kernel matrix
        """
        return self._compute_kernel(X)
    
    def get_support_vectors(self) -> np.ndarray:
        """Get the support vectors"""
        return self._support_vectors
    
    def get_alphas(self) -> np.ndarray:
        """Get the Lagrange multipliers"""
        return self._alphas
    
    def get_bias(self) -> float:
        """Get the bias term"""
        return self._bias
    
    def __repr__(self) -> str:
        trained = "trained" if self._support_vectors is not None else "untrained"
        return f"QSVM(n_qubits={self.n_qubits}, kernel={self.kernel_type}, C={self.C}, {trained})"
    
    def __str__(self) -> str:
        lines = [
            "Quantum Support Vector Machine:",
            f"  Qubits: {self.n_qubits}",
            f"  Kernel: {self.kernel_type}",
            f"  C: {self.C}",
            f"  Feature Map: {self.feature_map}",
        ]
        if self._support_vectors is not None:
            lines.append(f"  Support Vectors: {len(self._support_vectors)}")
            lines.append(f"  Bias: {self._bias:.6f}")
        else:
            lines.append("  Status: Not trained")
        return "\n".join(lines)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'SVMResult',
    'QSVM',
]
