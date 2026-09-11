# psiqit/qml/qnn.py

"""
Quantum Neural Network (QNN)
Variational quantum circuits for machine learning
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
# QNN RESULT CLASS
# ============================================================================

@dataclass
class QNNResult:
    """
    Result container for Quantum Neural Network training
    
    Attributes:
        final_loss: Final loss value
        accuracy: Final accuracy (for classification tasks)
        epochs: Number of epochs trained
        params: Final parameters
        predictions: Final predictions
        loss_history: Loss history during training
        accuracy_history: Accuracy history during training
        success: Whether training was successful
    """
    final_loss: float = 0.0
    accuracy: float = 0.0
    epochs: int = 0
    params: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    loss_history: List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)
    success: bool = True
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"QNNResult(epochs={self.epochs}, status={status}, loss={self.final_loss:.6f}, accuracy={self.accuracy:.4f})"
    
    def __str__(self) -> str:
        lines = [
            "Quantum Neural Network Results:",
            f"  Epochs: {self.epochs}",
            f"  Success: {self.success}",
            f"  Final Loss: {self.final_loss:.6f}",
            f"  Final Accuracy: {self.accuracy:.4f}",
        ]
        return "\n".join(lines)


# ============================================================================
# QNN LAYER CLASS
# ============================================================================

class QNNLayer:
    """
    A layer in a Quantum Neural Network
    
    Consists of single-qubit rotations followed by entanglement.
    
    Example:
        >>> layer = QNNLayer(n_qubits=2, entangler='cnot')
        >>> circuit = QuantumCircuit(2)
        >>> params = np.random.randn(6)
        >>> layer.apply(circuit, params)
    """
    
    def __init__(self, n_qubits: int, entangler: str = 'cnot'):
        """
        Initialize a QNN layer
        
        Args:
            n_qubits: Number of qubits
            entangler: Type of entangling gate ('cnot', 'cz', 'swap')
        """
        self.n_qubits = n_qubits
        self.entangler = entangler
        
        # Parameters per layer: Rx, Ry, Rz for each qubit
        self.n_params = n_qubits * 3
    
    def apply(self, circuit: QuantumCircuit, params: np.ndarray, start_idx: int = 0):
        """
        Apply the layer to a circuit
        
        Args:
            circuit: QuantumCircuit object
            params: Parameters for the layer
            start_idx: Starting index for parameters
        """
        # Single-qubit rotations
        for i in range(self.n_qubits):
            idx = start_idx + i * 3
            circuit.rx(i, params[idx])
            circuit.ry(i, params[idx + 1])
            circuit.rz(i, params[idx + 2])
        
        # Entangling gates
        if self.entangler == 'cnot':
            for i in range(self.n_qubits - 1):
                circuit.cx(i, i + 1)
            circuit.cx(self.n_qubits - 1, 0)
        elif self.entangler == 'cz':
            for i in range(self.n_qubits - 1):
                circuit.cz(i, i + 1)
            circuit.cz(self.n_qubits - 1, 0)
        elif self.entangler == 'swap':
            for i in range(0, self.n_qubits - 1, 2):
                circuit.swap(i, i + 1)
    
    def __repr__(self) -> str:
        return f"QNNLayer(n_qubits={self.n_qubits}, entangler={self.entangler})"


# ============================================================================
# QUANTUM NEURAL NETWORK
# ============================================================================

class QNN:
    """
    Quantum Neural Network
    
    A variational quantum circuit that can be trained for classification
    and regression tasks.
    
    Example:
        >>> from psiqit.qml import QNN
        >>> # Create a QNN for binary classification
        >>> qnn = QNN(n_qubits=2, n_layers=3)
        >>> # Train on synthetic data
        >>> X = np.random.randn(100, 2)  # Features
        >>> y = np.random.randint(0, 2, 100)  # Labels
        >>> result = qnn.train(X, y, epochs=50)
        >>> print(result.accuracy)
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 3,
        entangler: str = 'cnot',
        measurement: str = 'z'
    ):
        """
        Initialize Quantum Neural Network
        
        Args:
            n_qubits: Number of qubits (features will be encoded here)
            n_layers: Number of variational layers
            entangler: Type of entangling gate ('cnot', 'cz', 'swap')
            measurement: Measurement basis ('z', 'x', 'y')
        """
        validate_qubits(n_qubits)
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangler = entangler
        self.measurement = measurement
        
        # Parameters per layer: Rx, Ry, Rz per qubit
        self.params_per_layer = n_qubits * 3
        self.n_params = self.params_per_layer * n_layers
        
        # Initialize parameters
        self._params = None
        self._init_params_random()
        
        # Create layers
        self.layers = [QNNLayer(n_qubits, entangler) for _ in range(n_layers)]
        
        logger.info(f"QNN initialized: {n_qubits} qubits, {n_layers} layers, entangler={entangler}")
    
    def _init_params_random(self, seed: Optional[int] = None):
        """
        Initialize parameters randomly
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self._params = np.random.uniform(-np.pi, np.pi, size=self.n_params)
    
    def set_params(self, params: np.ndarray):
        """
        Set the parameters of the QNN
        
        Args:
            params: Parameter vector
        """
        if len(params) != self.n_params:
            raise ValueError(f"Parameter length {len(params)} does not match {self.n_params}")
        self._params = params
    
    def build_circuit(self, input_state: Optional[Union[Ket, np.ndarray]] = None) -> QuantumCircuit:
        """
        Build the QNN circuit
        
        Args:
            input_state: Input state (if None, uses |0...0⟩)
            
        Returns:
            QuantumCircuit: The QNN circuit
            
        Example:
            >>> circ = qnn.build_circuit()
            >>> print(circ.draw())
        """
        circuit = QuantumCircuit(self.n_qubits)
        
        # If input state is provided, initialize the circuit
        if input_state is not None:
            if isinstance(input_state, Ket):
                state_data = input_state.data
            else:
                state_data = np.array(input_state, dtype=complex)
            
            # Apply initial state preparation
            # For simplicity, we encode the state using rotations
            # This is a simplified version - in practice, state preparation
            # would require more sophisticated methods
            for i in range(min(self.n_qubits, len(state_data))):
                # Use the amplitude to set initial rotation
                angle = 2 * np.arccos(np.abs(state_data[i]))
                circuit.ry(i, angle)
        
        # Apply variational layers
        for layer_idx, layer in enumerate(self.layers):
            start_idx = layer_idx * self.params_per_layer
            layer.apply(circuit, self._params, start_idx)
        
        return circuit
    
    def forward(self, input_state: Optional[Union[Ket, np.ndarray]] = None) -> Ket:
        """
        Forward pass through the QNN
        
        Args:
            input_state: Input state (if None, uses |0...0⟩)
            
        Returns:
            Ket: Output state
            
        Example:
            >>> state = qnn.forward()
            >>> print(state)
        """
        circuit = self.build_circuit(input_state)
        return circuit.run()
    
    def measure_expectation(self, input_state: Optional[Union[Ket, np.ndarray]] = None) -> float:
        """
        Measure the expectation value of the QNN output
        
        Args:
            input_state: Input state
            
        Returns:
            float: Expectation value (-1 to 1)
            
        Example:
            >>> expectation = qnn.measure_expectation()
            >>> print(expectation)
        """
        # Get the output state
        state = self.forward(input_state)
        
        # Measure in the specified basis
        if self.measurement == 'z':
            # Measure Z expectation on the first qubit
            # For a single qubit state, this is |a|² - |b|²
            if state.dim == 2:
                return float(np.abs(state.data[0])**2 - np.abs(state.data[1])**2)
            else:
                # For multi-qubit states, use the reduced density matrix
                # Simplified: use the first qubit
                prob_0 = 0.0
                prob_1 = 0.0
                for i, amp in enumerate(state.data):
                    if (i >> 0) & 1:
                        prob_1 += np.abs(amp)**2
                    else:
                        prob_0 += np.abs(amp)**2
                return float(prob_0 - prob_1)
        
        elif self.measurement == 'x':
            # Measure X expectation (simplified)
            # This would require additional Hadamard gates
            # For simplicity, use a different approach
            H = hadamard().data
            state_x = Ket(H @ state.data)
            return float(np.abs(state_x.data[0])**2 - np.abs(state_x.data[1])**2)
        
        elif self.measurement == 'y':
            # Measure Y expectation (simplified)
            # This would require additional phase and Hadamard gates
            # For simplicity, use a different approach
            from ..quantum.operator import pauli_y
            Y = pauli_y().data
            expectation = np.vdot(state.data, Y @ state.data).real
            return float(expectation)
        
        else:
            raise ValueError(f"Unknown measurement basis: {self.measurement}")
    
    def _encode_data(self, x: np.ndarray) -> Ket:
        """
        Encode classical data into a quantum state
        
        Args:
            x: Feature vector
            
        Returns:
            Ket: Encoded quantum state
            
        Example:
            >>> state = qnn._encode_data(np.array([0.5, 0.3]))
        """
        # This is a simple angle encoding
        # In practice, more sophisticated encoding methods can be used
        n_features = min(len(x), self.n_qubits)
        
        # Create the state vector
        state_data = np.zeros(2 ** self.n_qubits, dtype=complex)
        state_data[0] = 1.0  # Start from |0...0⟩
        
        # Apply angle encoding
        # For simplicity, we create a product state
        # with each qubit rotated by the feature value
        from ..quantum.operator import ry
        
        # Build the state
        current_state = np.array([1.0, 0.0], dtype=complex)
        for i in range(n_features):
            # Encode feature as rotation angle
            angle = x[i] * np.pi  # Map to [0, π]
            R = ry(angle).data
            qubit_state = R @ np.array([1.0, 0.0], dtype=complex)
            current_state = np.kron(current_state, qubit_state)
        
        # Pad with zeros if needed
        if self.n_qubits > n_features:
            for _ in range(n_features, self.n_qubits):
                qubit_state = np.array([1.0, 0.0], dtype=complex)
                current_state = np.kron(current_state, qubit_state)
        
        return Ket(current_state)
    
    def predict(self, X: np.ndarray, as_probability: bool = True) -> np.ndarray:
        """
        Make predictions for a set of inputs
        
        Args:
            X: Input data (n_samples x n_features)
            as_probability: Return probabilities (True) or binary predictions (False)
            
        Returns:
            np.ndarray: Predictions
            
        Example:
            >>> X_test = np.random.randn(10, 2)
            >>> predictions = qnn.predict(X_test)
        """
        predictions = []
        
        for x in X:
            # Encode data
            state = self._encode_data(x)
            
            # Forward pass
            output_state = self.forward(state)
            
            # Measure expectation
            expectation = self.measure_expectation(output_state)
            
            # Convert to probability
            prob = (expectation + 1) / 2  # Map [-1, 1] to [0, 1]
            
            if as_probability:
                predictions.append(prob)
            else:
                predictions.append(1 if prob > 0.5 else 0)
        
        return np.array(predictions)
    
    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the loss (binary cross-entropy)
        
        Args:
            X: Input data
            y: Labels (0 or 1)
            
        Returns:
            float: Loss value
            
        Example:
            >>> loss = qnn.loss(X_train, y_train)
        """
        predictions = self.predict(X, as_probability=True)
        
        # Binary cross-entropy loss
        y = np.array(y)
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        
        return float(loss)
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the accuracy
        
        Args:
            X: Input data
            y: Labels (0 or 1)
            
        Returns:
            float: Accuracy (0-1)
            
        Example:
            >>> acc = qnn.accuracy(X_test, y_test)
        """
        predictions = self.predict(X, as_probability=False)
        y = np.array(y)
        accuracy = np.mean(predictions == y)
        return float(accuracy)
    
    def _parameter_shift_gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_idx: int,
        epsilon: float = 1e-5
    ) -> float:
        """
        Compute the gradient using parameter-shift rule
        
        Args:
            X: Input data
            y: Labels
            param_idx: Parameter index
            epsilon: Small shift for finite difference
            
        Returns:
            float: Gradient value
            
        Example:
            >>> grad = qnn._parameter_shift_gradient(X_train, y_train, 0)
        """
        # Finite difference approximation
        params_plus = self._params.copy()
        params_minus = self._params.copy()
        params_plus[param_idx] += epsilon
        params_minus[param_idx] -= epsilon
        
        # Save current params
        current_params = self._params.copy()
        
        # Compute loss with shifted params
        self.set_params(params_plus)
        loss_plus = self.loss(X, y)
        
        self.set_params(params_minus)
        loss_minus = self.loss(X, y)
        
        # Restore params
        self.set_params(current_params)
        
        return (loss_plus - loss_minus) / (2 * epsilon)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.1,
        batch_size: Optional[int] = None,
        verbose: bool = True
    ) -> QNNResult:
        """
        Train the QNN
        
        Args:
            X: Input data (n_samples x n_features)
            y: Labels (0 or 1)
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size (if None, uses full batch)
            verbose: Print progress
            
        Returns:
            QNNResult: Training results
            
        Example:
            >>> result = qnn.train(X_train, y_train, epochs=50)
            >>> print(f"Final accuracy: {result.accuracy}")
        """
        n_samples = len(X)
        if batch_size is None:
            batch_size = n_samples
        
        logger.info(f"Starting QNN training: {epochs} epochs, batch_size={batch_size}")
        
        # Initialize result
        result = QNNResult()
        result.params = self._params.copy()
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            epoch_loss = 0.0
            n_batches = 0
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # Compute gradients for all parameters
                grad = np.zeros(self.n_params)
                for param_idx in range(self.n_params):
                    grad[param_idx] = self._parameter_shift_gradient(
                        X_batch, y_batch, param_idx
                    )
                
                # Update parameters
                self._params -= learning_rate * grad
                
                # Compute batch loss
                batch_loss = self.loss(X_batch, y_batch)
                epoch_loss += batch_loss
                n_batches += 1
            
            # Average loss for the epoch
            epoch_loss /= n_batches
            result.loss_history.append(epoch_loss)
            
            # Compute accuracy
            epoch_accuracy = self.accuracy(X, y)
            result.accuracy_history.append(epoch_accuracy)
            
            # Progress
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: loss={epoch_loss:.6f}, accuracy={epoch_accuracy:.4f}")
        
        # Final results
        result.final_loss = self.loss(X, y)
        result.accuracy = self.accuracy(X, y)
        result.epochs = epochs
        result.params = self._params.copy()
        result.predictions = self.predict(X, as_probability=False)
        result.success = result.accuracy > 0.5
        
        logger.info(f"QNN training completed: final_loss={result.final_loss:.6f}, accuracy={result.accuracy:.4f}")
        
        return result
    
    def __repr__(self) -> str:
        return f"QNN(n_qubits={self.n_qubits}, n_layers={self.n_layers}, entangler={self.entangler})"
    
    def __str__(self) -> str:
        lines = [
            "Quantum Neural Network:",
            f"  Qubits: {self.n_qubits}",
            f"  Layers: {self.n_layers}",
            f"  Entangler: {self.entangler}",
            f"  Measurement: {self.measurement}",
            f"  Parameters: {self.n_params}",
        ]
        return "\n".join(lines)


# ============================================================================
# ALIASES
# ============================================================================

class QuantumClassifier(QNN):
    """
    Quantum Classifier - Alias for QNN
    
    A quantum neural network for classification tasks.
    """
    pass


class VariationalQuantumCircuit(QNN):
    """
    Variational Quantum Circuit - Alias for QNN
    
    A parameterized quantum circuit for variational algorithms.
    """
    pass


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'QNNResult',
    'QNNLayer',
    'QNN',
    'QuantumClassifier',
    'VariationalQuantumCircuit',
]
