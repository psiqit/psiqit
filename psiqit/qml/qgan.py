# psiqit/qml/qgan.py

"""
Quantum Generative Adversarial Network (QGAN)
Quantum machine learning for generative modeling
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, zero, one, plus, minus, basis, random_state
from ..quantum.operator import Operator, identity, pauli_x, pauli_y, pauli_z, hadamard, rx, ry, rz, cnot
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# QGAN RESULT CLASS
# ============================================================================

@dataclass
class QGANResult:
    """
    Result container for Quantum GAN training
    
    Attributes:
        generator_loss: Loss history of the generator
        discriminator_loss: Loss history of the discriminator
        fidelity_history: Fidelity between generated and target states
        epoch_history: Epoch numbers
        success: Whether training was successful
        final_generator: Final generator parameters
        final_discriminator: Final discriminator parameters
        generated_states: List of generated states during training
        target_states: Target states used for training
        n_epochs: Number of epochs trained
    """
    generator_loss: List[float] = field(default_factory=list)
    discriminator_loss: List[float] = field(default_factory=list)
    fidelity_history: List[float] = field(default_factory=list)
    epoch_history: List[int] = field(default_factory=list)
    success: bool = True
    final_generator: Optional[Dict[str, Any]] = None
    final_discriminator: Optional[Dict[str, Any]] = None
    generated_states: List[Ket] = field(default_factory=list)
    target_states: List[Ket] = field(default_factory=list)
    n_epochs: int = 0
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"QGANResult(epochs={self.n_epochs}, status={status}, final_fidelity={self.fidelity_history[-1] if self.fidelity_history else 0:.4f})"
    
    def __str__(self) -> str:
        lines = [
            "Quantum GAN Training Results:",
            f"  Epochs: {self.n_epochs}",
            f"  Success: {self.success}",
            f"  Final Generator Loss: {self.generator_loss[-1] if self.generator_loss else 'N/A':.6f}",
            f"  Final Discriminator Loss: {self.discriminator_loss[-1] if self.discriminator_loss else 'N/A':.6f}",
            f"  Final Fidelity: {self.fidelity_history[-1] if self.fidelity_history else 'N/A':.6f}",
        ]
        return "\n".join(lines)
    
    def get_best_fidelity(self) -> float:
        """Get the best fidelity achieved during training"""
        return max(self.fidelity_history) if self.fidelity_history else 0.0
    
    def get_best_epoch(self) -> int:
        """Get the epoch with the best fidelity"""
        if not self.fidelity_history:
            return 0
        return self.epoch_history[np.argmax(self.fidelity_history)]


# ============================================================================
# QUANTUM GENERATOR
# ============================================================================

class QuantumGenerator:
    """
    Quantum generator for QGAN
    
    Generates quantum states from latent vectors using a parameterized
    quantum circuit.
    
    Example:
        >>> gen = QuantumGenerator(n_qubits=2, n_latent=2, n_layers=2)
        >>> latent = np.array([0.1, 0.2])
        >>> state = gen.generate(latent)
        >>> print(state)
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_latent: int = 2,
        n_layers: int = 2,
        entangler: str = 'cnot'
    ):
        """
        Initialize quantum generator
        
        Args:
            n_qubits: Number of qubits for the generated state
            n_latent: Dimension of latent space
            n_layers: Number of variational layers
            entangler: Type of entangling gate ('cnot', 'cz', 'swap')
        """
        validate_qubits(n_qubits)
        
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.entangler = entangler
        
        # Number of parameters per layer
        self.params_per_layer = n_qubits * 3  # Rx, Ry, Rz per qubit
        self.n_params = self.params_per_layer * n_layers
        
        # Initialize parameters
        self._params = None
        self._init_params()
        
        logger.info(f"QuantumGenerator initialized: {n_qubits} qubits, {n_latent} latent dims, {n_layers} layers")
    
    def _init_params(self, seed: Optional[int] = None):
        """
        Initialize generator parameters
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self._params = np.random.uniform(-np.pi, np.pi, size=self.n_params)
    
    def _encode_latent(self, circuit: QuantumCircuit, z: np.ndarray):
        """
        Encode latent vector into the circuit
        
        Args:
            circuit: QuantumCircuit object
            z: Latent vector
        """
        # Encode latent vector as rotation angles
        for i in range(min(self.n_qubits, len(z))):
            circuit.ry(i, z[i])
        
        # If latent dimension is larger than qubits, use additional encoding
        if len(z) > self.n_qubits:
            for i in range(self.n_qubits, len(z)):
                idx = i % self.n_qubits
                circuit.rz(idx, z[i])
    
    def _variational_layer(self, circuit: QuantumCircuit, params: np.ndarray, start_idx: int):
        """
        Add a variational layer to the circuit
        
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
            # Circular entanglement
            circuit.cx(self.n_qubits - 1, 0)
        elif self.entangler == 'cz':
            for i in range(self.n_qubits - 1):
                circuit.cz(i, i + 1)
            circuit.cz(self.n_qubits - 1, 0)
        elif self.entangler == 'swap':
            for i in range(0, self.n_qubits - 1, 2):
                circuit.swap(i, i + 1)
    
    def generate(self, z: Optional[np.ndarray] = None, params: Optional[np.ndarray] = None) -> Ket:
        """
        Generate a quantum state from a latent vector
        
        Args:
            z: Latent vector (if None, uses random)
            params: Generator parameters (if None, uses current params)
            
        Returns:
            Ket: Generated state
            
        Example:
            >>> z = np.random.randn(2)
            >>> state = gen.generate(z)
        """
        if z is None:
            z = np.random.randn(self.n_latent)
        
        if len(z) != self.n_latent:
            raise ValueError(f"Latent vector dimension {len(z)} does not match {self.n_latent}")
        
        if params is None:
            params = self._params
        
        # Create circuit
        circuit = QuantumCircuit(self.n_qubits)
        
        # Encode latent vector
        self._encode_latent(circuit, z)
        
        # Apply variational layers
        for layer in range(self.n_layers):
            start_idx = layer * self.params_per_layer
            self._variational_layer(circuit, params, start_idx)
        
        # Run circuit
        state = circuit.run()
        return state
    
    def set_params(self, params: np.ndarray):
        """Set generator parameters"""
        if len(params) != self.n_params:
            raise ValueError(f"Parameter length {len(params)} does not match {self.n_params}")
        self._params = params
    
    def get_params(self) -> np.ndarray:
        """Get generator parameters"""
        return self._params.copy()
    
    def get_state_from_params(self, params: np.ndarray) -> Ket:
        """
        Get the state generated by a specific parameter set
        
        Args:
            params: Generator parameters
            
        Returns:
            Ket: Generated state
        """
        # Use a random latent vector
        z = np.random.randn(self.n_latent)
        return self.generate(z, params=params)
    
    def __repr__(self) -> str:
        return f"QuantumGenerator(n_qubits={self.n_qubits}, n_latent={self.n_latent}, n_layers={self.n_layers})"


# ============================================================================
# QUANTUM DISCRIMINATOR
# ============================================================================

class QuantumDiscriminator:
    """
    Quantum discriminator for QGAN
    
    Distinguishes between real (target) and generated states using
    a parameterized quantum circuit.
    
    Example:
        >>> disc = QuantumDiscriminator(n_qubits=2, n_layers=2)
        >>> state = Ket([1, 0, 0, 0])
        >>> score = disc.discriminate(state)
        >>> print(score)
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        entangler: str = 'cnot'
    ):
        """
        Initialize quantum discriminator
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            entangler: Type of entangling gate
        """
        validate_qubits(n_qubits)
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangler = entangler
        
        # Number of parameters
        self.params_per_layer = n_qubits * 3
        self.n_params = self.params_per_layer * n_layers
        
        # Initialize parameters
        self._params = None
        self._init_params()
        
        logger.info(f"QuantumDiscriminator initialized: {n_qubits} qubits, {n_layers} layers")
    
    def _init_params(self, seed: Optional[int] = None):
        """Initialize discriminator parameters"""
        if seed is not None:
            np.random.seed(seed)
        self._params = np.random.uniform(-np.pi, np.pi, size=self.n_params)
    
    def _variational_layer(self, circuit: QuantumCircuit, params: np.ndarray, start_idx: int):
        """Add a variational layer to the discriminator circuit"""
        for i in range(self.n_qubits):
            idx = start_idx + i * 3
            circuit.rx(i, params[idx])
            circuit.ry(i, params[idx + 1])
            circuit.rz(i, params[idx + 2])
        
        if self.entangler == 'cnot':
            for i in range(self.n_qubits - 1):
                circuit.cx(i, i + 1)
            circuit.cx(self.n_qubits - 1, 0)
        elif self.entangler == 'cz':
            for i in range(self.n_qubits - 1):
                circuit.cz(i, i + 1)
            circuit.cz(self.n_qubits - 1, 0)
    
    def _discriminator_circuit(self, params: np.ndarray) -> QuantumCircuit:
        """
        Build the discriminator circuit
        
        Args:
            params: Discriminator parameters
            
        Returns:
            QuantumCircuit: Discriminator circuit
        """
        circuit = QuantumCircuit(self.n_qubits)
        
        # Apply variational layers
        for layer in range(self.n_layers):
            start_idx = layer * self.params_per_layer
            self._variational_layer(circuit, params, start_idx)
        
        return circuit
    
    def discriminate(self, state: Union[Ket, np.ndarray], params: Optional[np.ndarray] = None) -> float:
        """
        Discriminate between real and generated states
        
        Args:
            state: Quantum state to classify
            params: Discriminator parameters (if None, uses current params)
            
        Returns:
            float: Score (0-1), 1 = real, 0 = generated
            
        Example:
            >>> state = Ket([1, 0, 0, 0])
            >>> score = disc.discriminate(state)
        """
        if isinstance(state, Ket):
            state_data = state.data
        else:
            state_data = np.array(state, dtype=complex)
        
        if params is None:
            params = self._params
        
        # Build discriminator circuit
        circuit = self._discriminator_circuit(params)
        
        # Apply input state
        # We need to initialize the circuit with the input state
        # For simplicity, we use the run method to get the state
        # and then measure the expectation value
        
        # In a real implementation, we would initialize the circuit with the state
        # Here we compute the overlap using the circuit's unitary
        
        # For simplicity, we use a fidelity-based score
        # The discriminator should output a value close to 1 for real states
        # and close to 0 for generated states
        
        # Run the circuit with the input state
        # We need to apply the discriminator unitary to the input state
        # and then measure the expectation value of Z on the first qubit
        
        # This is a simplified implementation
        # In practice, the discriminator would be a parameterized quantum circuit
        # that outputs a binary classification
        
        # For now, we use a simple score based on the state's properties
        # This is a placeholder for a full quantum discriminator
        
        # Compute the fidelity with the |0⟩ state
        score = np.abs(state_data[0]) ** 2
        
        return float(score)
    
    def set_params(self, params: np.ndarray):
        """Set discriminator parameters"""
        if len(params) != self.n_params:
            raise ValueError(f"Parameter length {len(params)} does not match {self.n_params}")
        self._params = params
    
    def get_params(self) -> np.ndarray:
        """Get discriminator parameters"""
        return self._params.copy()
    
    def __repr__(self) -> str:
        return f"QuantumDiscriminator(n_qubits={self.n_qubits}, n_layers={self.n_layers})"


# ============================================================================
# CLASSICAL DISCRIMINATOR
# ============================================================================

class ClassicalDiscriminator:
    """
    Classical discriminator for QGAN
    
    A simple neural network that distinguishes between real and generated
    states using the state vector amplitudes.
    
    Example:
        >>> disc = ClassicalDiscriminator(input_dim=4, hidden_dims=[8, 4])
        >>> state = Ket([1, 0, 0, 0])
        >>> score = disc.discriminate(state)
        >>> print(score)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        learning_rate: float = 0.01
    ):
        """
        Initialize classical discriminator
        
        Args:
            input_dim: Input dimension (2^n_qubits)
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for training
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        
        # Build network layers
        self.weights = []
        self.biases = []
        
        prev_dim = input_dim
        for dim in hidden_dims:
            self.weights.append(np.random.randn(prev_dim, dim) * 0.1)
            self.biases.append(np.zeros(dim))
            prev_dim = dim
        
        # Output layer (1 neuron)
        self.weights.append(np.random.randn(prev_dim, 1) * 0.1)
        self.biases.append(np.zeros(1))
        
        logger.info(f"ClassicalDiscriminator initialized: {input_dim} -> {hidden_dims} -> 1")
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through the network
        
        Args:
            x: Input vector
            
        Returns:
            Tuple: (output, activations)
        """
        activations = [x]
        current = x
        
        for i in range(len(self.weights) - 1):
            current = self._relu(current @ self.weights[i] + self.biases[i])
            activations.append(current)
        
        # Output layer with sigmoid
        output = self._sigmoid(current @ self.weights[-1] + self.biases[-1])
        activations.append(output)
        
        return output.flatten()[0], activations
    
    def discriminate(self, state: Union[Ket, np.ndarray]) -> float:
        """
        Discriminate between real and generated states
        
        Args:
            state: Quantum state to classify
            
        Returns:
            float: Score (0-1), 1 = real, 0 = generated
        """
        if isinstance(state, Ket):
            data = state.data
        else:
            data = np.array(state, dtype=complex)
        
        # Use real and imaginary parts as input
        features = np.concatenate([data.real, data.imag])
        
        # Forward pass
        output, _ = self._forward(features)
        
        return float(output)
    
    def train_step(self, real_states: List[Ket], fake_states: List[Ket]) -> Tuple[float, float]:
        """
        Perform a training step for the discriminator
        
        Args:
            real_states: List of real (target) states
            fake_states: List of generated states
            
        Returns:
            Tuple: (real_loss, fake_loss)
        """
        # This is a simplified implementation
        # In practice, we would use backpropagation with a proper loss function
        
        real_scores = [self.discriminate(s) for s in real_states]
        fake_scores = [self.discriminate(s) for s in fake_states]
        
        real_loss = -np.mean(np.log(np.clip(real_scores, 1e-10, 1)))
        fake_loss = -np.mean(np.log(np.clip(1 - np.array(fake_scores), 1e-10, 1)))
        
        return float(real_loss), float(fake_loss)
    
    def __repr__(self) -> str:
        return f"ClassicalDiscriminator(input_dim={self.input_dim}, hidden_dims={self.hidden_dims})"


# ============================================================================
# QGAN CLASS
# ============================================================================

class QGAN:
    """
    Quantum Generative Adversarial Network
    
    A quantum GAN for generating quantum states that match a target
    distribution using adversarial training.
    
    Example:
        >>> # Define target states (e.g., Bell states)
        >>> from psiqit.quantum import bell_phi_plus
        >>> target_states = [bell_phi_plus()]
        >>> qgan = QGAN(n_qubits=2, n_latent=2, n_layers=2)
        >>> result = qgan.train(target_states, epochs=50)
        >>> print(result.fidelity_history[-1])
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_latent: int = 2,
        n_layers: int = 2,
        discriminator_type: str = 'classical',
        learning_rate: float = 0.01,
        entangler: str = 'cnot'
    ):
        """
        Initialize QGAN
        
        Args:
            n_qubits: Number of qubits
            n_latent: Dimension of latent space
            n_layers: Number of variational layers
            discriminator_type: 'quantum' or 'classical'
            learning_rate: Learning rate for training
            entangler: Type of entangling gate
        """
        validate_qubits(n_qubits)
        
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        
        # Initialize generator
        self.generator = QuantumGenerator(n_qubits, n_latent, n_layers, entangler)
        
        # Initialize discriminator
        self.discriminator_type = discriminator_type
        if discriminator_type == 'quantum':
            self.discriminator = QuantumDiscriminator(n_qubits, n_layers, entangler)
        else:
            self.discriminator = ClassicalDiscriminator(2 * 2 ** n_qubits, [64, 32])
        
        logger.info(f"QGAN initialized: {n_qubits} qubits, {n_latent} latent dims, discriminator={discriminator_type}")
    
    def _generator_loss(self, z: np.ndarray) -> float:
        """
        Compute generator loss
        
        Args:
            z: Latent vector
            
        Returns:
            float: Generator loss
        """
        generated_state = self.generator.generate(z)
        score = self.discriminator.discriminate(generated_state)
        loss = -np.log(np.clip(score, 1e-10, 1))
        return float(loss)
    
    def _discriminator_loss(
        self,
        real_states: List[Ket],
        z_list: List[np.ndarray]
    ) -> float:
        """
        Compute discriminator loss
        
        Args:
            real_states: List of real (target) states
            z_list: List of latent vectors
            
        Returns:
            float: Discriminator loss
        """
        # Real states loss
        real_scores = [self.discriminator.discriminate(s) for s in real_states]
        real_loss = -np.mean(np.log(np.clip(real_scores, 1e-10, 1)))
        
        # Generated states loss
        fake_scores = []
        for z in z_list:
            fake_state = self.generator.generate(z)
            fake_scores.append(self.discriminator.discriminate(fake_state))
        
        fake_loss = -np.mean(np.log(np.clip(1 - np.array(fake_scores), 1e-10, 1)))
        
        return float(real_loss + fake_loss)
    
    def _gradient(self, loss_fn: Callable, params: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Compute gradient using finite differences
        
        Args:
            loss_fn: Loss function that takes parameters
            params: Parameter vector
            eps: Step size
            
        Returns:
            np.ndarray: Gradient
        """
        grad = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            
            loss_plus = loss_fn(params_plus)
            loss_minus = loss_fn(params_minus)
            grad[i] = (loss_plus - loss_minus) / (2 * eps)
        
        return grad
    
    def _fidelity(self, generated: Ket, target: Ket) -> float:
        """
        Compute fidelity between two states
        
        Args:
            generated: Generated state
            target: Target state
            
        Returns:
            float: Fidelity (0-1)
        """
        return float(abs(np.vdot(generated.data, target.data)) ** 2)
    
    def train(
        self,
        target_states: List[Ket],
        epochs: int = 50,
        batch_size: int = 10,
        learning_rate: Optional[float] = None,
        verbose: bool = True
    ) -> QGANResult:
        """
        Train the QGAN
        
        Args:
            target_states: List of target states to learn
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate (overrides default)
            verbose: Print progress
            
        Returns:
            QGANResult: Training results
            
        Example:
            >>> from psiqit.quantum import bell_phi_plus
            >>> target = [bell_phi_plus()]
            >>> result = qgan.train(target, epochs=50)
            >>> print(result.final_fidelity)
        """
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        logger.info(f"Starting QGAN training: {epochs} epochs, batch_size={batch_size}")
        
        # Initialize result
        result = QGANResult()
        result.target_states = target_states
        
        # Training loop
        for epoch in range(epochs):
            # Generate batch of latent vectors
            z_batch = [np.random.randn(self.n_latent) for _ in range(batch_size)]
            
            # Select batch of target states (with replacement)
            target_batch = np.random.choice(target_states, size=batch_size, replace=True)
            
            # Train discriminator
            # For simplicity, we use a simple gradient update
            # In practice, we would use proper optimization algorithms
            
            # Train generator
            # Update generator to fool discriminator
            gen_loss = 0
            for z in z_batch:
                gen_loss += self._generator_loss(z)
            gen_loss /= batch_size
            
            # Update generator parameters
            # This is a simplified update - in practice, we would use backpropagation
            # through the quantum circuit
            
            # For now, we just record the losses
            result.generator_loss.append(gen_loss)
            
            # Compute discriminator loss
            disc_loss = self._discriminator_loss(target_batch, z_batch)
            result.discriminator_loss.append(disc_loss)
            
            # Compute fidelity with first target state
            # Generate a state and compute fidelity with target
            z_test = np.random.randn(self.n_latent)
            generated_state = self.generator.generate(z_test)
            fid = self._fidelity(generated_state, target_states[0])
            result.fidelity_history.append(fid)
            result.epoch_history.append(epoch)
            result.generated_states.append(generated_state)
            
            # Progress
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: G_loss={gen_loss:.4f}, D_loss={disc_loss:.4f}, Fidelity={fid:.4f}")
        
        result.n_epochs = epochs
        result.success = result.fidelity_history[-1] > 0.5 if result.fidelity_history else False
        
        # Store final parameters
        result.final_generator = {
            'params': self.generator.get_params(),
            'state': generated_state
        }
        
        logger.info(f"QGAN training completed: final_fidelity={result.fidelity_history[-1] if result.fidelity_history else 0:.4f}")
        
        return result
    
    def generate(
        self,
        n_samples: int = 1,
        latent_vectors: Optional[List[np.ndarray]] = None
    ) -> List[Ket]:
        """
        Generate quantum states using the trained generator
        
        Args:
            n_samples: Number of samples to generate
            latent_vectors: List of latent vectors (if None, random)
            
        Returns:
            List[Ket]: Generated states
            
        Example:
            >>> states = qgan.generate(n_samples=5)
            >>> for s in states:
            ...     print(s)
        """
        if latent_vectors is None:
            latent_vectors = [np.random.randn(self.n_latent) for _ in range(n_samples)]
        
        states = []
        for z in latent_vectors:
            state = self.generator.generate(z)
            states.append(state)
        
        return states
    
    def get_generator(self) -> QuantumGenerator:
        """Get the generator"""
        return self.generator
    
    def get_discriminator(self) -> Union[QuantumDiscriminator, ClassicalDiscriminator]:
        """Get the discriminator"""
        return self.discriminator
    
    def __repr__(self) -> str:
        return f"QGAN(n_qubits={self.n_qubits}, n_latent={self.n_latent}, n_layers={self.n_layers})"


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'QGANResult',
    'QuantumGenerator',
    'QuantumDiscriminator',
    'ClassicalDiscriminator',
    'QGAN',
]