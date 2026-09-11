# psiqit/qml/vqc.py

"""
Variational Quantum Circuit (VQC)
Variational quantum algorithms and classifiers
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
# VQC RESULT CLASS
# ============================================================================

@dataclass
class VQCResult:
    """
    Result container for Variational Quantum Circuit optimization
    
    Attributes:
        optimal_params: Optimal parameters found
        optimal_cost: Optimal cost value
        n_iterations: Number of iterations
        history: History of cost values
        success: Whether optimization was successful
        message: Additional message
    """
    optimal_params: Optional[np.ndarray] = None
    optimal_cost: float = 0.0
    n_iterations: int = 0
    history: List[float] = field(default_factory=list)
    success: bool = True
    message: str = ""
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"VQCResult(iterations={self.n_iterations}, status={status}, cost={self.optimal_cost:.6f})"
    
    def __str__(self) -> str:
        lines = [
            "Variational Quantum Circuit Results:",
            f"  Success: {self.success}",
            f"  Iterations: {self.n_iterations}",
            f"  Optimal Cost: {self.optimal_cost:.6f}",
            f"  Message: {self.message}",
        ]
        if self.history:
            lines.append(f"  Initial Cost: {self.history[0]:.6f}")
            lines.append(f"  Final Cost: {self.history[-1]:.6f}")
        return "\n".join(lines)


# ============================================================================
# VARIATIONAL LAYER CLASS
# ============================================================================

class VariationalLayer:
    """
    A variational layer for VQC
    
    Consists of single-qubit rotations followed by entanglement.
    
    Example:
        >>> layer = VariationalLayer(n_qubits=2, entangler='cnot')
        >>> circuit = QuantumCircuit(2)
        >>> params = np.random.randn(6)
        >>> layer.apply(circuit, params)
    """
    
    def __init__(self, n_qubits: int, entangler: str = 'cnot'):
        """
        Initialize a variational layer
        
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
        return f"VariationalLayer(n_qubits={self.n_qubits}, entangler={self.entangler})"


# ============================================================================
# VARIATIONAL QUANTUM CIRCUIT
# ============================================================================

class VQC:
    """
    Variational Quantum Circuit
    
    A parameterized quantum circuit that can be optimized for various tasks
    including classification and Hamiltonian ground state finding.
    
    Example:
        >>> from psiqit.qml import VQC
        >>> # Create a VQC for classification
        >>> vqc = VQC(n_qubits=2, n_layers=3)
        >>> # Optimize for a specific cost function
        >>> def cost(params):
        ...     vqc.set_params(params)
        ...     return vqc.measure()
        >>> result = vqc.optimize(cost_function=cost, n_iterations=50)
        >>> print(result.optimal_cost)
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 3,
        entangler: str = 'cnot',
        measurement: str = 'z'
    ):
        """
        Initialize Variational Quantum Circuit
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            entangler: Type of entangling gate ('cnot', 'cz', 'swap')
            measurement: Measurement basis ('z', 'x', 'y')
        """
        validate_qubits(n_qubits)
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangler = entangler
        self.measurement = measurement
        
        # Parameters per layer
        self.params_per_layer = n_qubits * 3
        self.n_params = self.params_per_layer * n_layers
        
        # Create layers
        self.layers = [VariationalLayer(n_qubits, entangler) for _ in range(n_layers)]
        
        # Initialize parameters
        self._params = None
        self._init_random()
        
        logger.info(f"VQC initialized: {n_qubits} qubits, {n_layers} layers, entangler={entangler}")
    
    def _init_random(self, seed: Optional[int] = None):
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
        Set the parameters of the VQC
        
        Args:
            params: Parameter vector
        """
        if len(params) != self.n_params:
            raise ValueError(f"Parameter length {len(params)} does not match {self.n_params}")
        self._params = params
    
    def build_circuit(self, input_state: Optional[Union[Ket, np.ndarray]] = None) -> QuantumCircuit:
        """
        Build the VQC circuit
        
        Args:
            input_state: Input state (if None, uses |0...0⟩)
            
        Returns:
            QuantumCircuit: The VQC circuit
            
        Example:
            >>> circ = vqc.build_circuit()
            >>> print(circ.draw())
        """
        circuit = QuantumCircuit(self.n_qubits)
        
        # Prepare input state if provided
        if input_state is not None:
            if isinstance(input_state, Ket):
                state_data = input_state.data
            else:
                state_data = np.array(input_state, dtype=complex)
            
            # Simple state preparation
            for i in range(min(self.n_qubits, len(state_data))):
                if i < len(state_data):
                    # Use amplitude to set rotation
                    angle = 2 * np.arccos(np.abs(state_data[i]))
                    circuit.ry(i, angle)
        
        # Apply variational layers
        for layer_idx, layer in enumerate(self.layers):
            start_idx = layer_idx * self.params_per_layer
            layer.apply(circuit, self._params, start_idx)
        
        return circuit
    
    def get_state(self, input_state: Optional[Union[Ket, np.ndarray]] = None) -> Ket:
        """
        Get the output state of the VQC
        
        Args:
            input_state: Input state
            
        Returns:
            Ket: Output state
            
        Example:
            >>> state = vqc.get_state()
            >>> print(state)
        """
        circuit = self.build_circuit(input_state)
        return circuit.run()
    
    def measure(self, input_state: Optional[Union[Ket, np.ndarray]] = None) -> float:
        """
        Measure the expectation value of the VQC output
        
        Args:
            input_state: Input state
            
        Returns:
            float: Expectation value (-1 to 1)
            
        Example:
            >>> expectation = vqc.measure()
            >>> print(expectation)
        """
        state = self.get_state(input_state)
        
        if self.measurement == 'z':
            # Measure Z expectation on the first qubit
            if state.dim == 2:
                return float(np.abs(state.data[0])**2 - np.abs(state.data[1])**2)
            else:
                # For multi-qubit, use first qubit
                prob_0 = 0.0
                prob_1 = 0.0
                for i, amp in enumerate(state.data):
                    if (i >> 0) & 1:
                        prob_1 += np.abs(amp)**2
                    else:
                        prob_0 += np.abs(amp)**2
                return float(prob_0 - prob_1)
        
        elif self.measurement == 'x':
            H = hadamard().data
            state_x = Ket(H @ state.data)
            if state_x.dim == 2:
                return float(np.abs(state_x.data[0])**2 - np.abs(state_x.data[1])**2)
            return 0.0
        
        elif self.measurement == 'y':
            from ..quantum.operator import pauli_y
            Y = pauli_y().data
            expectation = np.vdot(state.data, Y @ state.data).real
            return float(expectation)
        
        else:
            raise ValueError(f"Unknown measurement basis: {self.measurement}")
    
    def evaluate_cost(
        self,
        params: Optional[np.ndarray] = None,
        input_state: Optional[Union[Ket, np.ndarray]] = None
    ) -> float:
        """
        Evaluate the cost function for given parameters
        
        Args:
            params: Parameters (if None, uses current params)
            input_state: Input state
            
        Returns:
            float: Cost value
            
        Example:
            >>> cost = vqc.evaluate_cost()
            >>> print(cost)
        """
        if params is not None:
            self.set_params(params)
        
        return self.measure(input_state)
    
    def parameter_shift_gradient(
        self,
        param_idx: int,
        input_state: Optional[Union[Ket, np.ndarray]] = None,
        epsilon: float = 1e-5
    ) -> float:
        """
        Compute the gradient using parameter-shift rule
        
        Args:
            param_idx: Parameter index
            input_state: Input state
            epsilon: Small shift for finite difference
            
        Returns:
            float: Gradient value
            
        Example:
            >>> grad = vqc.parameter_shift_gradient(0)
            >>> print(grad)
        """
        # Finite difference approximation
        params_plus = self._params.copy()
        params_minus = self._params.copy()
        params_plus[param_idx] += epsilon
        params_minus[param_idx] -= epsilon
        
        # Save current params
        current_params = self._params.copy()
        
        # Compute cost with shifted params
        self.set_params(params_plus)
        cost_plus = self.evaluate_cost(input_state=input_state)
        
        self.set_params(params_minus)
        cost_minus = self.evaluate_cost(input_state=input_state)
        
        # Restore params
        self.set_params(current_params)
        
        return (cost_plus - cost_minus) / (2 * epsilon)
    
    def gradient(
        self,
        input_state: Optional[Union[Ket, np.ndarray]] = None
    ) -> List[float]:
        """
        Compute the full gradient
        
        Args:
            input_state: Input state
            
        Returns:
            List[float]: Gradient vector
            
        Example:
            >>> grad = vqc.gradient()
            >>> print(len(grad))  # n_params
        """
        grad = np.zeros(self.n_params)
        
        for i in range(self.n_params):
            grad[i] = self.parameter_shift_gradient(i, input_state)
        
        return grad.tolist()
    
    def optimize(
        self,
        cost_function: Optional[Callable] = None,
        n_iterations: int = 100,
        learning_rate: float = 0.1,
        input_state: Optional[Union[Ket, np.ndarray]] = None,
        verbose: bool = True
    ) -> VQCResult:
        """
        Optimize the VQC parameters using gradient descent
        
        Args:
            cost_function: Cost function (if None, uses evaluate_cost)
            n_iterations: Number of iterations
            learning_rate: Learning rate
            input_state: Input state
            verbose: Print progress
            
        Returns:
            VQCResult: Optimization results
            
        Example:
            >>> result = vqc.optimize(n_iterations=50)
            >>> print(result.optimal_cost)
        """
        if cost_function is None:
            cost_function = lambda: self.evaluate_cost(input_state=input_state)
        
        logger.info(f"Starting VQC optimization: {n_iterations} iterations, lr={learning_rate}")
        
        history = []
        
        for iteration in range(n_iterations):
            # Compute gradient
            grad = self.gradient(input_state)
            
            # Update parameters
            self._params -= learning_rate * np.array(grad)
            
            # Evaluate cost
            cost = cost_function()
            history.append(cost)
            
            if verbose and (iteration + 1) % max(1, n_iterations // 10) == 0:
                logger.info(f"Iteration {iteration+1}/{n_iterations}: cost={cost:.6f}")
        
        # Final cost
        final_cost = cost_function()
        
        logger.info(f"Optimization completed: final_cost={final_cost:.6f}")
        
        return VQCResult(
            optimal_params=self._params.copy(),
            optimal_cost=final_cost,
            n_iterations=n_iterations,
            history=history,
            success=True,
            message="Optimization completed successfully"
        )
    
    def __repr__(self) -> str:
        return f"VQC(n_qubits={self.n_qubits}, n_layers={self.n_layers}, entangler={self.entangler})"
    
    def __str__(self) -> str:
        lines = [
            "Variational Quantum Circuit:",
            f"  Qubits: {self.n_qubits}",
            f"  Layers: {self.n_layers}",
            f"  Entangler: {self.entangler}",
            f"  Measurement: {self.measurement}",
            f"  Parameters: {self.n_params}",
        ]
        return "\n".join(lines)


# ============================================================================
# HAMILTONIAN VQE (Variational Quantum Eigensolver)
# ============================================================================

class HamiltonianVQE(VQC):
    """
    Hamiltonian Variational Quantum Eigensolver (VQE)
    
    A specialized VQC for finding the ground state energy of a Hamiltonian.
    
    Example:
        >>> from psiqit.qml import HamiltonianVQE
        >>> from psiqit.quantum import pauli_z
        >>> # Create Hamiltonian (e.g., for H2 molecule)
        >>> hamiltonian = {'Z0Z1': 1.0, 'Z0': 0.5, 'Z1': -0.5}
        >>> vqe = HamiltonianVQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=2)
        >>> result = vqe.run(n_iterations=50)
        >>> print(result.optimal_cost)
    """
    
    def __init__(
        self,
        n_qubits: int,
        hamiltonian: Dict[str, float],
        n_layers: int = 3,
        entangler: str = 'cnot'
    ):
        """
        Initialize Hamiltonian VQE
        
        Args:
            n_qubits: Number of qubits
            hamiltonian: Hamiltonian as Pauli strings
                        {'Z0Z1': 1.0, 'X0': 0.5, ...}
            n_layers: Number of variational layers
            entangler: Type of entangling gate
        """
        super().__init__(n_qubits, n_layers, entangler, measurement='z')
        
        self.hamiltonian = hamiltonian
        self._pauli_matrices = {
            'I': np.eye(2, dtype=complex),
            'X': pauli_x().data,
            'Y': pauli_y().data,
            'Z': pauli_z().data,
        }
        
        logger.info(f"HamiltonianVQE initialized: {len(hamiltonian)} terms")
    
    def _expectation_z(self, state: Ket) -> float:
        """
        Compute expectation value of Z on first qubit
        
        Args:
            state: Quantum state
            
        Returns:
            float: ⟨Z⟩
        """
        if state.dim == 2:
            return float(np.abs(state.data[0])**2 - np.abs(state.data[1])**2)
        else:
            # For multi-qubit, measure first qubit
            prob_0 = 0.0
            prob_1 = 0.0
            for i, amp in enumerate(state.data):
                if (i >> 0) & 1:
                    prob_1 += np.abs(amp)**2
                else:
                    prob_0 += np.abs(amp)**2
            return float(prob_0 - prob_1)
    
    def _expectation_x(self, state: Ket) -> float:
        """
        Compute expectation value of X on first qubit
        
        Args:
            state: Quantum state
            
        Returns:
            float: ⟨X⟩
        """
        H = hadamard().data
        state_x = Ket(H @ state.data)
        return self._expectation_z(state_x)
    
    def _expectation_y(self, state: Ket) -> float:
        """
        Compute expectation value of Y on first qubit
        
        Args:
            state: Quantum state
            
        Returns:
            float: ⟨Y⟩
        """
        # Apply phase gate to convert Y to Z measurement
        from ..quantum.operator import phase
        S = phase(np.pi/2).data
        state_y = Ket(S @ state.data)
        return self._expectation_z(state_y)
    
    def _expectation_pauli_string(self, state: Ket, pauli_string: str) -> float:
        """
        Compute expectation value of a Pauli string
        
        Args:
            state: Quantum state
            pauli_string: String of Pauli operators (e.g., 'Z0Z1')
            
        Returns:
            float: ⟨Pauli_string⟩
        """
        # Parse the Pauli string
        import re
        paulis = {}
        for match in re.finditer(r'([IXYZ])(\d+)', pauli_string):
            op, qubit = match.group(1), int(match.group(2))
            paulis[qubit] = op
        
        # If no paulis found, return 1 (identity)
        if not paulis:
            return 1.0
        
        # Compute expectation by measuring the Pauli string
        # This is a simplified version
        # In practice, we would need to apply basis transformations
        
        # For now, we assume we're measuring Z on all qubits
        # and convert other Paulis using rotations
        
        # This is a placeholder - a full implementation would require
        # applying the appropriate basis transformations
        
        result = 0.0
        # If all are Z, compute directly from state
        all_z = all(op == 'Z' for op in paulis.values())
        
        if all_z:
            # Compute expectation of product of Z operators
            # This is the parity of the qubits
            prob_even = 0.0
            prob_odd = 0.0
            for i, amp in enumerate(state.data):
                parity = 0
                for qubit in paulis.keys():
                    if (i >> qubit) & 1:
                        parity ^= 1
                if parity == 0:
                    prob_even += np.abs(amp)**2
                else:
                    prob_odd += np.abs(amp)**2
            result = prob_even - prob_odd
        else:
            # For non-Z Paulis, need basis transformation
            # This is a simplified version
            logger.warning(f"Non-Z Pauli string {pauli_string} not fully implemented")
            result = 0.0
        
        return float(result)
    
    def evaluate_hamiltonian(self, params: Optional[np.ndarray] = None) -> float:
        """
        Evaluate the Hamiltonian expectation value
        
        Args:
            params: Parameters (if None, uses current params)
            
        Returns:
            float: Hamiltonian expectation value
            
        Example:
            >>> energy = vqe.evaluate_hamiltonian()
            >>> print(energy)
        """
        if params is not None:
            self.set_params(params)
        
        # Get the state
        state = self.get_state()
        
        # Compute expectation of each term
        total_energy = 0.0
        for pauli_string, coeff in self.hamiltonian.items():
            if pauli_string == 'I':
                # Identity term
                total_energy += coeff
            else:
                # Pauli string term
                expectation = self._expectation_pauli_string(state, pauli_string)
                total_energy += coeff * expectation
        
        return float(total_energy)
    
    def run(
        self,
        n_iterations: int = 100,
        learning_rate: float = 0.1,
        verbose: bool = True
    ) -> VQCResult:
        """
        Run the VQE optimization
        
        Args:
            n_iterations: Number of iterations
            learning_rate: Learning rate
            verbose: Print progress
            
        Returns:
            VQCResult: Optimization results
            
        Example:
            >>> result = vqe.run(n_iterations=50)
            >>> print(result.optimal_cost)
        """
        logger.info(f"Running VQE: {n_iterations} iterations")
        
        # Define cost function
        def cost_function():
            return self.evaluate_hamiltonian()
        
        # Use the optimize method from VQC
        result = self.optimize(
            cost_function=cost_function,
            n_iterations=n_iterations,
            learning_rate=learning_rate,
            verbose=verbose
        )
        
        # Add VQE specific info
        result.message = f"VQE completed with energy {result.optimal_cost:.6f}"
        
        logger.info(f"VQE completed: energy={result.optimal_cost:.6f}")
        
        return result
    
    def get_energy_spectrum(self, n_states: int = 5) -> List[float]:
        """
        Get the energy spectrum of the Hamiltonian (classical computation)
        
        Args:
            n_states: Number of lowest states to return
            
        Returns:
            List[float]: Energy eigenvalues
            
        Example:
            >>> spectrum = vqe.get_energy_spectrum()
            >>> print(spectrum[0])  # Ground state energy
        """
        from ..info.entropy import von_neumann_entropy
        
        # Build the Hamiltonian matrix
        dim = 2 ** self.n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        
        for pauli_string, coeff in self.hamiltonian.items():
            if pauli_string == 'I':
                H += coeff * np.eye(dim, dtype=complex)
            else:
                # Build Pauli string matrix
                P = self._build_pauli_matrix(pauli_string)
                H += coeff * P
        
        # Compute eigenvalues
        eigvals = np.linalg.eigvalsh(H)
        return eigvals[:n_states].tolist()
    
    def _build_pauli_matrix(self, pauli_string: str) -> np.ndarray:
        """
        Build the matrix for a Pauli string
        
        Args:
            pauli_string: String of Pauli operators
            
        Returns:
            np.ndarray: Pauli string matrix
        """
        import re
        
        dim = 2 ** self.n_qubits
        matrix = np.eye(dim, dtype=complex)
        
        # Parse the Pauli string
        paulis = {}
        for match in re.finditer(r'([IXYZ])(\d+)', pauli_string):
            op, qubit = match.group(1), int(match.group(2))
            paulis[qubit] = op
        
        # Build the tensor product
        for qubit in range(self.n_qubits):
            op = paulis.get(qubit, 'I')
            if op == 'I':
                op_matrix = np.eye(2, dtype=complex)
            elif op == 'X':
                op_matrix = pauli_x().data
            elif op == 'Y':
                op_matrix = pauli_y().data
            elif op == 'Z':
                op_matrix = pauli_z().data
            else:
                raise ValueError(f"Unknown Pauli: {op}")
            
            matrix = np.kron(matrix, op_matrix)
        
        return matrix
    
    def __repr__(self) -> str:
        return f"HamiltonianVQE(n_qubits={self.n_qubits}, n_terms={len(self.hamiltonian)}, n_layers={self.n_layers})"
    
    def __str__(self) -> str:
        lines = [
            "Hamiltonian VQE:",
            f"  Qubits: {self.n_qubits}",
            f"  Hamiltonian Terms: {len(self.hamiltonian)}",
            f"  Layers: {self.n_layers}",
            f"  Entangler: {self.entangler}",
        ]
        return "\n".join(lines)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'VQCResult',
    'VariationalLayer',
    'VQC',
    'HamiltonianVQE',
]