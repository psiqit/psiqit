# psiqit/variational/vqe.py

"""
Variational Quantum Eigensolver (VQE)
Find the ground state energy of a Hamiltonian
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, zero, one, plus, minus, basis
from ..quantum.operator import Operator, identity, pauli_x, pauli_y, pauli_z, hadamard, rx, ry, rz, cnot, cz
from ..qml.vqc import VQC, VQCResult
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# VQE RESULT CLASS
# ============================================================================

@dataclass
class VQEResult:
    """
    Result container for VQE optimization
    
    Attributes:
        optimal_params: Optimal parameters found
        optimal_energy: Optimal energy (ground state energy)
        n_iterations: Number of iterations
        history: History of energies during optimization
        success: Whether optimization was successful
        message: Additional message
        final_state: Final state (if available)
    """
    optimal_params: Optional[np.ndarray] = None
    optimal_energy: float = 0.0
    n_iterations: int = 0
    history: List[float] = field(default_factory=list)
    success: bool = True
    message: str = ""
    final_state: Optional[Ket] = None
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"VQEResult(iterations={self.n_iterations}, status={status}, energy={self.optimal_energy:.6f})"
    
    def __str__(self) -> str:
        lines = [
            "VQE Results:",
            f"  Success: {self.success}",
            f"  Iterations: {self.n_iterations}",
            f"  Optimal Energy: {self.optimal_energy:.6f}",
            f"  Message: {self.message}",
        ]
        if self.history:
            lines.append(f"  Initial Energy: {self.history[0]:.6f}")
            lines.append(f"  Final Energy: {self.history[-1]:.6f}")
        return "\n".join(lines)


# ============================================================================
# VQE CLASS
# ============================================================================

class VQE:
    """
    Variational Quantum Eigensolver (VQE)
    
    Finds the ground state energy of a Hamiltonian by optimizing a
    parameterized quantum circuit.
    
    Example:
        >>> from psiqit.variational import VQE
        >>> # Define Hamiltonian for a 2-qubit system
        >>> hamiltonian = {'Z0Z1': 1.0, 'Z0': 0.5, 'Z1': -0.5}
        >>> vqe = VQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=2)
        >>> result = vqe.run(n_iterations=100)
        >>> print(result.optimal_energy)
    """
    
    def __init__(
        self,
        n_qubits: int,
        hamiltonian: Dict[str, float],
        n_layers: int = 2,
        optimizer: str = 'gradient',
        entangler: str = 'cnot'
    ):
        """
        Initialize VQE
        
        Args:
            n_qubits: Number of qubits
            hamiltonian: Hamiltonian as Pauli strings
                        {'Z0Z1': 1.0, 'X0': 0.5, ...}
            n_layers: Number of variational layers
            optimizer: Optimization method ('gradient', 'cobyla', 'nelder_mead')
            entangler: Type of entangling gate ('cnot', 'cz', 'swap')
        """
        validate_qubits(n_qubits)
        
        self.n_qubits = n_qubits
        self.hamiltonian = hamiltonian
        self.n_layers = n_layers
        self.optimizer = optimizer
        self.entangler = entangler
        
        # Create VQC (Variational Quantum Circuit)
        self.vqc = VQC(
            n_qubits=n_qubits,
            n_layers=n_layers,
            entangler=entangler,
            measurement='z'
        )
        
        # Pauli matrices for expectation calculations
        self._pauli_matrices = {
            'I': np.eye(2, dtype=complex),
            'X': pauli_x().data,
            'Y': pauli_y().data,
            'Z': pauli_z().data,
        }
        
        # Initialize parameters
        self._params = None
        self._init_random()
        
        logger.info(f"VQE initialized: {n_qubits} qubits, {len(hamiltonian)} terms, {n_layers} layers")
    
    def _init_random(self, seed: Optional[int] = None):
        """
        Initialize parameters randomly
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self._params = np.random.uniform(-np.pi, np.pi, size=self.vqc.n_params)
        self.vqc.set_params(self._params)
    
    def build_circuit(self, params: Optional[np.ndarray] = None) -> QuantumCircuit:
        """
        Build the VQE circuit
        
        Args:
            params: Parameters (if None, uses current params)
            
        Returns:
            QuantumCircuit: VQE circuit
            
        Example:
            >>> circ = vqe.build_circuit()
            >>> print(circ.draw())
        """
        if params is not None:
            self.vqc.set_params(params)
        
        return self.vqc.build_circuit()
    
    def _expectation_pauli(self, state: Ket, pauli_string: str) -> float:
        """
        Compute expectation value of a Pauli string
        
        Args:
            state: Quantum state
            pauli_string: String of Pauli operators (e.g., 'Z0Z1')
            
        Returns:
            float: Expectation value
            
        Example:
            >>> state = Ket([1, 0, 0, 0])
            >>> exp = vqe._expectation_pauli(state, 'Z0Z1')
            >>> print(exp)  # 1.0
        """
        import re
        
        # Parse the Pauli string
        paulis = {}
        for match in re.finditer(r'([IXYZ])(\d+)', pauli_string):
            op, qubit = match.group(1), int(match.group(2))
            paulis[qubit] = op
        
        # If no paulis found, return 1 (identity)
        if not paulis:
            return 1.0
        
        # Check if all are Z (diagonal in computational basis)
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
            return float(prob_even - prob_odd)
        
        # For non-Z Paulis, need basis transformation
        # This is a simplified version
        # In practice, we would apply the appropriate basis transformations
        
        # For X, apply Hadamard
        # For Y, apply phase gate
        # For general Pauli strings, we need to handle each qubit
        
        # Build the full Pauli operator
        dim = 2 ** self.n_qubits
        full_operator = np.eye(dim, dtype=complex)
        
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
            
            full_operator = np.kron(full_operator, op_matrix)
        
        # Compute expectation
        expectation = np.vdot(state.data, full_operator @ state.data).real
        return float(expectation)
    
    def evaluate_energy(self, params: Optional[np.ndarray] = None) -> float:
        """
        Evaluate the Hamiltonian expectation value
        
        Args:
            params: Parameters (if None, uses current params)
            
        Returns:
            float: Energy expectation value
            
        Example:
            >>> energy = vqe.evaluate_energy()
            >>> print(energy)
        """
        if params is not None:
            self.vqc.set_params(params)
        
        # Get the state
        state = self.vqc.get_state()
        
        # Compute expectation of each term
        total_energy = 0.0
        for pauli_string, coeff in self.hamiltonian.items():
            if pauli_string == 'I':
                # Identity term
                total_energy += coeff
            else:
                # Pauli string term
                expectation = self._expectation_pauli(state, pauli_string)
                total_energy += coeff * expectation
        
        return float(total_energy)
    
    def _gradient(self, params: np.ndarray, eps: float = 1e-5) -> List[float]:
        """
        Compute the gradient of the energy with respect to parameters
        
        Args:
            params: Parameters
            eps: Step size for finite difference
            
        Returns:
            List[float]: Gradient vector
            
        Example:
            >>> grad = vqe._gradient(params)
            >>> print(len(grad))  # n_params
        """
        grad = np.zeros(len(params))
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            
            energy_plus = self.evaluate_energy(params_plus)
            energy_minus = self.evaluate_energy(params_minus)
            
            grad[i] = (energy_plus - energy_minus) / (2 * eps)
        
        return grad.tolist()
    
    def run(
        self,
        n_iterations: int = 100,
        learning_rate: float = 0.1,
        verbose: bool = True,
        initial_params: Optional[np.ndarray] = None
    ) -> VQEResult:
        """
        Run the VQE optimization
        
        Args:
            n_iterations: Number of iterations
            learning_rate: Learning rate for gradient descent
            verbose: Print progress
            initial_params: Initial parameters (if None, uses random)
            
        Returns:
            VQEResult: Optimization results
            
        Example:
            >>> result = vqe.run(n_iterations=100)
            >>> print(result.optimal_energy)
        """
        logger.info(f"Starting VQE optimization: {n_iterations} iterations")
        
        # Set initial parameters
        if initial_params is not None:
            self._params = initial_params
            self.vqc.set_params(initial_params)
        else:
            self._init_random()
        
        history = []
        params = self._params.copy()
        
        # Run optimization
        if self.optimizer == 'gradient':
            # Gradient descent
            for iteration in range(n_iterations):
                # Compute gradient
                grad = self._gradient(params)
                
                # Update parameters
                params = params - learning_rate * np.array(grad)
                
                # Evaluate energy
                energy = self.evaluate_energy(params)
                history.append(energy)
                
                if verbose and (iteration + 1) % max(1, n_iterations // 10) == 0:
                    logger.info(f"Iteration {iteration+1}/{n_iterations}: energy={energy:.6f}")
        
        elif self.optimizer == 'cobyla':
            # COBYLA optimizer (requires scipy)
            try:
                from scipy.optimize import minimize
                
                def cost_func(x):
                    return self.evaluate_energy(x)
                
                result = minimize(
                    cost_func,
                    params,
                    method='COBYLA',
                    options={'maxiter': n_iterations, 'disp': verbose}
                )
                
                params = result.x
                history = [cost_func(p) for p in params]  # Not the full history
                
                logger.info(f"COBYLA optimization completed: {result.message}")
                
            except ImportError:
                logger.warning("scipy not available, falling back to gradient descent")
                # Fallback to gradient descent
                for iteration in range(n_iterations):
                    grad = self._gradient(params)
                    params = params - learning_rate * np.array(grad)
                    energy = self.evaluate_energy(params)
                    history.append(energy)
        
        elif self.optimizer == 'nelder_mead':
            # Nelder-Mead optimizer (requires scipy)
            try:
                from scipy.optimize import minimize
                
                def cost_func(x):
                    return self.evaluate_energy(x)
                
                result = minimize(
                    cost_func,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': n_iterations, 'disp': verbose}
                )
                
                params = result.x
                history = [cost_func(p) for p in params]  # Not the full history
                
                logger.info(f"Nelder-Mead optimization completed: {result.message}")
                
            except ImportError:
                logger.warning("scipy not available, falling back to gradient descent")
                # Fallback to gradient descent
                for iteration in range(n_iterations):
                    grad = self._gradient(params)
                    params = params - learning_rate * np.array(grad)
                    energy = self.evaluate_energy(params)
                    history.append(energy)
        
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        
        # Final energy
        final_energy = self.evaluate_energy(params)
        
        # Get final state
        self.vqc.set_params(params)
        final_state = self.vqc.get_state()
        
        logger.info(f"VQE completed: final_energy={final_energy:.6f}")
        
        return VQEResult(
            optimal_params=params,
            optimal_energy=final_energy,
            n_iterations=n_iterations,
            history=history,
            success=True,
            message=f"VQE completed with energy {final_energy:.6f}",
            final_state=final_state
        )
    
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
    
    def get_ground_state(self) -> Ket:
        """
        Get the ground state (after optimization)
        
        Returns:
            Ket: Ground state
            
        Example:
            >>> state = vqe.get_ground_state()
            >>> print(state)
        """
        return self.vqc.get_state()
    
    def get_params(self) -> np.ndarray:
        """Get the current parameters"""
        return self._params.copy()
    
    def set_params(self, params: np.ndarray):
        """Set the parameters"""
        self._params = params
        self.vqc.set_params(params)
    
    def __repr__(self) -> str:
        return f"VQE(n_qubits={self.n_qubits}, n_terms={len(self.hamiltonian)}, n_layers={self.n_layers})"
    
    def __str__(self) -> str:
        lines = [
            "Variational Quantum Eigensolver:",
            f"  Qubits: {self.n_qubits}",
            f"  Hamiltonian Terms: {len(self.hamiltonian)}",
            f"  Layers: {self.n_layers}",
            f"  Optimizer: {self.optimizer}",
            f"  Entangler: {self.entangler}",
        ]
        return "\n".join(lines)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'VQEResult',
    'VQE',
]