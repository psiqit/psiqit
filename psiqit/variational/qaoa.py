# psiqit/variational/qaoa.py

"""
Quantum Approximate Optimization Algorithm (QAOA)
Solve combinatorial optimization problems
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
# QAOA RESULT CLASS
# ============================================================================

@dataclass
class QAOResult:
    """
    Result container for QAOA optimization
    
    Attributes:
        optimal_params: Optimal parameters (gamma, beta)
        optimal_energy: Optimal energy (minimum of cost Hamiltonian)
        n_iterations: Number of iterations
        history: History of energies during optimization
        success: Whether optimization was successful
        message: Additional message
        final_state: Final state (if available)
        optimal_gamma: Optimal gamma parameters
        optimal_beta: Optimal beta parameters
    """
    optimal_params: Optional[np.ndarray] = None
    optimal_energy: float = 0.0
    n_iterations: int = 0
    history: List[float] = field(default_factory=list)
    success: bool = True
    message: str = ""
    final_state: Optional[Ket] = None
    optimal_gamma: Optional[np.ndarray] = None
    optimal_beta: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Extract gamma and beta from params if available"""
        if self.optimal_params is not None:
            p = len(self.optimal_params) // 2
            self.optimal_gamma = self.optimal_params[:p]
            self.optimal_beta = self.optimal_params[p:]
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"QAOResult(p={len(self.optimal_gamma) if self.optimal_gamma else 0}, status={status}, energy={self.optimal_energy:.6f})"
    
    def __str__(self) -> str:
        lines = [
            "QAOA Results:",
            f"  Success: {self.success}",
            f"  Iterations: {self.n_iterations}",
            f"  Optimal Energy: {self.optimal_energy:.6f}",
            f"  Message: {self.message}",
        ]
        if self.history:
            lines.append(f"  Initial Energy: {self.history[0]:.6f}")
            lines.append(f"  Final Energy: {self.history[-1]:.6f}")
        if self.optimal_gamma is not None:
            lines.append(f"  Optimal Gamma: {self.optimal_gamma.tolist()}")
        if self.optimal_beta is not None:
            lines.append(f"  Optimal Beta: {self.optimal_beta.tolist()}")
        return "\n".join(lines)


# ============================================================================
# QAOA CLASS
# ============================================================================

class QAOA:
    """
    Quantum Approximate Optimization Algorithm (QAOA)
    
    Solves combinatorial optimization problems by alternating between
    cost and mixer Hamiltonians.
    
    Example:
        >>> from psiqit.variational import QAOA
        >>> # MaxCut problem on a triangle graph
        >>> edges = [(0, 1), (1, 2), (0, 2)]
        >>> hamiltonian = maxcut_hamiltonian(edges)
        >>> qaoa = QAOA(n_qubits=3, hamiltonian=hamiltonian, p=2)
        >>> result = qaoa.run(n_iterations=100)
        >>> print(result.optimal_energy)
    """
    
    def __init__(
        self,
        n_qubits: int,
        hamiltonian: Dict[str, float],
        p: int = 2,
        optimizer: str = 'gradient',
        mixer: str = 'x'
    ):
        """
        Initialize QAOA
        
        Args:
            n_qubits: Number of qubits
            hamiltonian: Cost Hamiltonian as Pauli strings
                        {'Z0Z1': 1.0, 'Z0': 0.5, ...}
            p: Number of QAOA layers (depth)
            optimizer: Optimization method ('gradient', 'cobyla', 'nelder_mead')
            mixer: Mixer Hamiltonian type ('x', 'xy', 'custom')
        """
        validate_qubits(n_qubits)
        
        self.n_qubits = n_qubits
        self.hamiltonian = hamiltonian
        self.p = p
        self.optimizer = optimizer
        self.mixer = mixer
        
        # Number of parameters: p gamma + p beta
        self.n_params = 2 * p
        
        # Pauli matrices
        self._pauli_matrices = {
            'I': np.eye(2, dtype=complex),
            'X': pauli_x().data,
            'Y': pauli_y().data,
            'Z': pauli_z().data,
        }
        
        # Initialize parameters
        self._params = None
        self._init_random()
        
        # Parse Hamiltonian terms
        self._parse_hamiltonian()
        
        logger.info(f"QAOA initialized: {n_qubits} qubits, {len(hamiltonian)} terms, p={p}, mixer={mixer}")
    
    def _init_random(self, seed: Optional[int] = None):
        """
        Initialize parameters randomly
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize gamma and beta randomly between 0 and pi
        self._params = np.random.uniform(0, np.pi, size=self.n_params)
    
    def _parse_hamiltonian(self):
        """
        Parse the Hamiltonian into Pauli strings for efficient application
        """
        import re
        
        self._hamiltonian_terms = []
        
        for pauli_string, coeff in self.hamiltonian.items():
            if pauli_string == 'I':
                # Identity term - just add energy offset
                self._hamiltonian_terms.append({
                    'paulis': {},
                    'coeff': coeff,
                    'is_identity': True
                })
            else:
                # Parse Pauli string
                paulis = {}
                for match in re.finditer(r'([IXYZ])(\d+)', pauli_string):
                    op, qubit = match.group(1), int(match.group(2))
                    paulis[qubit] = op
                
                self._hamiltonian_terms.append({
                    'paulis': paulis,
                    'coeff': coeff,
                    'is_identity': False,
                    'string': pauli_string
                })
    
    def _apply_pauli_string(self, circuit: QuantumCircuit, paulis: Dict[int, str], angle: float):
        """
        Apply a Pauli string rotation to the circuit
        
        Args:
            circuit: QuantumCircuit object
            paulis: Dictionary of {qubit: Pauli operator}
            angle: Rotation angle
        """
        # For each qubit, apply the appropriate rotation
        for qubit, op in paulis.items():
            if op == 'Z':
                # Z rotation (phase gate)
                circuit.rz(qubit, angle)
            elif op == 'X':
                # X rotation
                circuit.rx(qubit, angle)
            elif op == 'Y':
                # Y rotation
                circuit.ry(qubit, angle)
        
        # For multi-qubit terms (like ZZ), we need to use CNOTs
        # This is a simplified version - for full ZZ terms, we would need
        # to use the standard decomposition with CNOTs
        
        # If there are multiple Zs, we need to implement ZZ rotation
        z_qubits = [q for q, op in paulis.items() if op == 'Z']
        if len(z_qubits) >= 2:
            # Implement ZZ rotation using CNOTs
            # For a ZZ term, we use: CNOT, Rz, CNOT
            for i in range(len(z_qubits) - 1):
                q1 = z_qubits[i]
                q2 = z_qubits[i + 1]
                circuit.cx(q1, q2)
                circuit.rz(q2, angle)
                circuit.cx(q1, q2)
    
    def _apply_cost_layer(self, circuit: QuantumCircuit, gamma: float):
        """
        Apply the cost Hamiltonian layer
        
        Args:
            circuit: QuantumCircuit object
            gamma: Cost layer parameter
        """
        for term in self._hamiltonian_terms:
            if term['is_identity']:
                # Identity term - no effect on circuit
                continue
            
            paulis = term['paulis']
            coeff = term['coeff']
            
            # Apply the Pauli string rotation with angle gamma * coeff
            # For each Pauli string, we need to apply the corresponding rotation
            # This is a simplified implementation
            
            # For single-qubit terms
            if len(paulis) == 1:
                qubit, op = list(paulis.items())[0]
                angle = gamma * coeff
                if op == 'Z':
                    circuit.rz(qubit, angle)
                elif op == 'X':
                    circuit.rx(qubit, angle)
                elif op == 'Y':
                    circuit.ry(qubit, angle)
            else:
                # Multi-qubit terms - simplified
                # For each qubit in the term, apply the appropriate rotation
                for qubit, op in paulis.items():
                    angle = gamma * coeff / len(paulis)
                    if op == 'Z':
                        circuit.rz(qubit, angle)
                    elif op == 'X':
                        circuit.rx(qubit, angle)
                    elif op == 'Y':
                        circuit.ry(qubit, angle)
                
                # For ZZ terms, add CNOTs to create entanglement
                z_qubits = [q for q, op in paulis.items() if op == 'Z']
                if len(z_qubits) >= 2:
                    # Use CNOT chain for ZZ rotation
                    # This is a simplified version
                    for i in range(len(z_qubits) - 1):
                        q1 = z_qubits[i]
                        q2 = z_qubits[i + 1]
                        circuit.cx(q1, q2)
                        circuit.rz(q2, gamma * coeff)
                        circuit.cx(q1, q2)
    
    def _apply_mixer_layer(self, circuit: QuantumCircuit, beta: float):
        """
        Apply the mixer Hamiltonian layer
        
        Args:
            circuit: QuantumCircuit object
            beta: Mixer layer parameter
        """
        if self.mixer == 'x':
            # Standard X mixer: apply RX to all qubits
            for i in range(self.n_qubits):
                circuit.rx(i, 2 * beta)
        
        elif self.mixer == 'xy':
            # XY mixer
            for i in range(self.n_qubits):
                circuit.rx(i, beta)
            for i in range(self.n_qubits - 1):
                circuit.cx(i, i + 1)
                circuit.rx(i + 1, beta)
                circuit.cx(i, i + 1)
        
        else:
            # Default: X mixer
            for i in range(self.n_qubits):
                circuit.rx(i, 2 * beta)
    
    def build_circuit(self, params: Optional[np.ndarray] = None) -> QuantumCircuit:
        """
        Build the QAOA circuit
        
        Args:
            params: Parameters [gamma_0, beta_0, gamma_1, beta_1, ...]
                    If None, uses current params
            
        Returns:
            QuantumCircuit: QAOA circuit
            
        Example:
            >>> circ = qaoa.build_circuit()
            >>> print(circ.draw())
        """
        if params is None:
            params = self._params
        
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(params)}")
        
        # Extract gamma and beta
        gamma = params[:self.p]
        beta = params[self.p:]
        
        # Create circuit
        circuit = QuantumCircuit(self.n_qubits)
        
        # Initialize to |+⟩⊗n
        for i in range(self.n_qubits):
            circuit.h(i)
        
        # Apply QAOA layers
        for layer in range(self.p):
            # Cost layer
            self._apply_cost_layer(circuit, gamma[layer])
            
            # Mixer layer
            self._apply_mixer_layer(circuit, beta[layer])
        
        return circuit
    
    def evaluate_energy(self, params: Optional[np.ndarray] = None) -> float:
        """
        Evaluate the cost Hamiltonian expectation value
        
        Args:
            params: Parameters (if None, uses current params)
            
        Returns:
            float: Energy expectation value
            
        Example:
            >>> energy = qaoa.evaluate_energy()
            >>> print(energy)
        """
        if params is not None:
            # Temporarily set params
            old_params = self._params.copy()
            self._params = params
            energy = self._evaluate_energy_with_current_params()
            self._params = old_params
            return energy
        
        return self._evaluate_energy_with_current_params()
    
    def _evaluate_energy_with_current_params(self) -> float:
        """Evaluate energy with current parameters"""
        # Build circuit
        circuit = self.build_circuit()
        
        # Get state
        state = circuit.run()
        
        # Compute expectation of each term
        total_energy = 0.0
        for term in self._hamiltonian_terms:
            if term['is_identity']:
                total_energy += term['coeff']
            else:
                # Compute expectation of Pauli string
                expectation = self._expectation_pauli(state, term['paulis'])
                total_energy += term['coeff'] * expectation
        
        return float(total_energy)
    
    def _expectation_pauli(self, state: Ket, paulis: Dict[int, str]) -> float:
        """
        Compute expectation value of a Pauli string
        
        Args:
            state: Quantum state
            paulis: Dictionary of {qubit: Pauli operator}
            
        Returns:
            float: Expectation value
        """
        # Check if all are Z (diagonal in computational basis)
        all_z = all(op == 'Z' for op in paulis.values())
        
        if all_z:
            # Compute expectation of product of Z operators
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
        
        # For non-Z Paulis, build the full operator
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
    
    def _gradient(self, params: np.ndarray, eps: float = 1e-5) -> List[float]:
        """
        Compute the gradient of the energy with respect to parameters
        
        Args:
            params: Parameters
            eps: Step size for finite difference
            
        Returns:
            List[float]: Gradient vector
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
    ) -> QAOResult:
        """
        Run the QAOA optimization
        
        Args:
            n_iterations: Number of iterations
            learning_rate: Learning rate for gradient descent
            verbose: Print progress
            initial_params: Initial parameters (if None, uses random)
            
        Returns:
            QAOResult: Optimization results
            
        Example:
            >>> result = qaoa.run(n_iterations=100)
            >>> print(result.optimal_energy)
        """
        logger.info(f"Starting QAOA optimization: {n_iterations} iterations, p={self.p}")
        
        # Set initial parameters
        if initial_params is not None:
            self._params = initial_params
        else:
            self._init_random()
        
        params = self._params.copy()
        history = []
        
        # Run optimization
        if self.optimizer == 'gradient':
            # Gradient descent
            for iteration in range(n_iterations):
                # Compute gradient
                grad = self._gradient(params)
                
                # Update parameters
                params = params - learning_rate * np.array(grad)
                
                # Clip parameters to reasonable range
                params = np.clip(params, 0, 2 * np.pi)
                
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
                    params = np.clip(params, 0, 2 * np.pi)
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
                    params = np.clip(params, 0, 2 * np.pi)
                    energy = self.evaluate_energy(params)
                    history.append(energy)
        
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        
        # Final energy
        self._params = params
        final_energy = self.evaluate_energy(params)
        
        # Get final state
        circuit = self.build_circuit()
        final_state = circuit.run()
        
        logger.info(f"QAOA completed: final_energy={final_energy:.6f}")
        
        return QAOResult(
            optimal_params=params,
            optimal_energy=final_energy,
            n_iterations=n_iterations,
            history=history,
            success=True,
            message=f"QAOA completed with energy {final_energy:.6f}",
            final_state=final_state
        )
    
    def get_params(self) -> np.ndarray:
        """Get the current parameters"""
        return self._params.copy()
    
    def set_params(self, params: np.ndarray):
        """Set the parameters"""
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(params)}")
        self._params = params
    
    def get_gamma(self) -> np.ndarray:
        """Get the gamma parameters"""
        return self._params[:self.p].copy()
    
    def get_beta(self) -> np.ndarray:
        """Get the beta parameters"""
        return self._params[self.p:].copy()
    
    def __repr__(self) -> str:
        return f"QAOA(n_qubits={self.n_qubits}, n_terms={len(self.hamiltonian)}, p={self.p})"
    
    def __str__(self) -> str:
        lines = [
            "Quantum Approximate Optimization Algorithm:",
            f"  Qubits: {self.n_qubits}",
            f"  Hamiltonian Terms: {len(self.hamiltonian)}",
            f"  Layers (p): {self.p}",
            f"  Optimizer: {self.optimizer}",
            f"  Mixer: {self.mixer}",
        ]
        return "\n".join(lines)


# ============================================================================
# HAMILTONIAN BUILDERS
# ============================================================================

def maxcut_hamiltonian(edges: List[Tuple[int, int]]) -> Dict[str, float]:
    """
    Create MaxCut problem Hamiltonian
    
    H = -Σ_{(i,j)∈E} Z_i Z_j
    
    Args:
        edges: List of edges in the graph
        
    Returns:
        Dict: Hamiltonian as Pauli strings
        
    Example:
        >>> edges = [(0, 1), (1, 2), (0, 2)]  # Triangle graph
        >>> H = maxcut_hamiltonian(edges)
        >>> # H = -Z0Z1 - Z1Z2 - Z0Z2
    """
    hamiltonian = {}
    
    for i, j in edges:
        term = f"Z{i}Z{j}"
        hamiltonian[term] = hamiltonian.get(term, 0) - 1.0
    
    return hamiltonian


def qubo_to_hamiltonian(
    qubo: Dict[Tuple[int, int], float],
    n_qubits: int
) -> Dict[str, float]:
    """
    Convert QUBO (Quadratic Unconstrained Binary Optimization) to Ising Hamiltonian
    
    QUBO: minimize Σ Q_ij x_i x_j, x_i ∈ {0, 1}
    Ising: H = Σ J_ij Z_i Z_j + Σ h_i Z_i + constant
    
    Mapping: x_i = (1 - Z_i)/2
    
    Args:
        qubo: QUBO matrix {(i, j): Q_ij}
        n_qubits: Number of qubits
        
    Returns:
        Dict: Hamiltonian as Pauli strings
        
    Example:
        >>> Q = {(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0}
        >>> H = qubo_to_hamiltonian(Q, 2)
    """
    hamiltonian = {}
    constant = 0.0
    
    for (i, j), Qij in qubo.items():
        if i == j:
            # Linear term: Q_ii * x_i = Q_ii * (1 - Z_i)/2
            # = Q_ii/2 - Q_ii/2 * Z_i
            constant += Qij / 2
            term = f"Z{i}"
            hamiltonian[term] = hamiltonian.get(term, 0) - Qij / 2
        else:
            # Quadratic term: Q_ij * x_i * x_j
            # = Q_ij/4 * (1 - Z_i - Z_j + Z_i Z_j)
            constant += Qij / 4
            term_i = f"Z{i}"
            hamiltonian[term_i] = hamiltonian.get(term_i, 0) - Qij / 4
            term_j = f"Z{j}"
            hamiltonian[term_j] = hamiltonian.get(term_j, 0) - Qij / 4
            term_ij = f"Z{i}Z{j}"
            hamiltonian[term_ij] = hamiltonian.get(term_ij, 0) + Qij / 4
    
    # Add constant term
    if abs(constant) > 1e-12:
        hamiltonian['I'] = constant
    
    return hamiltonian


def maxcut_from_graph(adjacency_matrix: np.ndarray) -> Dict[str, float]:
    """
    Create MaxCut Hamiltonian from adjacency matrix
    
    Args:
        adjacency_matrix: Adjacency matrix of the graph
        
    Returns:
        Dict: Hamiltonian as Pauli strings
        
    Example:
        >>> A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        >>> H = maxcut_from_graph(A)
    """
    n = adjacency_matrix.shape[0]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i, j] != 0:
                edges.append((i, j))
    
    return maxcut_hamiltonian(edges)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'QAOResult',
    'QAOA',
    'maxcut_hamiltonian',
    'qubo_to_hamiltonian',
    'maxcut_from_graph',
]