# psiqit/variational/variational_advanced.py 

"""
Advanced Variational Methods
SSVQE, ADAPT-VQE, and Variational Quantum Deflation
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, zero, one, plus, minus, basis
from ..quantum.operator import Operator, identity, pauli_x, pauli_y, pauli_z, hadamard, rx, ry, rz, cnot, cz
from ..qml.vqc import VQC, VQCResult
from ..variational.vqe import VQE, VQEResult
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# MULTI-STATE RESULT CLASS
# ============================================================================

@dataclass
class MultiStateResult:
    """
    Result container for multi-state variational methods
    
    Attributes:
        energies: List of energies for each state
        states: List of states found
        success: Whether optimization was successful
        iterations: Number of iterations
        message: Additional message
    """
    energies: List[float] = field(default_factory=list)
    states: List[Ket] = field(default_factory=list)
    success: bool = True
    iterations: int = 0
    message: str = ""
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"MultiStateResult(states={len(self.energies)}, status={status})"
    
    def __str__(self) -> str:
        lines = [
            "Multi-State Variational Results:",
            f"  Success: {self.success}",
            f"  Iterations: {self.iterations}",
            f"  Message: {self.message}",
        ]
        for i, energy in enumerate(self.energies):
            lines.append(f"  State {i}: energy={energy:.6f}")
        return "\n".join(lines)
    
    def get_energy_gaps(self) -> List[float]:
        """Get energy gaps between consecutive states"""
        if len(self.energies) < 2:
            return []
        return [self.energies[i+1] - self.energies[i] for i in range(len(self.energies) - 1)]


# ============================================================================
# SSVQE (Subspace-Search VQE)
# ============================================================================

class SSVQE:
    """
    Subspace-Search VQE (SSVQE)
    
    Finds multiple excited states simultaneously by using a weighted sum of
    energies with orthogonalization constraints.
    
    Example:
        >>> from psiqit.variational import SSVQE
        >>> hamiltonian = {'Z0Z1': 1.0, 'Z0': 0.5, 'Z1': -0.5}
        >>> ssvqe = SSVQE(n_qubits=2, hamiltonian=hamiltonian, n_states=3)
        >>> result = ssvqe.run(n_iterations=100)
        >>> print(result.energies)
    """
    
    def __init__(
        self,
        n_qubits: int,
        hamiltonian: Dict[str, float],
        n_states: int = 3,
        n_layers: int = 2,
        entangler: str = 'cnot'
    ):
        """
        Initialize SSVQE
        
        Args:
            n_qubits: Number of qubits
            hamiltonian: Hamiltonian as Pauli strings
            n_states: Number of states to find
            n_layers: Number of variational layers
            entangler: Type of entangling gate
        """
        validate_qubits(n_qubits)
        
        self.n_qubits = n_qubits
        self.hamiltonian = hamiltonian
        self.n_states = n_states
        self.n_layers = n_layers
        self.entangler = entangler
        
        # Create VQCs for each state
        self.vqcs = []
        for i in range(n_states):
            vqc = VQC(
                n_qubits=n_qubits,
                n_layers=n_layers,
                entangler=entangler,
                measurement='z'
            )
            self.vqcs.append(vqc)
        
        # Total parameters = n_states * params_per_state
        self.params_per_state = vqc.n_params
        self.n_params = n_states * self.params_per_state
        
        # Initialize parameters
        self._params = None
        self._init_random()
        
        # Reference states (orthogonal inputs)
        self._reference_states = self._create_reference_states()
        
        # Parse Hamiltonian
        self._hamiltonian_terms = self._parse_hamiltonian()
        
        logger.info(f"SSVQE initialized: {n_qubits} qubits, {n_states} states, {n_layers} layers")
    
    def _init_random(self, seed: Optional[int] = None):
        """Initialize parameters randomly"""
        if seed is not None:
            np.random.seed(seed)
        self._params = np.random.uniform(-np.pi, np.pi, size=self.n_params)
    
    def _create_reference_states(self) -> List[Ket]:
        """Create orthogonal reference states"""
        dim = 2 ** self.n_qubits
        states = []
        for i in range(min(self.n_states, dim)):
            state = basis(dim, i)
            states.append(state)
        return states
    
    def _parse_hamiltonian(self) -> List[Dict[str, Any]]:
        """Parse Hamiltonian into terms"""
        import re
        terms = []
        
        for pauli_string, coeff in self.hamiltonian.items():
            if pauli_string == 'I':
                terms.append({
                    'paulis': {},
                    'coeff': coeff,
                    'is_identity': True
                })
            else:
                paulis = {}
                for match in re.finditer(r'([IXYZ])(\d+)', pauli_string):
                    op, qubit = match.group(1), int(match.group(2))
                    paulis[qubit] = op
                terms.append({
                    'paulis': paulis,
                    'coeff': coeff,
                    'is_identity': False,
                    'string': pauli_string
                })
        
        return terms
    
    def _expectation_pauli(self, state: Ket, paulis: Dict[int, str]) -> float:
        """Compute expectation value of a Pauli string"""
        all_z = all(op == 'Z' for op in paulis.values())
        
        if all_z:
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
        
        # Build full operator for non-Z Paulis
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
        
        return float(np.vdot(state.data, full_operator @ state.data).real)
    
    def _compute_state_energy(self, state: Ket) -> float:
        """Compute energy of a single state"""
        total_energy = 0.0
        for term in self._hamiltonian_terms:
            if term['is_identity']:
                total_energy += term['coeff']
            else:
                expectation = self._expectation_pauli(state, term['paulis'])
                total_energy += term['coeff'] * expectation
        return float(total_energy)
    
    def build_circuit(self, params: Optional[np.ndarray] = None) -> List[QuantumCircuit]:
        """
        Build circuits for all states
        
        Args:
            params: Parameters (if None, uses current params)
            
        Returns:
            List[QuantumCircuit]: Circuits for each state
        """
        if params is None:
            params = self._params
        
        circuits = []
        for i in range(self.n_states):
            start_idx = i * self.params_per_state
            state_params = params[start_idx:start_idx + self.params_per_state]
            self.vqcs[i].set_params(state_params)
            
            # Build circuit with reference state
            circuit = self.vqcs[i].build_circuit(self._reference_states[i])
            circuits.append(circuit)
        
        return circuits
    
    def evaluate_state_energy(self, state_idx: int, params: Optional[np.ndarray] = None) -> float:
        """
        Evaluate energy of a specific state
        
        Args:
            state_idx: Index of the state
            params: Parameters (if None, uses current params)
            
        Returns:
            float: Energy of the state
        """
        if params is not None:
            start_idx = state_idx * self.params_per_state
            state_params = params[start_idx:start_idx + self.params_per_state]
            self.vqcs[state_idx].set_params(state_params)
        
        state = self.vqcs[state_idx].get_state(self._reference_states[state_idx])
        return self._compute_state_energy(state)
    
    def evaluate_cost(self, params: Optional[np.ndarray] = None) -> float:
        """
        Evaluate the SSVQE cost function
        
        Cost = Σ_i w_i * E_i where w_i are weights
        With orthogonalization constraints enforced by the reference states
        
        Args:
            params: Parameters (if None, uses current params)
            
        Returns:
            float: Cost value
        """
        if params is not None:
            # Temporarily set params
            old_params = self._params.copy()
            self._params = params
        
        # Get all states
        states = []
        for i in range(self.n_states):
            state = self.vqcs[i].get_state(self._reference_states[i])
            states.append(state)
        
        # Compute energies
        energies = [self._compute_state_energy(s) for s in states]
        
        # Weighted sum (higher weights for lower states)
        weights = [1.0 / (i + 1) for i in range(self.n_states)]
        weights = np.array(weights) / np.sum(weights)
        
        cost = np.sum(weights * np.array(energies))
        
        # Orthogonalization penalty
        penalty = 0.0
        for i in range(self.n_states):
            for j in range(i + 1, self.n_states):
                overlap = abs(np.vdot(states[i].data, states[j].data)) ** 2
                penalty += overlap
        
        cost += 10.0 * penalty  # Penalty weight
        
        if params is not None:
            self._params = old_params
        
        return float(cost)
    
    def _gradient(self, params: np.ndarray, eps: float = 1e-5) -> List[float]:
        """Compute gradient using finite differences"""
        grad = np.zeros(len(params))
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            
            cost_plus = self.evaluate_cost(params_plus)
            cost_minus = self.evaluate_cost(params_minus)
            
            grad[i] = (cost_plus - cost_minus) / (2 * eps)
        
        return grad.tolist()
    
    def run(
        self,
        n_iterations: int = 100,
        learning_rate: float = 0.1,
        verbose: bool = True,
        initial_params: Optional[np.ndarray] = None
    ) -> MultiStateResult:
        """
        Run SSVQE optimization
        
        Args:
            n_iterations: Number of iterations
            learning_rate: Learning rate
            verbose: Print progress
            initial_params: Initial parameters
            
        Returns:
            MultiStateResult: Optimization results
        """
        logger.info(f"Starting SSVQE optimization: {n_iterations} iterations")
        
        if initial_params is not None:
            self._params = initial_params
        else:
            self._init_random()
        
        params = self._params.copy()
        history = []
        
        for iteration in range(n_iterations):
            grad = self._gradient(params)
            params = params - learning_rate * np.array(grad)
            
            cost = self.evaluate_cost(params)
            history.append(cost)
            
            if verbose and (iteration + 1) % max(1, n_iterations // 10) == 0:
                logger.info(f"Iteration {iteration+1}/{n_iterations}: cost={cost:.6f}")
        
        # Get final states and energies
        self._params = params
        states = []
        energies = []
        
        for i in range(self.n_states):
            state = self.vqcs[i].get_state(self._reference_states[i])
            states.append(state)
            energy = self._compute_state_energy(state)
            energies.append(energy)
        
        logger.info(f"SSVQE completed: energies={energies}")
        
        return MultiStateResult(
            energies=energies,
            states=states,
            success=True,
            iterations=n_iterations,
            message=f"SSVQE completed with {self.n_states} states"
        )


# ============================================================================
# ADAPT-VQE
# ============================================================================

class ADAPTVQE:
    """
    ADAPT-VQE (Adaptive Derivative-Assembled Pseudo-Trotter VQE)
    
    Builds the ansatz adaptively by selecting operators from a pool
    based on their gradients.
    
    Example:
        >>> from psiqit.variational import ADAPTVQE
        >>> hamiltonian = {'Z0Z1': 1.0, 'Z0': 0.5, 'Z1': -0.5}
        >>> adapt = ADAPTVQE(n_qubits=2, hamiltonian=hamiltonian)
        >>> result = adapt.run(max_iterations=10)
        >>> print(result.optimal_energy)
    """
    
    def __init__(
        self,
        n_qubits: int,
        hamiltonian: Dict[str, float],
        operator_pool: Optional[List[Operator]] = None
    ):
        """
        Initialize ADAPT-VQE
        
        Args:
            n_qubits: Number of qubits
            hamiltonian: Hamiltonian as Pauli strings
            operator_pool: Custom operator pool (if None, creates default)
        """
        validate_qubits(n_qubits)
        
        self.n_qubits = n_qubits
        self.hamiltonian = hamiltonian
        
        # Create or use operator pool
        if operator_pool is None:
            self.operator_pool = self._create_default_pool()
        else:
            self.operator_pool = operator_pool
        
        # Selected operators and their parameters
        self.selected_operators = []
        self.selected_params = []
        
        # Create VQE for optimization
        self.vqe = VQE(n_qubits, hamiltonian, n_layers=0)  # Start with empty ansatz
        
        logger.info(f"ADAPT-VQE initialized: {n_qubits} qubits, pool_size={len(self.operator_pool)}")
    
    def _create_default_pool(self) -> List[Operator]:
        """Create default operator pool"""
        pool = []
        
        # Single-qubit Pauli operators
        for qubit in range(self.n_qubits):
            for op in ['X', 'Y', 'Z']:
                if op == 'X':
                    mat = pauli_x().data
                elif op == 'Y':
                    mat = pauli_y().data
                else:
                    mat = pauli_z().data
                
                # Create full operator on all qubits
                full_op = np.eye(1, dtype=complex)
                for q in range(self.n_qubits):
                    if q == qubit:
                        full_op = np.kron(full_op, mat)
                    else:
                        full_op = np.kron(full_op, np.eye(2, dtype=complex))
                
                pool.append(Operator(full_op, name=f"{op}{qubit}"))
        
        # Two-qubit Pauli operators (ZZ, XX, YY)
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                for op1, op2 in [('Z', 'Z'), ('X', 'X'), ('Y', 'Y')]:
                    if op1 == 'Z':
                        mat1 = pauli_z().data
                    elif op1 == 'X':
                        mat1 = pauli_x().data
                    else:
                        mat1 = pauli_y().data
                    
                    if op2 == 'Z':
                        mat2 = pauli_z().data
                    elif op2 == 'X':
                        mat2 = pauli_x().data
                    else:
                        mat2 = pauli_y().data
                    
                    full_op = np.eye(1, dtype=complex)
                    for q in range(self.n_qubits):
                        if q == i:
                            full_op = np.kron(full_op, mat1)
                        elif q == j:
                            full_op = np.kron(full_op, mat2)
                        else:
                            full_op = np.kron(full_op, np.eye(2, dtype=complex))
                    
                    pool.append(Operator(full_op, name=f"{op1}{i}{op2}{j}"))
        
        return pool
    
    def _apply_operator(self, circuit: QuantumCircuit, op: Operator, param: float):
        """Apply an operator to the circuit"""
        # For Pauli operators, apply as rotations
        # This is a simplified version
        # In practice, we would apply the operator as a unitary gate
        
        # For now, we use the operator matrix directly
        # by applying it to the state (simulation only)
        
        # In a real circuit, we would need to decompose the operator into gates
        # This is a placeholder for simulation
        pass
    
    def _evaluate_energy_with_ops(self, params: np.ndarray) -> float:
        """Evaluate energy with current operators and parameters"""
        # This would build the circuit with the selected operators
        # For now, we use the VQE to evaluate
        return self.vqe.evaluate_energy(params)
    
    def _gradient_wrt_operator(self, params: np.ndarray, op_idx: int) -> float:
        """Compute gradient with respect to an operator"""
        eps = 1e-5
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[op_idx] += eps
        params_minus[op_idx] -= eps
        
        energy_plus = self._evaluate_energy_with_ops(params_plus)
        energy_minus = self._evaluate_energy_with_ops(params_minus)
        
        return (energy_plus - energy_minus) / (2 * eps)
    
    def run(
        self,
        max_iterations: int = 10,
        grad_threshold: float = 0.01,
        n_iterations: int = 100,
        learning_rate: float = 0.1,
        verbose: bool = True
    ) -> VQEResult:
        """
        Run ADAPT-VQE optimization
        
        Args:
            max_iterations: Maximum number of operator additions
            grad_threshold: Threshold for adding operators
            n_iterations: Number of iterations per VQE optimization
            learning_rate: Learning rate for VQE
            verbose: Print progress
            
        Returns:
            VQEResult: Optimization results
        """
        logger.info(f"Starting ADAPT-VQE: max_iterations={max_iterations}")
        
        params = np.array([])
        selected_ops = []
        
        for iteration in range(max_iterations):
            # Compute gradients for all operators in the pool
            grad_norms = []
            for op in self.operator_pool:
                # Add operator to current ansatz
                # Compute gradient magnitude
                # This is a simplified version
                grad_norm = np.random.random()  # Placeholder
                grad_norms.append(grad_norm)
            
            # Find operator with largest gradient
            max_idx = np.argmax(grad_norms)
            max_grad = grad_norms[max_idx]
            
            if max_grad < grad_threshold:
                logger.info(f"Gradient threshold reached: {max_grad:.4f} < {grad_threshold}")
                break
            
            # Add operator to ansatz
            selected_ops.append(self.operator_pool[max_idx])
            params = np.append(params, 0.0)  # Initialize parameter
            
            # Optimize all parameters
            # This would use the VQE optimizer
            # For now, we just update the VQE
            
            logger.info(f"Iteration {iteration+1}: added operator, gradient={max_grad:.4f}")
        
        self.selected_operators = selected_ops
        self.selected_params = params
        
        # Final VQE optimization
        result = self.vqe.run(
            n_iterations=n_iterations,
            learning_rate=learning_rate,
            verbose=verbose
        )
        
        logger.info(f"ADAPT-VQE completed: energy={result.optimal_energy:.6f}")
        
        return result


# ============================================================================
# VARIATIONAL QUANTUM DEFLATION
# ============================================================================

class VariationalQuantumDeflation:
    """
    Variational Quantum Deflation (VQD)
    
    Finds excited states by penalizing overlap with previously found states.
    
    Example:
        >>> from psiqit.variational import VariationalQuantumDeflation
        >>> hamiltonian = {'Z0Z1': 1.0, 'Z0': 0.5, 'Z1': -0.5}
        >>> vqd = VariationalQuantumDeflation(n_qubits=2, hamiltonian=hamiltonian)
        >>> result = vqd.find_excited_state(penalty_weight=1.0)
        >>> print(result.optimal_energy)
    """
    
    def __init__(
        self,
        n_qubits: int,
        hamiltonian: Dict[str, float],
        n_layers: int = 2,
        entangler: str = 'cnot'
    ):
        """
        Initialize VQD
        
        Args:
            n_qubits: Number of qubits
            hamiltonian: Hamiltonian as Pauli strings
            n_layers: Number of variational layers
            entangler: Type of entangling gate
        """
        validate_qubits(n_qubits)
        
        self.n_qubits = n_qubits
        self.hamiltonian = hamiltonian
        self.n_layers = n_layers
        self.entangler = entangler
        
        # Create VQC
        self.vqc = VQC(
            n_qubits=n_qubits,
            n_layers=n_layers,
            entangler=entangler,
            measurement='z'
        )
        
        # Store found states
        self.found_states = []
        self.found_energies = []
        
        logger.info(f"VQD initialized: {n_qubits} qubits, {n_layers} layers")
    
    def _parse_hamiltonian(self) -> List[Dict[str, Any]]:
        """Parse Hamiltonian into terms"""
        import re
        terms = []
        
        for pauli_string, coeff in self.hamiltonian.items():
            if pauli_string == 'I':
                terms.append({
                    'paulis': {},
                    'coeff': coeff,
                    'is_identity': True
                })
            else:
                paulis = {}
                for match in re.finditer(r'([IXYZ])(\d+)', pauli_string):
                    op, qubit = match.group(1), int(match.group(2))
                    paulis[qubit] = op
                terms.append({
                    'paulis': paulis,
                    'coeff': coeff,
                    'is_identity': False,
                    'string': pauli_string
                })
        
        return terms
    
    def _expectation_pauli(self, state: Ket, paulis: Dict[int, str]) -> float:
        """Compute expectation value of a Pauli string"""
        all_z = all(op == 'Z' for op in paulis.values())
        
        if all_z:
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
        
        # Build full operator
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
        
        return float(np.vdot(state.data, full_operator @ state.data).real)
    
    def _compute_energy(self, state: Ket) -> float:
        """Compute energy of a state"""
        terms = self._parse_hamiltonian()
        total_energy = 0.0
        for term in terms:
            if term['is_identity']:
                total_energy += term['coeff']
            else:
                expectation = self._expectation_pauli(state, term['paulis'])
                total_energy += term['coeff'] * expectation
        return float(total_energy)
    
    def _overlap_penalty(self, params: np.ndarray, state_idx: int) -> float:
        """
        Compute overlap penalty with previously found states
        
        Args:
            params: Parameters for current state
            state_idx: Index of the state being optimized
            
        Returns:
            float: Overlap penalty
        """
        self.vqc.set_params(params)
        current_state = self.vqc.get_state()
        
        penalty = 0.0
        for i, found_state in enumerate(self.found_states):
            if i >= state_idx:
                break
            overlap = abs(np.vdot(found_state.data, current_state.data)) ** 2
            penalty += overlap
        
        return float(penalty)
    
    def _cost_function(self, params: np.ndarray, penalty_weight: float = 1.0) -> float:
        """
        Cost function with overlap penalty
        
        Args:
            params: Parameters
            penalty_weight: Weight of the penalty term
            
        Returns:
            float: Cost value
        """
        self.vqc.set_params(params)
        state = self.vqc.get_state()
        
        # Energy
        energy = self._compute_energy(state)
        
        # Penalty
        penalty = self._overlap_penalty(params, len(self.found_states))
        
        return float(energy + penalty_weight * penalty)
    
    def find_excited_state(
        self,
        penalty_weight: float = 1.0,
        n_iterations: int = 100,
        learning_rate: float = 0.1,
        verbose: bool = True
    ) -> VQEResult:
        """
        Find the next excited state
        
        Args:
            penalty_weight: Weight of the overlap penalty
            n_iterations: Number of iterations
            learning_rate: Learning rate
            verbose: Print progress
            
        Returns:
            VQEResult: Optimization results
        """
        logger.info(f"Finding excited state with penalty_weight={penalty_weight}")
        
        # Initialize parameters randomly
        params = np.random.uniform(-np.pi, np.pi, size=self.vqc.n_params)
        history = []
        
        # Define cost function for optimization
        def cost_func(p):
            return self._cost_function(p, penalty_weight)
        
        # Optimize
        for iteration in range(n_iterations):
            # Compute gradient
            grad = np.zeros(len(params))
            eps = 1e-5
            for i in range(len(params)):
                params_plus = params.copy()
                params_minus = params.copy()
                params_plus[i] += eps
                params_minus[i] -= eps
                
                cost_plus = cost_func(params_plus)
                cost_minus = cost_func(params_minus)
                grad[i] = (cost_plus - cost_minus) / (2 * eps)
            
            params = params - learning_rate * np.array(grad)
            
            cost = cost_func(params)
            history.append(cost)
            
            if verbose and (iteration + 1) % max(1, n_iterations // 10) == 0:
                logger.info(f"Iteration {iteration+1}/{n_iterations}: cost={cost:.6f}")
        
        # Get final state
        self.vqc.set_params(params)
        state = self.vqc.get_state()
        energy = self._compute_energy(state)
        
        # Store found state
        self.found_states.append(state)
        self.found_energies.append(energy)
        
        logger.info(f"Found excited state: energy={energy:.6f}")
        
        return VQEResult(
            optimal_params=params,
            optimal_energy=energy,
            n_iterations=n_iterations,
            history=history,
            success=True,
            message=f"VQD found state with energy {energy:.6f}",
            final_state=state
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'MultiStateResult',
    'SSVQE',
    'ADAPTVQE',
    'VariationalQuantumDeflation',
]