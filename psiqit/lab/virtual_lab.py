# psiqit/lab/virtual_lab.py 

"""
Virtual Quantum Lab Module
Quantum experiment simulation and management
"""

import numpy as np
import time
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from ..circuits.circuit import QuantumCircuit
from ..circuits.qubit import Qubit
from ..circuits.register import QuantumRegister
from ..quantum.state import Ket, zero, one, plus, minus, bell_phi_plus, ghz
from ..quantum.operator import pauli_x, pauli_y, pauli_z, hadamard, cnot, toffoli
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# EXPERIMENT STATUS ENUM
# ============================================================================

class ExperimentStatus(Enum):
    """Status of an experiment"""
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"ExperimentStatus.{self.name}"


# ============================================================================
# EXPERIMENT RESULT CLASS
# ============================================================================

@dataclass
class ExperimentResult:
    """
    Result of a quantum experiment
    
    Attributes:
        name: Name of the experiment
        timestamp: Time of execution
        n_qubits: Number of qubits
        shots: Number of measurement shots
        counts: Measurement counts
        probabilities: Probability distribution
        final_state: Final state vector
        success_rate: Success rate of the experiment
        execution_time: Execution time in seconds
        status: Status of the experiment
        gates: Number of gates applied
        depth: Circuit depth
    """
    name: str
    timestamp: str
    n_qubits: int
    shots: int
    counts: Dict[str, int]
    probabilities: List[float]
    final_state: Optional[Ket] = None
    success_rate: float = 0.0
    execution_time: float = 0.0
    status: ExperimentStatus = ExperimentStatus.COMPLETED
    gates: int = 0
    depth: int = 0
    
    def summary(self) -> str:
        """
        Generate a summary of the experiment results
        
        Returns:
            str: Summary string
            
        Example:
            >>> result = experiment.run()
            >>> print(result.summary())
        """
        lines = [
            "=" * 50,
            f"Experiment: {self.name}",
            f"Status: {self.status}",
            f"Timestamp: {self.timestamp}",
            f"Qubits: {self.n_qubits}",
            f"Shots: {self.shots}",
            f"Gates: {self.gates}",
            f"Depth: {self.depth}",
            f"Execution Time: {self.execution_time:.4f}s",
            f"Success Rate: {self.success_rate:.2%}",
            "-" * 50,
            "Measurement Results:",
        ]
        
        # Sort counts by state
        sorted_counts = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)
        for state, count in sorted_counts[:10]:
            prob = count / self.shots
            lines.append(f"  {state}: {count} ({prob:.2%})")
        
        if len(sorted_counts) > 10:
            lines.append(f"  ... and {len(sorted_counts) - 10} more states")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def get_most_likely(self) -> Tuple[str, int, float]:
        """
        Get the most likely measurement outcome
        
        Returns:
            Tuple[str, int, float]: (state, count, probability)
        """
        if not self.counts:
            return "", 0, 0.0
        
        most_likely = max(self.counts.items(), key=lambda x: x[1])
        state, count = most_likely
        prob = count / self.shots
        return state, count, prob
    
    def get_probability(self, state: str) -> float:
        """
        Get the probability of a specific state
        
        Args:
            state: State string (e.g., '00', '101')
            
        Returns:
            float: Probability (0-1)
        """
        return self.counts.get(state, 0) / self.shots
    
    def __repr__(self) -> str:
        return f"ExperimentResult(name='{self.name}', status={self.status}, shots={self.shots})"


# ============================================================================
# EXPERIMENT CLASS
# ============================================================================

class Experiment:
    """
    Quantum experiment configuration
    
    An experiment defines a quantum circuit to be run in the virtual lab.
    
    Example:
        >>> exp = Experiment("Bell State", n_qubits=2)
        >>> exp.add_gate('h', 0)
        >>> exp.add_gate('cx', 0, 1)
        >>> lab = QuantumLab()
        >>> result = lab.run_experiment(exp)
        >>> print(result.summary())
    """
    
    def __init__(
        self,
        name: str,
        n_qubits: int,
        gates: Optional[List[Dict[str, Any]]] = None,
        shots: int = 1024,
        description: str = ""
    ):
        """
        Initialize an experiment
        
        Args:
            name: Name of the experiment
            n_qubits: Number of qubits
            gates: Initial list of gates (optional)
            shots: Number of measurement shots
            description: Description of the experiment
            
        Example:
            >>> exp = Experiment("My Experiment", n_qubits=2, shots=1000)
        """
        validate_qubits(n_qubits)
        
        self.name = name
        self.n_qubits = n_qubits
        self.shots = shots
        self.description = description
        self.gates: List[Dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()
        self.status = ExperimentStatus.DRAFT
        
        if gates:
            for gate in gates:
                self.add_gate(**gate)
        
        logger.info(f"Experiment '{name}' created with {n_qubits} qubits")
    
    def add_gate(
        self,
        gate_name: str,
        *qubits: int,
        params: Optional[List[float]] = None
    ) -> 'Experiment':
        """
        Add a gate to the experiment
        
        Args:
            gate_name: Name of the gate ('h', 'x', 'y', 'z', 'cx', 'cz', 'swap', 'toffoli', ...)
            *qubits: Qubit indices
            params: Parameters for parametric gates (e.g., rotation angles)
            
        Returns:
            Experiment: Self for chaining
            
        Example:
            >>> exp.add_gate('h', 0)
            >>> exp.add_gate('cx', 0, 1)
            >>> exp.add_gate('rx', 0, params=[np.pi/2])
        """
        # Validate qubits
        for q in qubits:
            if q < 0 or q >= self.n_qubits:
                raise ValueError(f"Qubit {q} out of range [0, {self.n_qubits-1}]")
        
        gate = {
            'name': gate_name,
            'qubits': list(qubits),
            'params': params or []
        }
        
        self.gates.append(gate)
        self.status = ExperimentStatus.DRAFT
        
        logger.debug(f"Added gate {gate_name} on qubits {qubits}")
        return self
    
    def clear_gates(self) -> 'Experiment':
        """
        Clear all gates from the experiment
        
        Returns:
            Experiment: Self for chaining
            
        Example:
            >>> exp.clear_gates()
        """
        self.gates = []
        self.status = ExperimentStatus.DRAFT
        logger.debug(f"Cleared all gates from experiment '{self.name}'")
        return self
    
    def build_circuit(self) -> QuantumCircuit:
        """
        Build the quantum circuit from the experiment gates
        
        Returns:
            QuantumCircuit: Quantum circuit
            
        Example:
            >>> circ = exp.build_circuit()
            >>> print(circ.draw())
        """
        circuit = QuantumCircuit(self.n_qubits)
        
        # Gate mapping
        gate_map = {
            'h': circuit.h,
            'x': circuit.x,
            'y': circuit.y,
            'z': circuit.z,
            's': circuit.s,
            't': circuit.t,
            'rx': circuit.rx,
            'ry': circuit.ry,
            'rz': circuit.rz,
            'cx': circuit.cx,
            'cz': circuit.cz,
            'swap': circuit.swap,
            'toffoli': circuit.toffoli,
        }
        
        for gate in self.gates:
            name = gate['name']
            qubits = gate['qubits']
            params = gate['params']
            
            if name in gate_map:
                if params:
                    # Parametric gate
                    gate_map[name](*qubits, *params)
                else:
                    # Non-parametric gate
                    gate_map[name](*qubits)
            else:
                logger.warning(f"Unknown gate: {name}, skipping")
        
        return circuit
    
    def run(self, shots: Optional[int] = None) -> ExperimentResult:
        """
        Run the experiment
        
        Args:
            shots: Number of shots (overrides default)
            
        Returns:
            ExperimentResult: Experiment results
            
        Example:
            >>> result = exp.run()
            >>> print(result.summary())
        """
        lab = QuantumLab()
        return lab.run_experiment(self, shots=shots)
    
    def get_state(self) -> Ket:
        """
        Get the state vector of the experiment
        
        Returns:
            Ket: State vector
            
        Example:
            >>> state = exp.get_state()
            >>> print(state)
        """
        lab = QuantumLab()
        return lab.get_state_vector(self)
    
    def get_bloch(self, qubit: int = 0) -> Tuple[float, float, float]:
        """
        Get Bloch sphere coordinates for a qubit
        
        Args:
            qubit: Qubit index
            
        Returns:
            Tuple[float, float, float]: (x, y, z) coordinates
            
        Example:
            >>> x, y, z = exp.get_bloch(0)
            >>> print(f"Bloch: ({x:.3f}, {y:.3f}, {z:.3f})")
        """
        lab = QuantumLab()
        return lab.get_bloch_coordinates(self, qubit)
    
    def __repr__(self) -> str:
        return f"Experiment(name='{self.name}', n_qubits={self.n_qubits}, gates={len(self.gates)}, status={self.status})"
    
    def __str__(self) -> str:
        lines = [
            f"Experiment: {self.name}",
            f"  Qubits: {self.n_qubits}",
            f"  Gates: {len(self.gates)}",
            f"  Shots: {self.shots}",
            f"  Status: {self.status}",
            f"  Created: {self.created_at}",
        ]
        if self.description:
            lines.append(f"  Description: {self.description}")
        
        if self.gates:
            lines.append("  Gates:")
            for i, gate in enumerate(self.gates[:10]):
                lines.append(f"    {i+1}: {gate['name']} on {gate['qubits']}")
            if len(self.gates) > 10:
                lines.append(f"    ... and {len(self.gates) - 10} more gates")
        
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self.gates)


# ============================================================================
# QUANTUM LAB CLASS
# ============================================================================

class QuantumLab:
    """
    Virtual Quantum Laboratory
    
    A virtual lab for creating and running quantum experiments.
    
    Example:
        >>> lab = QuantumLab()
        >>> exp = lab.create_experiment("Bell State", n_qubits=2)
        >>> exp.add_gate('h', 0).add_gate('cx', 0, 1)
        >>> result = lab.run_experiment(exp)
        >>> print(result.summary())
        >>> experiments = lab.list_experiments()
    """
    
    def __init__(self):
        """Initialize the quantum lab"""
        self.experiments: List[Experiment] = []
        self.results: List[ExperimentResult] = []
        self._experiment_counter = 0
        
        logger.info("QuantumLab initialized")
    
    def create_experiment(
        self,
        name: str,
        n_qubits: int,
        shots: int = 1024,
        description: str = ""
    ) -> Experiment:
        """
        Create a new experiment
        
        Args:
            name: Name of the experiment
            n_qubits: Number of qubits
            shots: Number of measurement shots
            description: Description of the experiment
            
        Returns:
            Experiment: The created experiment
            
        Example:
            >>> exp = lab.create_experiment("My Experiment", 2, shots=1000)
        """
        exp = Experiment(name, n_qubits, shots=shots, description=description)
        self.experiments.append(exp)
        self._experiment_counter += 1
        return exp
    
    def run_experiment(
        self,
        experiment: Experiment,
        shots: Optional[int] = None
    ) -> ExperimentResult:
        """
        Run an experiment
        
        Args:
            experiment: Experiment to run
            shots: Number of shots (overrides experiment default)
            
        Returns:
            ExperimentResult: Results of the experiment
            
        Example:
            >>> result = lab.run_experiment(exp)
        """
        start_time = time.time()
        
        # Update status
        experiment.status = ExperimentStatus.RUNNING
        
        # Build circuit
        circuit = experiment.build_circuit()
        
        # Get number of shots
        num_shots = shots if shots is not None else experiment.shots
        
        try:
            # Run the circuit
            result = circuit.measure(shots=num_shots)
            
            # Get state vector
            state = circuit.run()
            
            # Calculate probabilities
            probs = np.abs(state.data) ** 2
            probs = probs / np.sum(probs)
            
            # Calculate success rate (for simplicity, use fidelity with ideal state)
            # This is a placeholder - in a real lab, this would be more complex
            success_rate = 1.0
            
            # Create result
            exec_time = time.time() - start_time
            
            exp_result = ExperimentResult(
                name=experiment.name,
                timestamp=datetime.now().isoformat(),
                n_qubits=experiment.n_qubits,
                shots=num_shots,
                counts=result['counts'],
                probabilities=probs.tolist(),
                final_state=state,
                success_rate=success_rate,
                execution_time=exec_time,
                status=ExperimentStatus.COMPLETED,
                gates=len(experiment.gates),
                depth=circuit.depth
            )
            
            # Update experiment status
            experiment.status = ExperimentStatus.COMPLETED
            
            # Store result
            self.results.append(exp_result)
            
            logger.info(f"Experiment '{experiment.name}' completed in {exec_time:.4f}s")
            
            return exp_result
            
        except Exception as e:
            # Update status
            experiment.status = ExperimentStatus.FAILED
            logger.error(f"Experiment '{experiment.name}' failed: {e}")
            
            # Create failed result
            exp_result = ExperimentResult(
                name=experiment.name,
                timestamp=datetime.now().isoformat(),
                n_qubits=experiment.n_qubits,
                shots=num_shots,
                counts={},
                probabilities=[],
                final_state=None,
                success_rate=0.0,
                execution_time=time.time() - start_time,
                status=ExperimentStatus.FAILED,
                gates=len(experiment.gates),
                depth=0
            )
            
            self.results.append(exp_result)
            return exp_result
    
    def get_state_vector(self, experiment: Experiment) -> Ket:
        """
        Get the state vector of an experiment
        
        Args:
            experiment: Experiment to get state from
            
        Returns:
            Ket: State vector
            
        Example:
            >>> state = lab.get_state_vector(exp)
            >>> print(state)
        """
        circuit = experiment.build_circuit()
        return circuit.run()
    
    def get_bloch_coordinates(
        self,
        experiment: Experiment,
        qubit: int = 0
    ) -> Tuple[float, float, float]:
        """
        Get Bloch sphere coordinates for a qubit in an experiment
        
        Args:
            experiment: Experiment to analyze
            qubit: Qubit index
            
        Returns:
            Tuple[float, float, float]: (x, y, z) coordinates
            
        Example:
            >>> x, y, z = lab.get_bloch_coordinates(exp, 0)
        """
        state = self.get_state_vector(experiment)
        
        # For a single qubit or reduced state
        if state.dim == 2:
            # Direct calculation for single qubit
            a, b = state.data[0], state.data[1]
            x = 2 * np.real(b * np.conj(a))
            y = 2 * np.imag(b * np.conj(a))
            z = np.abs(a)**2 - np.abs(b)**2
            return (float(x), float(y), float(z))
        
        # For multi-qubit states, we need to trace out other qubits
        # This is a simplified version - just return the reduced state
        # For a proper implementation, we would compute the reduced density matrix
        
        # Compute reduced density matrix for the given qubit
        n_qubits = int(np.log2(state.dim))
        if qubit >= n_qubits:
            raise ValueError(f"Qubit {qubit} out of range [0, {n_qubits-1}]")
        
        # Build reduced density matrix by tracing out other qubits
        # This is a simplified version
        dim = state.dim
        rho_reduced = np.zeros((2, 2), dtype=complex)
        
        # For each basis state
        for i in range(dim):
            for j in range(dim):
                # Check if qubit states match
                if (i >> qubit) & 1 == (j >> qubit) & 1:
                    # Trace out other qubits
                    # This is a simplified approximation
                    rho_reduced[(i >> qubit) & 1, (j >> qubit) & 1] += state.data[i] * np.conj(state.data[j])
        
        # Normalize
        trace = np.trace(rho_reduced).real
        if trace > 0:
            rho_reduced = rho_reduced / trace
        
        # Compute Bloch coordinates from reduced density matrix
        # ρ = (I + xσ_x + yσ_y + zσ_z) / 2
        x = 2 * rho_reduced[0, 1].real
        y = 2 * rho_reduced[0, 1].imag
        z = rho_reduced[0, 0].real - rho_reduced[1, 1].real
        
        return (float(x), float(y), float(z))
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments in the lab
        
        Returns:
            List[Dict]: List of experiment information
            
        Example:
            >>> experiments = lab.list_experiments()
            >>> for exp in experiments:
            ...     print(exp['name'], exp['status'])
        """
        experiments = []
        for exp in self.experiments:
            experiments.append({
                'name': exp.name,
                'n_qubits': exp.n_qubits,
                'gates': len(exp.gates),
                'shots': exp.shots,
                'status': str(exp.status),
                'created_at': exp.created_at
            })
        return experiments
    
    def get_results(self) -> List[ExperimentResult]:
        """
        Get all experiment results
        
        Returns:
            List[ExperimentResult]: List of results
            
        Example:
            >>> results = lab.get_results()
            >>> for result in results:
            ...     print(result.name, result.status)
        """
        return self.results
    
    def get_last_result(self) -> Optional[ExperimentResult]:
        """
        Get the last experiment result
        
        Returns:
            Optional[ExperimentResult]: Last result or None
            
        Example:
            >>> result = lab.get_last_result()
            >>> if result:
            ...     print(result.summary())
        """
        if self.results:
            return self.results[-1]
        return None
    
    def clear(self) -> 'QuantumLab':
        """
        Clear all experiments and results
        
        Returns:
            QuantumLab: Self for chaining
            
        Example:
            >>> lab.clear()
        """
        self.experiments = []
        self.results = []
        self._experiment_counter = 0
        logger.info("QuantumLab cleared")
        return self
    
    def __repr__(self) -> str:
        return f"QuantumLab(experiments={len(self.experiments)}, results={len(self.results)})"
    
    def __str__(self) -> str:
        lines = [
            "Quantum Virtual Laboratory",
            f"  Experiments: {len(self.experiments)}",
            f"  Results: {len(self.results)}",
        ]
        
        if self.experiments:
            lines.append("  Recent Experiments:")
            for exp in self.experiments[-5:]:
                lines.append(f"    - {exp.name}: {exp.status}")
        
        return "\n".join(lines)


# ============================================================================
# PREDEFINED EXPERIMENTS
# ============================================================================

class PredefinedExperiments:
    """
    Predefined quantum experiments
    
    Example:
        >>> exp = PredefinedExperiments.bell_state()
        >>> lab = QuantumLab()
        >>> result = lab.run_experiment(exp)
        >>> print(result.summary())
    """
    
    @staticmethod
    def bell_state() -> Experiment:
        """
        Create a Bell state experiment
        
        Creates the Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        
        Returns:
            Experiment: Bell state experiment
            
        Example:
            >>> exp = PredefinedExperiments.bell_state()
            >>> lab = QuantumLab()
            >>> result = lab.run_experiment(exp)
            >>> print(result.summary())
        """
        exp = Experiment("Bell State", n_qubits=2, shots=1024)
        exp.add_gate('h', 0)
        exp.add_gate('cx', 0, 1)
        exp.description = "Creates the Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2"
        return exp
    
    @staticmethod
    def ghz_state(n: int = 3) -> Experiment:
        """
        Create a GHZ state experiment
        
        Creates the GHZ state (|00...0⟩ + |11...1⟩)/√2
        
        Args:
            n: Number of qubits (default: 3)
            
        Returns:
            Experiment: GHZ state experiment
            
        Example:
            >>> exp = PredefinedExperiments.ghz_state(3)
            >>> lab = QuantumLab()
            >>> result = lab.run_experiment(exp)
            >>> print(result.summary())
        """
        validate_qubits(n)
        exp = Experiment(f"GHZ State ({n} qubits)", n_qubits=n, shots=1024)
        exp.add_gate('h', 0)
        for i in range(1, n):
            exp.add_gate('cx', 0, i)
        exp.description = f"Creates the GHZ state (|0⟩⊗{n} + |1⟩⊗{n})/√2"
        return exp
    
    @staticmethod
    def w_state(n: int = 3) -> Experiment:
        """
        Create a W state experiment
        
        Creates the W state (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
        
        Args:
            n: Number of qubits (default: 3)
            
        Returns:
            Experiment: W state experiment
            
        Example:
            >>> exp = PredefinedExperiments.w_state(3)
            >>> lab = QuantumLab()
            >>> result = lab.run_experiment(exp)
            >>> print(result.summary())
        """
        validate_qubits(n)
        exp = Experiment(f"W State ({n} qubits)", n_qubits=n, shots=1024)
        
        # W state circuit for n qubits
        # This is a simplified circuit for W state generation
        exp.add_gate('ry', 0, params=[2 * np.arccos(1/np.sqrt(n))])
        for i in range(1, n):
            exp.add_gate('cx', i-1, i)
            exp.add_gate('ry', i, params=[2 * np.arccos(1/np.sqrt(n - i))])
            exp.add_gate('cx', i-1, i)
        
        exp.description = f"Creates the W state for {n} qubits"
        return exp
    
    @staticmethod
    def superposition(n: int = 1) -> Experiment:
        """
        Create a superposition state experiment
        
        Creates the state |+⟩⊗n
        
        Args:
            n: Number of qubits (default: 1)
            
        Returns:
            Experiment: Superposition experiment
            
        Example:
            >>> exp = PredefinedExperiments.superposition(2)
            >>> lab = QuantumLab()
            >>> result = lab.run_experiment(exp)
        """
        validate_qubits(n)
        exp = Experiment(f"Superposition ({n} qubits)", n_qubits=n, shots=1024)
        for i in range(n):
            exp.add_gate('h', i)
        exp.description = f"Creates the superposition state |+⟩⊗{n}"
        return exp
    
    @staticmethod
    def quantum_teleportation() -> Experiment:
        """
        Create a quantum teleportation experiment
        
        Returns:
            Experiment: Quantum teleportation experiment
            
        Example:
            >>> exp = PredefinedExperiments.quantum_teleportation()
            >>> lab = QuantumLab()
            >>> result = lab.run_experiment(exp)
        """
        exp = Experiment("Quantum Teleportation", n_qubits=3, shots=1024)
        
        # Prepare state to teleport (|+⟩ on qubit 0)
        exp.add_gate('h', 0)
        
        # Create Bell state on qubits 1 and 2
        exp.add_gate('h', 1)
        exp.add_gate('cx', 1, 2)
        
        # Bell measurement on qubits 0 and 1
        exp.add_gate('cx', 0, 1)
        exp.add_gate('h', 0)
        
        # Measure qubits 0 and 1 (in simulation, we just apply corrections)
        # Apply corrections based on measurement (simplified)
        
        exp.description = "Quantum teleportation of a state from qubit 0 to qubit 2"
        return exp


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ExperimentStatus',
    'ExperimentResult',
    'Experiment',
    'QuantumLab',
    'PredefinedExperiments',
]