# psiqit/dynamics/lindblad.py


import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from ..quantum.operator import Operator, identity, dagger, trace
from ..quantum.state import Ket
from ..math.qalgebra import commutator, anticommutator
from ..utils.logger import logger
from ..utils.validation import is_hermitian


# ============================================================================
# LINDBLAD RESULT CLASS
# ============================================================================

@dataclass
class LindbladResult:
    """
    Result container for Lindblad master equation evolution
    
    Attributes:
        times: Time points
        states: List of density matrices at each time
        populations: List of populations for each state
        success: Whether evolution was successful
        message: Additional information
        n_steps: Number of time steps
        final_state: Final density matrix
        fidelity: Fidelity with steady state (if available)
    """
    times: np.ndarray
    states: List[np.ndarray]
    populations: List[np.ndarray]
    success: bool = True
    message: str = ""
    n_steps: int = 0
    final_state: Optional[np.ndarray] = None
    fidelity: Optional[float] = None
    
    def __post_init__(self):
        """Post-process the result"""
        self.n_steps = len(self.times)
        
        if self.states:
            self.final_state = self.states[-1]
    
    def get_state_at_time(self, index: int) -> Optional[np.ndarray]:
        """Get density matrix at a specific time index"""
        if 0 <= index < len(self.states):
            return self.states[index]
        return None
    
    def get_final_state(self) -> Optional[np.ndarray]:
        """Get the final density matrix"""
        return self.final_state
    
    def get_population_at_time(self, index: int) -> Optional[np.ndarray]:
        """Get populations at a specific time index"""
        if 0 <= index < len(self.populations):
            return self.populations[index]
        return None
    
    def get_population_evolution(self, state_index: int) -> List[float]:
        """
        Get the evolution of a specific population
        
        Args:
            state_index: Index of the state
            
        Returns:
            List[float]: Population values over time
        """
        if state_index < 0 or state_index >= len(self.populations[0]):
            raise ValueError(f"State index {state_index} out of range")
        
        return [p[state_index] for p in self.populations]
    
    def get_fidelity_with_initial(self) -> List[float]:
        """
        Get fidelity of each state with the initial state
        
        Returns:
            List[float]: Fidelity values
        """
        if not self.states or self.states[0] is None:
            return []
        
        rho0 = self.states[0]
        fidelities = []
        
        for rho in self.states:
            # Fidelity = Tr(√(√ρ σ √ρ))²
            # For density matrices, we compute the square root fidelity
            sqrt_rho = np.linalg.sqrtm(rho)
            sqrt_rho_sigma_sqrt_rho = sqrt_rho @ rho0 @ sqrt_rho
            fid = np.trace(np.linalg.sqrtm(sqrt_rho_sigma_sqrt_rho)).real ** 2
            fidelities.append(float(fid))
        
        return fidelities
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"LindbladResult(steps={self.n_steps}, status={status})"
    
    def __str__(self) -> str:
        lines = [
            "Lindblad Evolution Results:",
            f"  Steps: {self.n_steps}",
            f"  Success: {self.success}",
            f"  Message: {self.message}",
        ]
        if self.final_state is not None:
            lines.append(f"  Final State Trace: {np.trace(self.final_state).real:.6f}")
        if self.fidelity is not None:
            lines.append(f"  Fidelity with Steady State: {self.fidelity:.6f}")
        return "\n".join(lines)


# ============================================================================
# LINDBLAD SOLVER
# ============================================================================

class LindbladSolver:
    """
    Lindblad master equation solver for open quantum systems
    
    The Lindblad master equation describes the evolution of an open
    quantum system:
    
    dρ/dt = -i/ħ [H, ρ] + Σ_k (L_k ρ L_k† - 1/2{L_k†L_k, ρ})
    
    where H is the Hamiltonian and L_k are collapse operators.
    
    Example:
        >>> from psiqit.quantum import pauli_z, pauli_x
        >>> H = pauli_z()  # Hamiltonian
        >>> L = pauli_x()  # Collapse operator
        >>> solver = LindbladSolver(H, [L], gamma=[0.1])
        >>> rho0 = np.array([[1, 0], [0, 0]])  # Initial state |0⟩⟨0|
        >>> result = solver.evolve(rho0, t_max=10.0, dt=0.01)
        >>> print(result.final_state)
    """
    
    def __init__(
        self,
        hamiltonian: Operator,
        collapse_ops: List[Operator],
        gamma: Optional[List[float]] = None,
        hbar: float = 1.0
    ):
        """
        Initialize Lindblad solver
        
        Args:
            hamiltonian: System Hamiltonian (Hermitian)
            collapse_ops: List of collapse operators L_k
            gamma: List of decay rates for each collapse operator
                   If None, all rates are set to 1.0
            hbar: Reduced Planck constant
        
        Example:
            >>> H = pauli_z()
            >>> L = pauli_x()
            >>> solver = LindbladSolver(H, [L], gamma=[0.1])
        """
        if not hamiltonian.is_hermitian:
            logger.warning("Hamiltonian is not Hermitian")
        
        self.hamiltonian = hamiltonian
        self.collapse_ops = collapse_ops
        self.hbar = hbar
        self.dim = hamiltonian.dim
        
        # Set decay rates
        if gamma is None:
            self.gamma = [1.0] * len(collapse_ops)
        else:
            if len(gamma) != len(collapse_ops):
                raise ValueError(f"Number of gamma values {len(gamma)} does not match number of collapse operators {len(collapse_ops)}")
            self.gamma = gamma
        
        # Precompute Lindblad superoperator components
        self._lindblad_components = self._compute_lindblad_components()
        
        logger.info(f"LindbladSolver initialized: dim={self.dim}, n_ops={len(collapse_ops)}")
    
    def _compute_lindblad_components(self) -> List[np.ndarray]:
        """
        Precompute Lindblad superoperator components
        
        Returns:
            List[np.ndarray]: List of Lindblad matrices
        """
        components = []
        
        for L, gamma_k in zip(self.collapse_ops, self.gamma):
            L_dag = L.dagger()
            L_dag_L = L_dag @ L
            
            # Lindblad term: gamma * (L ρ L† - 1/2{L†L, ρ})
            # We'll compute this in the evolution
            components.append({
                'L': L.data,
                'L_dag': L_dag.data,
                'L_dag_L': L_dag_L.data,
                'gamma': gamma_k
            })
        
        return components
    
    def _lindblad_superoperator(self, rho: np.ndarray) -> np.ndarray:
        """
        Apply the Lindblad superoperator to a density matrix
        
        Args:
            rho: Density matrix
            
        Returns:
            np.ndarray: dρ/dt
        """
        # Unitary part: -i/ħ [H, ρ]
        H = self.hamiltonian.data
        drho = -1j * (H @ rho - rho @ H) / self.hbar
        
        # Dissipative part
        for comp in self._lindblad_components:
            L = comp['L']
            L_dag = comp['L_dag']
            L_dag_L = comp['L_dag_L']
            gamma = comp['gamma']
            
            # L ρ L† - 1/2{L†L, ρ}
            term = L @ rho @ L_dag - 0.5 * (L_dag_L @ rho + rho @ L_dag_L)
            drho += gamma * term
        
        return drho
    
    def _flatten_matrix(self, rho: np.ndarray) -> np.ndarray:
        """Flatten a density matrix to a vector"""
        return rho.flatten()
    
    def _unflatten_vector(self, vec: np.ndarray) -> np.ndarray:
        """Unflatten a vector to a density matrix"""
        return vec.reshape(self.dim, self.dim)
    
    def _ensure_hermitian(self, rho: np.ndarray) -> np.ndarray:
        """Ensure the density matrix is Hermitian"""
        return (rho + rho.conj().T) / 2
    
    def _ensure_trace(self, rho: np.ndarray) -> np.ndarray:
        """Ensure the density matrix has trace 1"""
        trace_val = np.trace(rho).real
        if trace_val > 0:
            rho = rho / trace_val
        return rho
    
    def evolve(
        self,
        rho0: np.ndarray,
        t_max: float,
        dt: float = 0.01,
        method: str = 'euler',
        save_every: int = 1
    ) -> LindbladResult:
        """
        Evolve the density matrix according to the Lindblad master equation
        
        Args:
            rho0: Initial density matrix
            t_max: Maximum evolution time
            dt: Time step
            method: Integration method ('euler' or 'rk4')
            save_every: Save state every N steps
            
        Returns:
            LindbladResult: Evolution results
        
        Example:
            >>> rho0 = np.array([[1, 0], [0, 0]])
            >>> result = solver.evolve(rho0, t_max=10.0, dt=0.01)
        """
        # Validate initial state
        if rho0.shape != (self.dim, self.dim):
            raise ValueError(f"rho0 shape {rho0.shape} does not match dimension {self.dim}")
        
        # Ensure Hermitian and trace 1
        rho0 = self._ensure_hermitian(rho0)
        rho0 = self._ensure_trace(rho0)
        
        logger.info(f"Starting Lindblad evolution: t_max={t_max:.3f}, dt={dt:.4f}, method={method}")
        
        # Number of steps
        n_steps = int(np.ceil(t_max / dt))
        dt_actual = t_max / n_steps
        
        # Initialize
        rho = rho0.copy()
        
        # Storage
        times = []
        states = []
        populations = []
        
        # Initial state
        times.append(0.0)
        states.append(rho.copy())
        
        # Initial populations (diagonal elements)
        diag = np.diag(rho).real
        populations.append(diag.copy())
        
        success = True
        message = "Evolution completed successfully"
        
        try:
            for step in range(n_steps):
                t = step * dt_actual
                
                # Flatten for ODE solvers
                rho_flat = self._flatten_matrix(rho)
                
                if method == 'euler':
                    # Euler method
                    drho = self._lindblad_superoperator(rho)
                    rho = rho + dt_actual * drho
                
                elif method == 'rk4':
                    # RK4 method
                    k1 = self._lindblad_superoperator(rho)
                    k2 = self._lindblad_superoperator(rho + dt_actual/2 * k1)
                    k3 = self._lindblad_superoperator(rho + dt_actual/2 * k2)
                    k4 = self._lindblad_superoperator(rho + dt_actual * k3)
                    
                    rho = rho + dt_actual/6 * (k1 + 2*k2 + 2*k3 + k4)
                
                else:
                    raise ValueError(f"Unknown method: {method}. Use 'euler' or 'rk4'")
                
                # Ensure Hermitian
                rho = self._ensure_hermitian(rho)
                
                # Ensure trace 1
                rho = self._ensure_trace(rho)
                
                # Check positivity (simplified)
                eigvals = np.linalg.eigvalsh(rho)
                if np.any(eigvals < -1e-8):
                    logger.warning(f"Negative eigenvalues at step {step}: {eigvals}")
                    # Project to positive semidefinite
                    rho = self._project_positive(rho)
                
                # Save state
                if (step + 1) % save_every == 0 or step == n_steps - 1:
                    times.append((step + 1) * dt_actual)
                    states.append(rho.copy())
                    
                    diag = np.diag(rho).real
                    populations.append(diag.copy())
                
                # Progress logging
                if (step + 1) % max(1, n_steps // 10) == 0:
                    logger.debug(f"Progress: {step+1}/{n_steps} steps")
        
        except Exception as e:
            success = False
            message = f"Evolution failed: {str(e)}"
            logger.error(message)
        
        # Convert lists to arrays
        times = np.array(times)
        
        logger.info(f"Lindblad evolution completed: {len(states)} states saved")
        
        return LindbladResult(
            times=times,
            states=states,
            populations=populations,
            success=success,
            message=message
        )
    
    def _project_positive(self, rho: np.ndarray) -> np.ndarray:
        """
        Project a matrix to the positive semidefinite cone
        
        Args:
            rho: Density matrix
            
        Returns:
            np.ndarray: Positive semidefinite matrix
        """
        eigvals, eigvecs = np.linalg.eigh(rho)
        
        # Set negative eigenvalues to zero
        eigvals = np.maximum(eigvals, 0)
        
        # Reconstruct
        rho_pos = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        
        # Ensure trace 1
        rho_pos = self._ensure_trace(rho_pos)
        
        return rho_pos
    
    def steady_state(
        self,
        tol: float = 1e-10,
        max_iter: int = 1000,
        initial_guess: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Find the steady state (stationary solution) of the Lindblad equation
        
        Solves: dρ/dt = 0
        
        Args:
            tol: Convergence tolerance
            max_iter: Maximum number of iterations
            initial_guess: Initial guess for steady state
            
        Returns:
            np.ndarray: Steady state density matrix
        
        Example:
            >>> rho_ss = solver.steady_state()
            >>> print(np.trace(rho_ss))  # 1.0
        """
        logger.info(f"Finding steady state with tolerance {tol:.2e}")
        
        # Default initial guess: thermal state or maximally mixed
        if initial_guess is None:
            rho = np.eye(self.dim, dtype=complex) / self.dim
        else:
            rho = initial_guess.copy()
            rho = self._ensure_hermitian(rho)
            rho = self._ensure_trace(rho)
        
        # Iterative method (imaginary time evolution)
        # For small systems, we can use direct solution
        # For larger systems, we use iterative method
        
        # Try direct solution for small systems
        if self.dim <= 4:
            try:
                # Build the Liouvillian superoperator
                Liouv = self._build_liouvillian()
                
                # Find nullspace (eigenvalue = 0)
                eigvals, eigvecs = np.linalg.eig(Liouv)
                
                # Find index of zero eigenvalue
                zero_idx = np.argmin(np.abs(eigvals))
                
                # Get steady state vector
                rho_vec = eigvecs[:, zero_idx]
                
                # Reshape to matrix
                rho_ss = rho_vec.reshape(self.dim, self.dim)
                
                # Ensure Hermitian and trace 1
                rho_ss = self._ensure_hermitian(rho_ss)
                rho_ss = self._ensure_trace(rho_ss)
                
                logger.info("Steady state found using direct method")
                return rho_ss
            
            except Exception as e:
                logger.warning(f"Direct method failed: {e}. Using iterative method.")
        
        # Iterative method (simple fixed-point iteration)
        # rho_{n+1} = rho_n + dt * Lindblad(rho_n)
        # We evolve until convergence
        
        dt = 0.01
        for iteration in range(max_iter):
            drho = self._lindblad_superoperator(rho)
            rho_new = rho + dt * drho
            
            # Ensure Hermitian and trace 1
            rho_new = self._ensure_hermitian(rho_new)
            rho_new = self._ensure_trace(rho_new)
            rho_new = self._project_positive(rho_new)
            
            # Check convergence
            diff = np.max(np.abs(rho_new - rho))
            rho = rho_new
            
            if diff < tol:
                logger.info(f"Steady state converged after {iteration + 1} iterations")
                return rho
        
        logger.warning(f"Steady state did not converge after {max_iter} iterations")
        return rho
    
    def _build_liouvillian(self) -> np.ndarray:
        """
        Build the Liouvillian superoperator matrix
        
        Returns:
            np.ndarray: Liouvillian superoperator (dim² x dim²)
        """
        dim = self.dim
        dim2 = dim * dim
        
        Liouv = np.zeros((dim2, dim2), dtype=complex)
        
        # Unitary part: -i/ħ [H, ρ]
        H = self.hamiltonian.data
        
        for i in range(dim):
            for j in range(dim):
                # Index for ρ_{ij}
                idx = i * dim + j
                
                # -i/ħ (H ρ - ρ H)
                for k in range(dim):
                    # H ρ term: -i/ħ * H_{ik} ρ_{kj}
                    Liouv[idx, k * dim + j] -= 1j * H[i, k] / self.hbar
                    
                    # - ρ H term: +i/ħ * ρ_{ik} H_{kj}
                    Liouv[idx, i * dim + k] += 1j * H[k, j] / self.hbar
        
        # Dissipative part
        for comp in self._lindblad_components:
            L = comp['L']
            L_dag = comp['L_dag']
            L_dag_L = comp['L_dag_L']
            gamma = comp['gamma']
            
            # L ρ L† term
            for i in range(dim):
                for j in range(dim):
                    idx = i * dim + j
                    for k in range(dim):
                        for l in range(dim):
                            # (L ρ L†)_{ij} = Σ_{kl} L_{ik} ρ_{kl} L†_{lj}
                            Liouv[idx, k * dim + l] += gamma * L[i, k] * L_dag[l, j]
            
            # -1/2 {L†L, ρ} term
            for i in range(dim):
                for j in range(dim):
                    idx = i * dim + j
                    # -1/2 (L†L ρ)_{ij}
                    for k in range(dim):
                        Liouv[idx, k * dim + j] -= 0.5 * gamma * L_dag_L[i, k]
                    # -1/2 (ρ L†L)_{ij}
                    for k in range(dim):
                        Liouv[idx, i * dim + k] -= 0.5 * gamma * L_dag_L[k, j]
        
        return Liouv
    
    def get_steady_state_analytical(self) -> Optional[np.ndarray]:
        """
        Try to find the steady state analytically (for simple systems)
        
        Returns:
            Optional[np.ndarray]: Steady state or None if not found
        """
        # For two-level systems, we can solve analytically
        if self.dim == 2 and len(self.collapse_ops) >= 1:
            # For a two-level system with one collapse operator
            # This is a simplified version
            try:
                rho_ss = self.steady_state(tol=1e-12, max_iter=1000)
                return rho_ss
            except:
                return None
        
        return None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def thermal_state(
    H: Operator,
    beta: float,
    hbar: float = 1.0
) -> np.ndarray:
    """
    Create a thermal (Gibbs) state
    
    ρ = e^{-βH} / Tr(e^{-βH})
    
    Args:
        H: Hamiltonian
        beta: Inverse temperature (1/k_B T)
        hbar: Reduced Planck constant
        
    Returns:
        np.ndarray: Thermal density matrix
        
    Example:
        >>> from psiqit.quantum import pauli_z
        >>> rho = thermal_state(pauli_z(), beta=1.0)
        >>> print(np.trace(rho))  # 1.0
    """
    H_data = H.data
    exp_neg_beta_H = np.linalg.expm(-beta * H_data)
    rho = exp_neg_beta_H / np.trace(exp_neg_beta_H).real
    return rho


def dephasing_channel(
    rho: np.ndarray,
    gamma: float,
    dt: float
) -> np.ndarray:
    """
    Apply a dephasing channel to a density matrix
    
    ρ → (1 - p)ρ + p Z ρ Z
    
    where p = (1 - e^{-γt})/2
    
    Args:
        rho: Density matrix
        gamma: Dephasing rate
        dt: Time step
        
    Returns:
        np.ndarray: Dephased density matrix
    """
    from ..quantum.operator import pauli_z
    
    Z = pauli_z().data
    p = (1 - np.exp(-gamma * dt)) / 2
    
    rho_new = (1 - p) * rho + p * Z @ rho @ Z
    return rho_new


def amplitude_damping_channel(
    rho: np.ndarray,
    gamma: float,
    dt: float
) -> np.ndarray:
    """
    Apply an amplitude damping channel to a density matrix
    
    Args:
        rho: Density matrix
        gamma: Damping rate
        dt: Time step
        
    Returns:
        np.ndarray: Damped density matrix
    """
    from ..quantum.operator import pauli_x, pauli_y, pauli_z
    
    # Amplitude damping operators
    p = 1 - np.exp(-gamma * dt)
    
    # Kraus operators
    K0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)
    
    rho_new = K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T
    return rho_new


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'LindbladResult',
    'LindbladSolver',
    'thermal_state',
    'dephasing_channel',
    'amplitude_damping_channel',
]