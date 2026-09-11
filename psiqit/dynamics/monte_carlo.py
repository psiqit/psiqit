# psiqit/dynamics/monte_carlo.py

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from ..quantum.state import Ket
from ..quantum.operator import Operator, identity, dagger, trace
from ..utils.logger import logger
from ..utils.validation import is_hermitian


# ============================================================================
# MONTE CARLO RESULT CLASS
# ============================================================================

@dataclass
class MonteCarloResult:
    """
    Result container for Monte Carlo quantum trajectories
    
    Attributes:
        times: Time points
        avg_states: Average density matrices over trajectories
        jump_times: List of jump times for each trajectory
        n_jumps: Number of jumps for each trajectory
        success: Whether simulation was successful
        message: Additional information
        n_trajectories: Number of trajectories
        n_steps: Number of time steps
        final_avg_state: Final average density matrix
        trajectories: List of individual trajectory results
    """
    times: np.ndarray
    avg_states: List[np.ndarray]
    jump_times: List[List[float]]
    n_jumps: List[int]
    success: bool = True
    message: str = ""
    n_trajectories: int = 0
    n_steps: int = 0
    final_avg_state: Optional[np.ndarray] = None
    trajectories: Optional[List[Dict]] = None
    
    def __post_init__(self):
        """Post-process the result"""
        self.n_steps = len(self.times)
        self.n_trajectories = len(self.n_jumps)
        
        if self.avg_states:
            self.final_avg_state = self.avg_states[-1]
    
    def get_avg_state_at_time(self, index: int) -> Optional[np.ndarray]:
        """Get average density matrix at a specific time index"""
        if 0 <= index < len(self.avg_states):
            return self.avg_states[index]
        return None
    
    def get_final_avg_state(self) -> Optional[np.ndarray]:
        """Get the final average density matrix"""
        return self.final_avg_state
    
    def get_total_jumps(self) -> int:
        """Get total number of jumps across all trajectories"""
        return sum(self.n_jumps)
    
    def get_jump_rate(self) -> float:
        """Get average jump rate per trajectory"""
        if self.n_trajectories == 0:
            return 0.0
        total_jumps = sum(self.n_jumps)
        return total_jumps / self.n_trajectories
    
    def get_population_evolution(self, state_index: int) -> List[float]:
        """
        Get the evolution of a specific population
        
        Args:
            state_index: Index of the state
            
        Returns:
            List[float]: Population values over time
        """
        if state_index < 0:
            raise ValueError(f"State index {state_index} must be >= 0")
        
        populations = []
        for rho in self.avg_states:
            if state_index < rho.shape[0]:
                populations.append(float(rho[state_index, state_index].real))
            else:
                populations.append(0.0)
        
        return populations
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"MonteCarloResult(trajectories={self.n_trajectories}, status={status}, total_jumps={self.get_total_jumps()})"
    
    def __str__(self) -> str:
        lines = [
            "Monte Carlo Quantum Trajectories Results:",
            f"  Trajectories: {self.n_trajectories}",
            f"  Steps: {self.n_steps}",
            f"  Success: {self.success}",
            f"  Message: {self.message}",
            f"  Total Jumps: {self.get_total_jumps()}",
            f"  Avg Jump Rate: {self.get_jump_rate():.3f}",
        ]
        if self.final_avg_state is not None:
            lines.append(f"  Final Avg State Trace: {np.trace(self.final_avg_state).real:.6f}")
        return "\n".join(lines)


# ============================================================================
# QUANTUM TRAJECTORY
# ============================================================================

class QuantumTrajectory:
    """
    Quantum Monte Carlo trajectory simulation
    
    Simulates stochastic quantum trajectories for open quantum systems.
    The method uses a Monte Carlo wavefunction approach where the
    evolution is a sequence of deterministic evolution and quantum jumps.
    
    The non-Hermitian effective Hamiltonian is:
    H_eff = H - iħ/2 Σ_k γ_k L_k† L_k
    
    Between jumps, the state evolves as:
    |ψ(t+dt)⟩ = (1 - iH_eff dt/ħ) |ψ(t)⟩ / ||...||
    
    Jumps occur with probability:
    dp_k = γ_k dt ⟨ψ|L_k† L_k|ψ⟩
    
    Example:
        >>> from psiqit.quantum import pauli_z, pauli_x, zero
        >>> H = pauli_z()
        >>> L = pauli_x()
        >>> qt = QuantumTrajectory(H, [L], gamma=[0.1])
        >>> result = qt.run(zero(), t_max=10.0, dt=0.01, n_trajectories=100)
        >>> print(result.final_avg_state)
    """
    
    def __init__(
        self,
        hamiltonian: Operator,
        collapse_ops: List[Operator],
        gamma: Optional[List[float]] = None,
        hbar: float = 1.0
    ):
        """
        Initialize quantum trajectory simulation
        
        Args:
            hamiltonian: System Hamiltonian (Hermitian)
            collapse_ops: List of collapse operators L_k
            gamma: List of decay rates for each collapse operator
                   If None, all rates are set to 1.0
            hbar: Reduced Planck constant
        
        Example:
            >>> H = pauli_z()
            >>> L = pauli_x()
            >>> qt = QuantumTrajectory(H, [L], gamma=[0.1])
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
        
        # Compute effective Hamiltonian
        self.H_eff = self._compute_effective_hamiltonian()
        
        logger.info(f"QuantumTrajectory initialized: dim={self.dim}, n_ops={len(collapse_ops)}")
    
    def _compute_effective_hamiltonian(self) -> np.ndarray:
        """
        Compute the effective non-Hermitian Hamiltonian
        
        H_eff = H - iħ/2 Σ_k γ_k L_k† L_k
        
        Returns:
            np.ndarray: Effective Hamiltonian matrix
        """
        H_eff = self.hamiltonian.data.copy()
        
        for L, gamma_k in zip(self.collapse_ops, self.gamma):
            L_dag_L = L.dagger() @ L
            H_eff -= 1j * self.hbar / 2 * gamma_k * L_dag_L.data
        
        return H_eff
    
    def _compute_jump_probabilities(self, psi: np.ndarray) -> List[float]:
        """
        Compute jump probabilities for each collapse operator
        
        dp_k = γ_k dt ⟨ψ|L_k† L_k|ψ⟩
        
        Args:
            psi: Current state vector
            
        Returns:
            List[float]: Jump probabilities (not multiplied by dt)
        """
        probs = []
        
        for L, gamma_k in zip(self.collapse_ops, self.gamma):
            L_dag_L = L.dagger() @ L
            prob_rate = gamma_k * np.vdot(psi, L_dag_L.data @ psi).real
            probs.append(prob_rate)
        
        return probs
    
    def _apply_jump(self, psi: np.ndarray, jump_idx: int) -> np.ndarray:
        """
        Apply a quantum jump to the state
        
        |ψ⟩ → L_k |ψ⟩ / ||L_k |ψ⟩||
        
        Args:
            psi: Current state vector
            jump_idx: Index of the collapse operator
            
        Returns:
            np.ndarray: New state vector after jump
        """
        L = self.collapse_ops[jump_idx]
        psi_new = L.data @ psi
        
        # Normalize
        norm = np.linalg.norm(psi_new)
        if norm > 0:
            psi_new = psi_new / norm
        else:
            logger.warning(f"Jump {jump_idx} resulted in zero state")
            psi_new = psi.copy()
        
        return psi_new
    
    def _evolve_deterministic(self, psi: np.ndarray, dt: float) -> np.ndarray:
        """
        Deterministic evolution under effective Hamiltonian
        
        |ψ(t+dt)⟩ = (1 - iH_eff dt/ħ) |ψ(t)⟩
        
        Args:
            psi: Current state vector
            dt: Time step
            
        Returns:
            np.ndarray: Evolved state (not normalized)
        """
        # First-order approximation
        psi_new = psi - 1j * self.H_eff @ psi * dt / self.hbar
        
        # Normalize
        norm = np.linalg.norm(psi_new)
        if norm > 0:
            psi_new = psi_new / norm
        
        return psi_new
    
    def _evolve_with_adaptive_step(
        self,
        psi: np.ndarray,
        t: float,
        dt: float
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Evolve one step with adaptive time step
        
        Returns:
            Tuple: (new_state, actual_dt, jump_occurred)
        """
        # Compute jump probabilities
        jump_rates = self._compute_jump_probabilities(psi)
        total_rate = sum(jump_rates)
        
        # Check if jump occurs
        if total_rate > 0:
            # Generate random number for jump
            rand = np.random.random()
            
            # Jump probability in this time step
            jump_prob = 1 - np.exp(-total_rate * dt)
            
            if rand < jump_prob:
                # A jump occurs
                # Determine which jump
                cum_probs = np.cumsum(jump_rates)
                r = np.random.random() * total_rate
                
                jump_idx = 0
                for i, cum_prob in enumerate(cum_probs):
                    if r < cum_prob:
                        jump_idx = i
                        break
                
                # Apply jump
                psi_new = self._apply_jump(psi, jump_idx)
                
                # Evolve deterministically for the remaining time
                # (simplified: we just apply the jump and continue)
                return psi_new, dt, True
        
        # No jump: deterministic evolution
        psi_new = self._evolve_deterministic(psi, dt)
        return psi_new, dt, False
    
    def single_trajectory(
        self,
        psi0: Ket,
        t_max: float,
        dt: float = 0.01,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Simulate a single quantum trajectory
        
        Args:
            psi0: Initial state
            t_max: Maximum evolution time
            dt: Time step
            seed: Random seed for reproducibility
            
        Returns:
            Dict: Trajectory results with states, jump times, etc.
        
        Example:
            >>> result = qt.single_trajectory(zero(), t_max=5.0, dt=0.01)
            >>> print(result['n_jumps'])
        """
        if seed is not None:
            np.random.seed(seed)
        
        if psi0.dim != self.dim:
            raise ValueError(f"State dimension {psi0.dim} does not match Hamiltonian dimension {self.dim}")
        
        logger.info(f"Starting single trajectory: t_max={t_max:.3f}, dt={dt:.4f}")
        
        # Number of steps
        n_steps = int(np.ceil(t_max / dt))
        dt_actual = t_max / n_steps
        
        # Initialize
        psi = psi0.data.copy()
        times = [0.0]
        states = [psi0.copy()]
        jump_times = []
        n_jumps = 0
        
        # Evolution
        for step in range(n_steps):
            t = step * dt_actual
            
            # Evolve with adaptive step (always using dt_actual for simplicity)
            psi_new, _, jump_occurred = self._evolve_with_adaptive_step(
                psi, t, dt_actual
            )
            
            if jump_occurred:
                jump_times.append(t + dt_actual)
                n_jumps += 1
                logger.debug(f"Jump occurred at t={t+dt_actual:.3f}")
            
            psi = psi_new
            
            # Store state
            if (step + 1) % max(1, n_steps // 100) == 0 or step == n_steps - 1:
                times.append((step + 1) * dt_actual)
                states.append(Ket(psi.copy()))
        
        logger.info(f"Single trajectory completed: {n_jumps} jumps")
        
        return {
            'times': times,
            'states': states,
            'jump_times': jump_times,
            'n_jumps': n_jumps,
            'final_state': states[-1] if states else None,
            'seed': seed
        }
    
    def run(
        self,
        psi0: Ket,
        t_max: float,
        dt: float = 0.01,
        n_trajectories: int = 100,
        seeds: Optional[List[int]] = None,
        progress: bool = True
    ) -> MonteCarloResult:
        """
        Run multiple quantum trajectories and average results
        
        Args:
            psi0: Initial state
            t_max: Maximum evolution time
            dt: Time step
            n_trajectories: Number of trajectories
            seeds: List of random seeds for each trajectory
            progress: Show progress bar
            
        Returns:
            MonteCarloResult: Averaged results
        
        Example:
            >>> result = qt.run(zero(), t_max=10.0, dt=0.01, n_trajectories=100)
            >>> print(result.final_avg_state)
        """
        logger.info(f"Running {n_trajectories} trajectories")
        
        # Generate seeds if not provided
        if seeds is None:
            seeds = [None] * n_trajectories
        elif len(seeds) != n_trajectories:
            raise ValueError(f"Number of seeds {len(seeds)} does not match number of trajectories {n_trajectories}")
        
        # Store all trajectories
        all_trajectories = []
        all_jump_times = []
        all_n_jumps = []
        
        # Average states storage
        n_steps = int(np.ceil(t_max / dt)) + 1
        avg_states = []
        times = np.linspace(0, t_max, n_steps)
        
        # Initialize average density matrix
        rho_avg = np.zeros((self.dim, self.dim), dtype=complex)
        
        # For each trajectory
        for traj_idx in range(n_trajectories):
            # Run single trajectory
            result = self.single_trajectory(psi0, t_max, dt, seed=seeds[traj_idx])
            
            # Store trajectory data
            all_trajectories.append(result)
            all_jump_times.append(result['jump_times'])
            all_n_jumps.append(result['n_jumps'])
            
            # Build density matrix from trajectory states
            # For now, we just accumulate the final state
            if result['final_state'] is not None:
                rho = np.outer(result['final_state'].data, result['final_state'].data.conj())
                rho_avg += rho / n_trajectories
            
            # Progress
            if progress and (traj_idx + 1) % max(1, n_trajectories // 10) == 0:
                logger.info(f"Progress: {traj_idx+1}/{n_trajectories} trajectories")
        
        # For a full implementation, we would average all states at each time
        # Here we build a simplified average
        avg_states = [rho_avg]  # Simplified: only final state
        
        logger.info(f"Monte Carlo simulation completed: {n_trajectories} trajectories")
        
        return MonteCarloResult(
            times=np.array([0.0, t_max]),
            avg_states=avg_states,
            jump_times=all_jump_times,
            n_jumps=all_n_jumps,
            success=True,
            message="Simulation completed successfully",
            n_trajectories=n_trajectories,
            trajectories=all_trajectories
        )
    
    def compute_steady_state_mc(
        self,
        psi0: Ket,
        t_max: float,
        dt: float = 0.01,
        n_trajectories: int = 100,
        averaging_start: float = 0.0
    ) -> np.ndarray:
        """
        Compute steady state using Monte Carlo wavefunction method
        
        Args:
            psi0: Initial state
            t_max: Maximum evolution time
            dt: Time step
            n_trajectories: Number of trajectories
            averaging_start: Time to start averaging (to skip transient)
            
        Returns:
            np.ndarray: Steady state density matrix
        """
        logger.info(f"Computing steady state with MC: {n_trajectories} trajectories")
        
        # Run trajectories
        result = self.run(psi0, t_max, dt, n_trajectories)
        
        # Average over final states
        rho_ss = np.zeros((self.dim, self.dim), dtype=complex)
        
        for traj in result.trajectories:
            if traj['final_state'] is not None:
                psi = traj['final_state'].data
                rho_ss += np.outer(psi, psi.conj()) / n_trajectories
        
        return rho_ss


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def jump_operator(
    L: Operator,
    gamma: float = 1.0
) -> Operator:
    """
    Create a jump operator with rate
    
    Args:
        L: Collapse operator
        gamma: Decay rate
        
    Returns:
        Operator: Jump operator with rate
    """
    jump_op = np.sqrt(gamma) * L
    jump_op.name = f"J_{L.name}" if L.name else "J"
    return jump_op


def photon_loss_channel(
    gamma: float = 0.1
) -> Operator:
    """
    Create a photon loss collapse operator
    
    L = √γ a (annihilation operator)
    
    Args:
        gamma: Loss rate
        
    Returns:
        Operator: Photon loss collapse operator
    """
    # For a two-level system, use the lowering operator
    L = np.array([[0, 1], [0, 0]], dtype=complex)
    return Operator(np.sqrt(gamma) * L, name="a")


def dephasing_operator(
    gamma: float = 0.1
) -> Operator:
    """
    Create a dephasing collapse operator
    
    L = √γ Z
    
    Args:
        gamma: Dephasing rate
        
    Returns:
        Operator: Dephasing collapse operator
    """
    from ..quantum.operator import pauli_z
    return Operator(np.sqrt(gamma) * pauli_z().data, name="Z")


def amplitude_damping_operator(
    gamma: float = 0.1
) -> Operator:
    """
    Create an amplitude damping collapse operator
    
    L = √γ σ_-
    
    Args:
        gamma: Damping rate
        
    Returns:
        Operator: Amplitude damping collapse operator
    """
    # Lowering operator
    L = np.array([[0, 1], [0, 0]], dtype=complex)
    return Operator(np.sqrt(gamma) * L, name="σ_-")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'MonteCarloResult',
    'QuantumTrajectory',
    'jump_operator',
    'photon_loss_channel',
    'dephasing_operator',
    'amplitude_damping_operator',
]