#psiqit/quantum/interference.py

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from ..math.qalgebra import PI, SQRT2
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# INTERFERENCE RESULT CLASS
# ============================================================================

@dataclass
class InterferenceResult:
    """
    Result container for interference calculations
    
    Attributes:
        x: Position array
        intensity: Intensity array
        visibility: Fringe visibility (0-1)
        fringe_spacing: Distance between adjacent fringes
        max_intensity: Maximum intensity
        min_intensity: Minimum intensity
        contrast: Contrast ratio (max - min) / (max + min)
    """
    x: np.ndarray
    intensity: np.ndarray
    visibility: float = 0.0
    fringe_spacing: float = 0.0
    max_intensity: float = 0.0
    min_intensity: float = 0.0
    contrast: float = 0.0
    
    def __post_init__(self):
        """Calculate derived quantities"""
        if len(self.intensity) > 0:
            self.max_intensity = float(np.max(self.intensity))
            self.min_intensity = float(np.min(self.intensity))
            
            if self.max_intensity + self.min_intensity > 0:
                self.contrast = (self.max_intensity - self.min_intensity) / \
                               (self.max_intensity + self.min_intensity)
                self.visibility = self.contrast
    
    def get_peaks(self) -> np.ndarray:
        """Find positions of intensity maxima"""
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(self.intensity)
        return self.x[peaks]
    
    def get_peaks_indices(self) -> np.ndarray:
        """Find indices of intensity maxima"""
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(self.intensity)
        return peaks
    
    def __repr__(self) -> str:
        return f"InterferenceResult(visibility={self.visibility:.3f}, fringe_spacing={self.fringe_spacing:.4f})"


# ============================================================================
# DOUBLE SLIT INTERFERENCE
# ============================================================================

class DoubleSlit:
    """
    Double-slit interference simulation with quantum probability
    
    This class simulates the double-slit experiment including both
    single-slit diffraction and double-slit interference patterns.
    
    Example:
        >>> ds = DoubleSlit(slit_distance=1e-3, slit_width=0.1e-3, wavelength=500e-9)
        >>> result = ds.pattern(x_range=(-0.01, 0.01), n_points=1000)
        >>> print(f"Visibility: {result.visibility:.3f}")
    """
    
    def __init__(
        self,
        slit_distance: float,
        slit_width: float,
        wavelength: float,
        distance_to_screen: float = 1.0
    ):
        """
        Initialize double-slit setup
        
        Args:
            slit_distance: Distance between the two slits (d)
            slit_width: Width of each slit (a)
            wavelength: Wavelength of the light/particles (λ)
            distance_to_screen: Distance from slits to screen (L)
        """
        self.slit_distance = slit_distance
        self.slit_width = slit_width
        self.wavelength = wavelength
        self.distance_to_screen = distance_to_screen
        
        # Wave number
        self.k = 2 * PI / wavelength
        
        logger.info(f"Double slit initialized: d={slit_distance:.2e}m, a={slit_width:.2e}m, λ={wavelength:.2e}m")
    
    def _single_slit_diffraction(self, theta: float) -> float:
        """
        Single-slit diffraction pattern
        
        I(θ) = I₀ [sin(β)/β]² where β = (π a sin θ)/λ
        
        Args:
            theta: Angle from center
            
        Returns:
            float: Diffraction intensity (relative to max)
        """
        if self.slit_width == 0:
            return 1.0
        
        beta = PI * self.slit_width * np.sin(theta) / self.wavelength
        
        if abs(beta) < 1e-15:
            return 1.0
        
        return (np.sin(beta) / beta) ** 2
    
    def _double_slit_interference(self, theta: float) -> float:
        """
        Double-slit interference pattern
        
        I(θ) = I₀ cos²(δ/2) where δ = (2π d sin θ)/λ
        
        Args:
            theta: Angle from center
            
        Returns:
            float: Interference intensity (relative to max)
        """
        delta = 2 * PI * self.slit_distance * np.sin(theta) / self.wavelength
        return np.cos(delta / 2) ** 2
    
    def intensity(self, theta: float) -> float:
        """
        Total intensity at angle θ
        
        I(θ) = I₀ [sin(β)/β]² cos²(δ/2)
        
        Args:
            theta: Angle from center
            
        Returns:
            float: Total intensity
        """
        return self._single_slit_diffraction(theta) * self._double_slit_interference(theta)
    
    def pattern(
        self,
        x_range: Tuple[float, float],
        n_points: int = 1000
    ) -> InterferenceResult:
        """
        Calculate interference pattern on the screen
        
        Args:
            x_range: (x_min, x_max) on the screen
            n_points: Number of points to calculate
            
        Returns:
            InterferenceResult: Pattern with intensities
            
        Example:
            >>> ds = DoubleSlit(1e-3, 0.1e-3, 500e-9)
            >>> result = ds.pattern((-0.01, 0.01), 2000)
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(result.x, result.intensity)
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        
        # Small angle approximation: θ ≈ x/L
        theta = x / self.distance_to_screen
        
        intensity = np.array([self.intensity(t) for t in theta])
        
        # Normalize intensity
        if np.max(intensity) > 0:
            intensity = intensity / np.max(intensity)
        
        # Calculate fringe spacing
        fringe_spacing = self._calculate_fringe_spacing(x, intensity)
        
        logger.info(f"Double-slit pattern calculated with {n_points} points")
        
        return InterferenceResult(
            x=x,
            intensity=intensity,
            fringe_spacing=fringe_spacing
        )
    
    def _calculate_fringe_spacing(self, x: np.ndarray, intensity: np.ndarray) -> float:
        """
        Calculate fringe spacing from intensity pattern
        
        Args:
            x: Position array
            intensity: Intensity array
            
        Returns:
            float: Fringe spacing
        """
        try:
            from scipy.signal import find_peaks
            
            # Find peaks
            peaks, _ = find_peaks(intensity, height=0.5 * np.max(intensity))
            
            if len(peaks) >= 2:
                # Average spacing between peaks
                spacings = np.diff(x[peaks])
                return float(np.mean(spacings))
            
            # Theoretical fringe spacing: Δx = λL/d
            theoretical = self.wavelength * self.distance_to_screen / self.slit_distance
            return theoretical
            
        except ImportError:
            # Use theoretical value
            return self.wavelength * self.distance_to_screen / self.slit_distance
    
    def quantum_probability(self, position: float, n_qubits: int = 8) -> float:
        """
        Calculate quantum probability at a position using superposition
        
        This simulates the quantum mechanical probability distribution
        using a qubit representation of the path taken.
        
        Args:
            position: Position on the screen
            n_qubits: Number of qubits for path discretization
            
        Returns:
            float: Quantum probability
            
        Example:
            >>> ds = DoubleSlit(1e-3, 0.1e-3, 500e-9)
            >>> prob = ds.quantum_probability(0.001, n_qubits=10)
        """
        validate_qubits(n_qubits)
        
        # Discretize the path space
        n_paths = 2 ** n_qubits
        
        # Create superposition of all paths
        # This is a simplified quantum model of the double-slit
        theta = position / self.distance_to_screen
        
        # Phase difference between paths
        delta_phase = 2 * PI * self.slit_distance * np.sin(theta) / self.wavelength
        
        # Quantum amplitude for each path
        amplitudes = np.zeros(n_paths, dtype=complex)
        
        # Two main paths (through slit 1 and slit 2)
        path1_phase = delta_phase / 2
        path2_phase = -delta_phase / 2
        
        # Add single-slit diffraction envelope
        diffraction = np.sqrt(self._single_slit_diffraction(theta))
        
        amplitudes[0] = np.exp(1j * path1_phase) * diffraction / SQRT2
        amplitudes[1] = np.exp(1j * path2_phase) * diffraction / SQRT2
        
        # Add some quantum noise (other paths)
        for i in range(2, min(n_paths, 10)):
            amplitudes[i] = 0.01 * np.exp(1j * np.random.uniform(0, 2*PI))
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(amplitudes) ** 2))
        if norm > 0:
            amplitudes = amplitudes / norm
        
        # Probability = |amplitude|²
        probabilities = np.abs(amplitudes) ** 2
        
        # Sum probabilities of all paths
        total_prob = np.sum(probabilities)
        
        return float(total_prob)


# ============================================================================
# MACH-ZEHNDER INTERFEROMETER
# ============================================================================

class MachZehnderInterferometer:
    """
    Mach-Zehnder interferometer simulation
    
    Simulates a Mach-Zehnder interferometer with quantum circuit representation.
    Includes phase shifting and beam splitting operations.
    
    Example:
        >>> mz = MachZehnderInterferometer()
        >>> mz.set_phase(np.pi/2)
        >>> prob = mz.output_probability(output_port=0)
        >>> print(f"Probability at port 0: {prob:.3f}")
    """
    
    def __init__(self):
        """Initialize Mach-Zehnder interferometer"""
        self._phase = 0.0
        self._state = None
        self._circuit = None
        
        # Beam splitter matrix (50:50)
        self.bs = 1 / SQRT2 * np.array([[1, 1j], [1j, 1]], dtype=complex)
        
        logger.info("Mach-Zehnder interferometer initialized")
    
    def set_phase(self, phi: float) -> None:
        """
        Set the phase shift in one arm
        
        Args:
            phi: Phase shift in radians
        """
        self._phase = phi
        logger.debug(f"Phase set to {phi:.4f} rad")
    
    def get_phase(self) -> float:
        """
        Get the current phase shift
        
        Returns:
            float: Phase shift in radians
        """
        return self._phase
    
    def quantum_circuit(self) -> 'QuantumCircuit':
        """
        Create a quantum circuit representation of the interferometer
        
        Returns:
            QuantumCircuit: Circuit with beam splitters and phase shift
            
        Example:
            >>> mz = MachZehnderInterferometer()
            >>> circ = mz.quantum_circuit()
            >>> print(circ.draw())
        """
        # Lazy import to avoid circular dependency
        from ..circuits.circuit import QuantumCircuit
        
        # Create circuit with 1 qubit
        circ = QuantumCircuit(1)
        
        # First beam splitter (Hadamard)
        circ.h(0)
        
        # Phase shift (phase gate)
        circ.rz(self._phase, 0)
        
        # Second beam splitter (Hadamard)
        circ.h(0)
        
        self._circuit = circ
        
        return circ
    
    def get_state(self) -> np.ndarray:
        """
        Get the quantum state after the interferometer
        
        Returns:
            np.ndarray: State vector (complex)
            
        Example:
            >>> mz = MachZehnderInterferometer()
            >>> mz.set_phase(np.pi)
            >>> state = mz.get_state()
            >>> print(state)  # [0, 1] (completely output port 1)
        """
        # Build the circuit
        circ = self.quantum_circuit()
        
        # Run the circuit
        self._state = circ.run()
        
        return self._state
    
    def output_probability(self, output_port: int = 0) -> float:
        """
        Calculate probability of output at a specific port
        
        Args:
            output_port: 0 (upper port) or 1 (lower port)
            
        Returns:
            float: Probability (0-1)
            
        Example:
            >>> mz = MachZehnderInterferometer()
            >>> mz.set_phase(0)
            >>> print(mz.output_probability(0))  # 1.0
            >>> print(mz.output_probability(1))  # 0.0
        """
        if output_port not in [0, 1]:
            raise ValueError(f"Output port must be 0 or 1, got {output_port}")
        
        # Get the state
        state = self.get_state()
        
        # Probability = |amplitude|²
        prob = float(np.abs(state[output_port]) ** 2)
        
        return prob
    
    def interference_fringe(
        self,
        phase_range: Tuple[float, float] = (-2*PI, 2*PI),
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate interference fringe pattern
        
        Args:
            phase_range: (min_phase, max_phase)
            n_points: Number of points
            
        Returns:
            Tuple: (phases, probabilities)
            
        Example:
            >>> mz = MachZehnderInterferometer()
            >>> phases, probs = mz.interference_fringe((0, 4*PI), 200)
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(phases, probs)
        """
        phases = np.linspace(phase_range[0], phase_range[1], n_points)
        probabilities = np.zeros(n_points)
        
        for i, phi in enumerate(phases):
            self.set_phase(phi)
            probabilities[i] = self.output_probability(0)
        
        logger.info(f"Interference fringe calculated with {n_points} points")
        
        return phases, probabilities
    
    def visibility(
        self,
        phase_range: Tuple[float, float] = (-2*PI, 2*PI),
        n_points: int = 100
    ) -> float:
        """
        Calculate fringe visibility
        
        V = (I_max - I_min) / (I_max + I_min)
        
        Args:
            phase_range: Range of phases to scan
            n_points: Number of points
            
        Returns:
            float: Visibility (0-1)
            
        Example:
            >>> mz = MachZehnderInterferometer()
            >>> vis = mz.visibility()
            >>> print(f"Visibility: {vis:.3f}")  # Should be close to 1.0
        """
        phases, probs = self.interference_fringe(phase_range, n_points)
        
        I_max = np.max(probs)
        I_min = np.min(probs)
        
        if I_max + I_min == 0:
            return 0.0
        
        return (I_max - I_min) / (I_max + I_min)
    
    def set_phase_difference(self, phi_a: float, phi_b: float) -> None:
        """
        Set phase difference between two arms
        
        Args:
            phi_a: Phase in arm A
            phi_b: Phase in arm B
            
        Note:
            The effective phase is phi_a - phi_b
        """
        self.set_phase(phi_a - phi_b)
    
    def get_phase_difference(self) -> float:
        """
        Get the phase difference between arms
        
        Returns:
            float: Phase difference (phi_a - phi_b)
        """
        return self._phase
    
    def to_matrix(self) -> np.ndarray:
        """
        Get the transfer matrix of the interferometer
        
        Returns:
            np.ndarray: 2x2 transfer matrix
            
        Example:
            >>> mz = MachZehnderInterferometer()
            >>> mz.set_phase(np.pi)
            >>> M = mz.to_matrix()
            >>> print(M)
        """
        # Phase shift matrix
        P = np.array([[1, 0], [0, np.exp(1j * self._phase)]], dtype=complex)
        
        # Beam splitter matrices
        BS = 1 / SQRT2 * np.array([[1, 1j], [1j, 1]], dtype=complex)
        
        # Transfer matrix: U = BS @ P @ BS
        M = BS @ P @ BS
        
        return M


# ============================================================================
# ADDITIONAL INTERFERENCE TOOLS
# ============================================================================

def fringe_visibility(
    I_max: float,
    I_min: float
) -> float:
    """
    Calculate fringe visibility from intensities
    
    V = (I_max - I_min) / (I_max + I_min)
    
    Args:
        I_max: Maximum intensity
        I_min: Minimum intensity
        
    Returns:
        float: Visibility (0-1)
    """
    if I_max + I_min == 0:
        return 0.0
    return (I_max - I_min) / (I_max + I_min)


def optical_path_difference(
    path1: float,
    path2: float,
    refractive_index: float = 1.0
) -> float:
    """
    Calculate optical path difference (OPD)
    
    OPD = n * (path2 - path1)
    
    Args:
        path1: Length of path 1
        path2: Length of path 2
        refractive_index: Refractive index of medium
        
    Returns:
        float: Optical path difference
    """
    return refractive_index * (path2 - path1)


def phase_from_optical_path(
    optical_path_difference: float,
    wavelength: float
) -> float:
    """
    Calculate phase difference from optical path difference
    
    Δφ = 2π * OPD / λ
    
    Args:
        optical_path_difference: Optical path difference
        wavelength: Wavelength of light
        
    Returns:
        float: Phase difference in radians
    """
    return 2 * PI * optical_path_difference / wavelength


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Result class
    'InterferenceResult',
    
    # Double slit
    'DoubleSlit',
    
    # Mach-Zehnder
    'MachZehnderInterferometer',
    
    # Additional tools
    'fringe_visibility',
    'optical_path_difference',
    'phase_from_optical_path',
]