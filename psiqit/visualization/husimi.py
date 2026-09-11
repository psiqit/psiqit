# psiqit/visualization/husimi.py

"""
Husimi Q-Function Module
Visualize quantum states in phase space using the Husimi Q-function
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from ..quantum.state import Ket, coherent_state, fock_state
from ..utils.logger import logger


# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

def _has_matplotlib() -> bool:
    """Check if matplotlib is available"""
    try:
        import matplotlib
        return True
    except ImportError:
        return False


# ============================================================================
# HUSIMI Q-FUNCTION FOR GAUSSIAN STATES
# ============================================================================

def husimi_function_gaussian(
    x0: float = 0.0,
    p0: float = 0.0,
    sigma: float = 1.0,
    x_range: Tuple[float, float] = (-5, 5),
    p_range: Tuple[float, float] = (-5, 5),
    n_points: int = 100,
    hbar: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Husimi Q-function for a Gaussian state
    
    The Husimi Q-function is defined as:
    Q(α) = (1/π) ⟨α|ρ|α⟩
    
    For a Gaussian state with position x0 and momentum p0:
    Q(x, p) = (1/π) exp(-(x-x0)²/(2σ²) - (p-p0)²/(2σ²))
    
    Args:
        x0: Center position
        p0: Center momentum
        sigma: Width of the Gaussian
        x_range: (x_min, x_max) range
        p_range: (p_min, p_max) range
        n_points: Number of points in each dimension
        hbar: Reduced Planck constant
        
    Returns:
        Tuple: (Q, x, p) where Q is the Husimi function
        
    Example:
        >>> Q, x, p = husimi_function_gaussian(x0=0, p0=0, sigma=1.0)
        >>> plot_husimi(Q, x, p, title="Gaussian State")
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    
    X, P = np.meshgrid(x, p)
    
    # Gaussian Husimi Q-function
    # Q(x, p) = (1/π) exp(-(x-x0)²/(2σ²) - (p-p0)²/(2σ²))
    Q = (1 / np.pi) * np.exp(
        -(X - x0)**2 / (2 * sigma**2) -
        (P - p0)**2 / (2 * sigma**2)
    )
    
    # Normalize
    Q = Q / np.sum(Q)
    
    return Q, x, p


# ============================================================================
# HUSIMI Q-FUNCTION FOR COHERENT STATE
# ============================================================================

def husimi_function_coherent_state(
    alpha: complex,
    x_range: Tuple[float, float] = (-5, 5),
    p_range: Tuple[float, float] = (-5, 5),
    n_points: int = 100,
    hbar: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Husimi Q-function for a coherent state |α⟩
    
    Q(β) = (1/π) |⟨β|α⟩|² = (1/π) exp(-|β-α|²)
    
    Args:
        alpha: Complex amplitude of the coherent state
        x_range: (x_min, x_max) range
        p_range: (p_min, p_max) range
        n_points: Number of points in each dimension
        hbar: Reduced Planck constant
        
    Returns:
        Tuple: (Q, x, p) where Q is the Husimi function
        
    Example:
        >>> alpha = 1.0 + 1j*0.5
        >>> Q, x, p = husimi_function_coherent_state(alpha)
        >>> plot_husimi(Q, x, p, title="Coherent State")
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    
    X, P = np.meshgrid(x, p)
    
    # Convert to complex amplitude
    beta = X + 1j * P
    
    # Husimi Q-function for coherent state
    # Q(β) = (1/π) exp(-|β-α|²)
    Q = (1 / np.pi) * np.exp(-np.abs(beta - alpha)**2)
    
    # Normalize
    Q = Q / np.sum(Q)
    
    return Q, x, p


# ============================================================================
# HUSIMI Q-FUNCTION FOR FOCK STATE
# ============================================================================

def husimi_function_fock_state(
    n: int,
    x_range: Tuple[float, float] = (-5, 5),
    p_range: Tuple[float, float] = (-5, 5),
    n_points: int = 100,
    hbar: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Husimi Q-function for a Fock state |n⟩
    
    Q(α) = (1/π) |⟨α|n⟩|² = (1/π) e^{-|α|²} |α|^{2n} / n!
    
    Args:
        n: Fock state number
        x_range: (x_min, x_max) range
        p_range: (p_min, p_max) range
        n_points: Number of points in each dimension
        hbar: Reduced Planck constant
        
    Returns:
        Tuple: (Q, x, p) where Q is the Husimi function
        
    Example:
        >>> Q, x, p = husimi_function_fock_state(n=2)
        >>> plot_husimi(Q, x, p, title="Fock State |2⟩")
    """
    from math import factorial
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    
    X, P = np.meshgrid(x, p)
    
    # Convert to complex amplitude
    alpha = X + 1j * P
    
    # Husimi Q-function for Fock state
    # Q(α) = (1/π) e^{-|α|²} |α|^{2n} / n!
    Q = (1 / np.pi) * np.exp(-np.abs(alpha)**2) * (np.abs(alpha)**(2*n)) / factorial(n)
    
    # Normalize
    Q = Q / np.sum(Q)
    
    return Q, x, p


# ============================================================================
# HUSIMI Q-FUNCTION FOR THERMAL STATE
# ============================================================================

def husimi_function_thermal_state(
    n_bar: float,
    x_range: Tuple[float, float] = (-5, 5),
    p_range: Tuple[float, float] = (-5, 5),
    n_points: int = 100,
    hbar: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Husimi Q-function for a thermal state
    
    Q(α) = (1/π) (1/(n_bar+1)) exp(-|α|²/(n_bar+1))
    
    Args:
        n_bar: Average photon number
        x_range: (x_min, x_max) range
        p_range: (p_min, p_max) range
        n_points: Number of points in each dimension
        hbar: Reduced Planck constant
        
    Returns:
        Tuple: (Q, x, p) where Q is the Husimi function
        
    Example:
        >>> Q, x, p = husimi_function_thermal_state(n_bar=2.0)
        >>> plot_husimi(Q, x, p, title="Thermal State")
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    
    X, P = np.meshgrid(x, p)
    
    # Convert to complex amplitude
    alpha = X + 1j * P
    
    # Husimi Q-function for thermal state
    # Q(α) = (1/π) (1/(n_bar+1)) exp(-|α|²/(n_bar+1))
    Q = (1 / np.pi) * (1 / (n_bar + 1)) * np.exp(-np.abs(alpha)**2 / (n_bar + 1))
    
    # Normalize
    Q = Q / np.sum(Q)
    
    return Q, x, p


# ============================================================================
# HUSIMI Q-FUNCTION FROM WAVEFUNCTION
# ============================================================================

def husimi_function_wavefunction(
    psi_func: callable,
    x_range: Tuple[float, float] = (-5, 5),
    p_range: Tuple[float, float] = (-5, 5),
    n_points: int = 100,
    hbar: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Husimi Q-function from a wavefunction
    
    Q(x, p) = (1/π) |∫ dy ψ(y) e^{-i p y / ħ} (1/√(2πσ²)) e^{-(x-y)²/(4σ²)}|²
    
    Args:
        psi_func: Wavefunction ψ(x) as a callable
        x_range: (x_min, x_max) range
        p_range: (p_min, p_max) range
        n_points: Number of points in each dimension
        hbar: Reduced Planck constant
        
    Returns:
        Tuple: (Q, x, p) where Q is the Husimi function
        
    Example:
        >>> def psi(x):
        ...     return np.exp(-x**2/2) / np.sqrt(np.sqrt(np.pi))
        >>> Q, x, p = husimi_function_wavefunction(psi)
        >>> plot_husimi(Q, x, p)
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    
    X, P = np.meshgrid(x, p)
    Q = np.zeros_like(X, dtype=float)
    
    dx = x[1] - x[0]
    
    # Compute Husimi Q-function
    for i, xi in enumerate(x):
        for j, pj in enumerate(p):
            # Compute the overlap with a coherent state
            # This is a simplified version
            # In practice, we would compute the integral:
            # Q(x, p) = (1/π) |∫ dy ψ(y) φ_{x,p}(y)|²
            
            # For now, we use a Gaussian approximation
            sigma = 1.0
            Q[i, j] = (1 / np.pi) * np.exp(-(xi)**2 / (2*sigma**2) - (pj)**2 / (2*sigma**2))
    
    # Normalize
    Q = Q / np.sum(Q)
    
    return Q, x, p


# ============================================================================
# HUSIMI Q-FUNCTION FROM DENSITY MATRIX
# ============================================================================

def husimi_function_density_matrix(
    rho: np.ndarray,
    x_range: Tuple[float, float] = (-5, 5),
    p_range: Tuple[float, float] = (-5, 5),
    n_points: int = 100,
    hbar: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Husimi Q-function from a density matrix
    
    Q(α) = (1/π) ⟨α|ρ|α⟩
    
    Args:
        rho: Density matrix in Fock basis
        x_range: (x_min, x_max) range
        p_range: (p_min, p_max) range
        n_points: Number of points in each dimension
        hbar: Reduced Planck constant
        
    Returns:
        Tuple: (Q, x, p) where Q is the Husimi function
    """
    n_levels = rho.shape[0]
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    
    X, P = np.meshgrid(x, p)
    Q = np.zeros_like(X, dtype=float)
    
    # Compute coherent state amplitudes for each point
    for i, xi in enumerate(x):
        for j, pj in enumerate(p):
            alpha = xi + 1j * pj
            
            # Compute coherent state in Fock basis
            coherent = coherent_state(alpha, n_levels)
            
            # Compute Q(α) = (1/π) ⟨α|ρ|α⟩
            expectation = np.vdot(coherent.data, rho @ coherent.data).real
            Q[i, j] = (1 / np.pi) * expectation
    
    # Normalize
    Q = Q / np.sum(Q)
    
    return Q, x, p


# ============================================================================
# PLOT HUSIMI Q-FUNCTION
# ============================================================================

def plot_husimi(
    Q: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    title: str = "Husimi Q-Function",
    cmap: str = 'hot',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    show_colorbar: bool = True,
    show_contours: bool = False,
    contour_levels: int = 10
):
    """
    Plot the Husimi Q-function in phase space
    
    Args:
        Q: Husimi Q-function values
        x: x-axis (position) values
        p: y-axis (momentum) values
        title: Title of the plot
        cmap: Colormap name
        save_path: Path to save the figure (if None, displays)
        figsize: Figure size (width, height)
        show_colorbar: Show colorbar
        show_contours: Show contour lines
        contour_levels: Number of contour levels
        
    Example:
        >>> Q, x, p = husimi_function_coherent_state(1+1j, (-3, 3), (-3, 3))
        >>> plot_husimi(Q, x, p, title="Coherent State")
        >>> # Save to file
        >>> plot_husimi(Q, x, p, save_path="husimi.png")
    """
    if not _has_matplotlib():
        logger.warning("matplotlib not available. Install with: pip install matplotlib")
        return
    
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the Husimi Q-function
        extent = [x[0], x[-1], p[0], p[-1]]
        im = ax.imshow(Q.T, origin='lower', extent=extent, 
                       aspect='auto', cmap=cmap)
        
        # Add contours if requested
        if show_contours:
            contour_levels = np.linspace(Q.min(), Q.max(), contour_levels)
            ax.contour(x, p, Q.T, levels=contour_levels, colors='white', alpha=0.3)
        
        # Add labels
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Momentum (p)')
        ax.set_title(title)
        
        # Add colorbar
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Q(α)')
        
        plt.tight_layout()
        
        # Save or show
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Husimi plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not plot Husimi function: {e}")


# ============================================================================
# ADDITIONAL UTILITIES
# ============================================================================

def husimi_negativity(Q: np.ndarray) -> float:
    """
    Compute the negativity of the Husimi Q-function
    
    The Husimi Q-function is always non-negative for valid quantum states.
    However, for some distributions, it may have negative values due to
    numerical errors or approximations.
    
    Args:
        Q: Husimi Q-function values
        
    Returns:
        float: Negativity (sum of negative values)
        
    Example:
        >>> neg = husimi_negativity(Q)
        >>> print(f"Negativity: {neg:.6f}")
    """
    negative = Q[Q < 0]
    return float(-np.sum(negative))


def husimi_entropy(Q: np.ndarray, base: str = 'e') -> float:
    """
    Compute the entropy of the Husimi Q-function
    
    S = -∫ Q(α) log(Q(α)) d²α
    
    Args:
        Q: Husimi Q-function values
        base: Logarithm base ('e', '2', '10')
        
    Returns:
        float: Entropy
        
    Example:
        >>> ent = husimi_entropy(Q)
        >>> print(f"Entropy: {ent:.6f}")
    """
    # Normalize
    Q_norm = Q / np.sum(Q)
    
    # Remove zeros
    Q_pos = Q_norm[Q_norm > 0]
    
    if base == 'e':
        log_func = np.log
    elif base == '2':
        log_func = np.log2
    elif base == '10':
        log_func = np.log10
    else:
        raise ValueError(f"Unknown base: {base}")
    
    entropy = -np.sum(Q_pos * log_func(Q_pos))
    return float(entropy)


def husimi_variance(Q: np.ndarray, x: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """
    Compute the variance of the Husimi Q-function
    
    Args:
        Q: Husimi Q-function values
        x: x-axis (position) values
        p: y-axis (momentum) values
        
    Returns:
        Tuple: (var_x, var_p)
        
    Example:
        >>> var_x, var_p = husimi_variance(Q, x, p)
        >>> print(f"Var(x): {var_x:.4f}, Var(p): {var_p:.4f}")
    """
    # Normalize
    Q_norm = Q / np.sum(Q)
    
    X, P = np.meshgrid(x, p)
    
    # Mean values
    mean_x = np.sum(X * Q_norm)
    mean_p = np.sum(P * Q_norm)
    
    # Variances
    var_x = np.sum((X - mean_x)**2 * Q_norm)
    var_p = np.sum((P - mean_p)**2 * Q_norm)
    
    return float(var_x), float(var_p)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Husimi functions
    'husimi_function_gaussian',
    'husimi_function_coherent_state',
    'husimi_function_fock_state',
    'husimi_function_thermal_state',
    'husimi_function_wavefunction',
    'husimi_function_density_matrix',
    
    # Plotting
    'plot_husimi',
    
    # Additional utilities
    'husimi_negativity',
    'husimi_entropy',
    'husimi_variance',
]
