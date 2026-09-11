# psiqit/visualization/bloch.py

"""
Bloch Sphere Visualization Module
Visualize qubit states on the Bloch sphere
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from ..quantum.state import Ket, zero, one, plus, minus
from ..quantum.operator import Operator
from ..utils.logger import logger
from ..utils.conversion import to_bloch_coordinates, from_bloch_coordinates


# ============================================================================
# DEPENDENCY CHECK
# ============================================================================

def _has_matplotlib() -> bool:
    """Check if matplotlib is available"""
    try:
        import matplotlib
        return True
    except ImportError:
        return False


def _has_mpl_toolkits() -> bool:
    """Check if mpl_toolkits is available"""
    try:
        import mpl_toolkits
        return True
    except ImportError:
        return False


# ============================================================================
# COORDINATE CONVERSIONS
# ============================================================================

def state_to_bloch(state: Union[Ket, np.ndarray, List]) -> Tuple[float, float, float]:
    """
    Convert a single qubit state to Bloch sphere coordinates
    
    Args:
        state: Single qubit state (Ket, list, or numpy array)
        
    Returns:
        Tuple[float, float, float]: (x, y, z) coordinates
        
    Example:
        >>> from psiqit.quantum import zero, plus
        >>> x, y, z = state_to_bloch(zero())
        >>> print(x, y, z)  # 0.0, 0.0, 1.0
        >>> x, y, z = state_to_bloch(plus())
        >>> print(x, y, z)  # 1.0, 0.0, 0.0
    """
    return to_bloch_coordinates(state)


def bloch_to_state(x: float, y: float, z: float) -> Ket:
    """
    Create a single qubit state from Bloch sphere coordinates
    
    Args:
        x: x-coordinate
        y: y-coordinate
        z: z-coordinate
        
    Returns:
        Ket: Single qubit state
        
    Example:
        >>> state = bloch_to_state(0, 0, 1)  # |0⟩
        >>> state = bloch_to_state(1, 0, 0)  # |+⟩
    """
    return from_bloch_coordinates(x, y, z)


def spherical_to_bloch(theta: float, phi: float) -> Tuple[float, float, float]:
    """
    Convert spherical coordinates to Bloch sphere coordinates
    
    Args:
        theta: Polar angle (0 to π)
        phi: Azimuthal angle (0 to 2π)
        
    Returns:
        Tuple[float, float, float]: (x, y, z) coordinates
        
    Example:
        >>> x, y, z = spherical_to_bloch(np.pi/2, 0)  # |+⟩
        >>> print(x, y, z)  # 1.0, 0.0, 0.0
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return (float(x), float(y), float(z))


def bloch_to_spherical(x: float, y: float, z: float) -> Tuple[float, float]:
    """
    Convert Bloch sphere coordinates to spherical coordinates
    
    Args:
        x: x-coordinate
        y: y-coordinate
        z: z-coordinate
        
    Returns:
        Tuple[float, float]: (theta, phi) angles
        
    Example:
        >>> theta, phi = bloch_to_spherical(1, 0, 0)  # |+⟩
        >>> print(theta, phi)  # π/2, 0
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    if r < 1e-10:
        return (0.0, 0.0)
    
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    return (float(theta), float(phi))


# ============================================================================
# BLOCH SPHERE PLOTTING
# ============================================================================

def bloch_sphere(
    state: Optional[Union[Ket, np.ndarray, List]] = None,
    points: Optional[List[Tuple[float, float, float]]] = None,
    title: str = "Bloch Sphere",
    figsize: Tuple[float, float] = (8, 8),
    show_axes: bool = True,
    show_labels: bool = True,
    show_grid: bool = True,
    save_path: Optional[str] = None,
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    labels: Optional[List[str]] = None
):
    """
    Plot a Bloch sphere with a state vector and/or points
    
    Args:
        state: Single qubit state to display as a vector
        points: List of (x, y, z) points to display on the sphere
        title: Title of the plot
        figsize: Figure size (width, height)
        show_axes: Show x, y, z axes
        show_labels: Show axis labels
        show_grid: Show grid on the sphere
        save_path: Path to save the figure (if None, displays)
        colors: List of colors for points
        markers: List of markers for points
        labels: List of labels for points
        
    Example:
        >>> from psiqit.quantum import zero, plus, minus
        >>> bloch_sphere(zero(), title="|0⟩ State")
        >>> bloch_sphere(points=[(1,0,0), (0,1,0), (0,0,1)], 
        ...              labels=['X', 'Y', 'Z'])
    """
    if not _has_matplotlib():
        logger.warning("matplotlib not available. Install with: pip install matplotlib")
        return
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Bloch sphere
    _draw_sphere(ax, show_grid)
    
    # Draw axes
    if show_axes:
        _draw_axes(ax, show_labels)
    
    # Draw state vector if provided
    if state is not None:
        x, y, z = state_to_bloch(state)
        _draw_vector(ax, x, y, z, color='purple', label='State')
        
        # Draw point
        ax.scatter([x], [y], [z], color='purple', s=100, zorder=10)
    
    # Draw points if provided
    if points is not None:
        if colors is None:
            colors = ['red'] * len(points)
        if markers is None:
            markers = ['o'] * len(points)
        
        for i, (x, y, z) in enumerate(points):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            label = labels[i] if labels and i < len(labels) else None
            
            # Check if point is on the sphere
            r = np.sqrt(x**2 + y**2 + z**2)
            if abs(r - 1.0) > 0.01:
                logger.warning(f"Point ({x:.3f}, {y:.3f}, {z:.3f}) is not on the Bloch sphere (r={r:.3f})")
            
            ax.scatter([x], [y], [z], color=color, marker=marker, 
                      s=80, zorder=10, label=label)
    
    # Set title and limits
    ax.set_title(title)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1, 1, 1])
    
    # Legend
    if labels:
        ax.legend()
    
    plt.tight_layout()
    
    # Save or show
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Bloch sphere saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def _draw_sphere(ax, show_grid: bool = True):
    """
    Draw the Bloch sphere
    
    Args:
        ax: Matplotlib 3D axis
        show_grid: Show grid lines
    """
    # Draw sphere surface
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.15, edgecolor='none')
    
    # Draw equator
    theta = np.linspace(0, 2 * np.pi, 100)
    x_eq = np.cos(theta)
    y_eq = np.sin(theta)
    z_eq = np.zeros_like(theta)
    ax.plot(x_eq, y_eq, z_eq, color='gray', alpha=0.5, linewidth=1)
    
    # Draw meridians
    if show_grid:
        for phi in np.linspace(0, 2 * np.pi, 8):
            x_mer = np.sin(v) * np.cos(phi)
            y_mer = np.sin(v) * np.sin(phi)
            z_mer = np.cos(v)
            ax.plot(x_mer, y_mer, z_mer, color='gray', alpha=0.3, linewidth=0.5)
        
        for theta_mer in np.linspace(0, np.pi, 6):
            x_mer = np.sin(theta_mer) * np.cos(u)
            y_mer = np.sin(theta_mer) * np.sin(u)
            z_mer = np.cos(theta_mer) * np.ones_like(u)
            ax.plot(x_mer, y_mer, z_mer, color='gray', alpha=0.3, linewidth=0.5)


def _draw_axes(ax, show_labels: bool = True):
    """
    Draw Bloch sphere axes
    
    Args:
        ax: Matplotlib 3D axis
        show_labels: Show axis labels
    """
    # Axis length
    L = 1.2
    
    # X-axis (red)
    ax.quiver(0, 0, 0, L, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2)
    if show_labels:
        ax.text(L * 1.1, 0, 0, 'X', color='red', fontsize=12)
    
    # Y-axis (green)
    ax.quiver(0, 0, 0, 0, L, 0, color='green', arrow_length_ratio=0.1, linewidth=2)
    if show_labels:
        ax.text(0, L * 1.1, 0, 'Y', color='green', fontsize=12)
    
    # Z-axis (blue)
    ax.quiver(0, 0, 0, 0, 0, L, color='blue', arrow_length_ratio=0.1, linewidth=2)
    if show_labels:
        ax.text(0, 0, L * 1.1, 'Z', color='blue', fontsize=12)
    
    # Negative axes (dashed)
    ax.plot([-L, 0], [0, 0], [0, 0], color='red', alpha=0.3, linestyle='--')
    ax.plot([0, 0], [-L, 0], [0, 0], color='green', alpha=0.3, linestyle='--')
    ax.plot([0, 0], [0, 0], [-L, 0], color='blue', alpha=0.3, linestyle='--')


def _draw_vector(ax, x: float, y: float, z: float, color: str = 'purple', label: str = None):
    """
    Draw a state vector on the Bloch sphere
    
    Args:
        ax: Matplotlib 3D axis
        x: x-coordinate
        y: y-coordinate
        z: z-coordinate
        color: Color of the vector
        label: Label for the vector
    """
    # Normalize if needed
    r = np.sqrt(x**2 + y**2 + z**2)
    if r > 1e-10:
        x, y, z = x/r, y/r, z/r
    
    # Draw vector
    ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.15, 
              linewidth=3, zorder=10, label=label)


# ============================================================================
# MULTIPLE STATES PLOTTING
# ============================================================================

def plot_multiple_states(
    states: List[Union[Ket, np.ndarray, List, Tuple[float, float, float]]],
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    title: str = "Multiple States on Bloch Sphere",
    figsize: Tuple[float, float] = (8, 8),
    show_vectors: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot multiple states on the Bloch sphere
    
    Args:
        states: List of states (Ket, coordinates, or state vectors)
        colors: List of colors for each state
        labels: List of labels for each state
        title: Title of the plot
        figsize: Figure size
        show_vectors: Show state vectors
        save_path: Path to save the figure
        
    Example:
        >>> from psiqit.quantum import zero, plus, minus
        >>> states = [zero(), plus(), minus()]
        >>> labels = ['|0⟩', '|+⟩', '|-⟩']
        >>> plot_multiple_states(states, labels=labels)
    """
    if not _has_matplotlib():
        logger.warning("matplotlib not available. Install with: pip install matplotlib")
        return
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Convert states to coordinates
    coords = []
    for state in states:
        if isinstance(state, (list, tuple)) and len(state) == 3:
            # Already coordinates
            coords.append(state)
        else:
            # Convert to Bloch coordinates
            x, y, z = state_to_bloch(state)
            coords.append((x, y, z))
    
    # Default colors
    if colors is None:
        default_colors = ['purple', 'red', 'blue', 'green', 'orange', 'cyan', 'magenta', 'brown']
        colors = default_colors[:len(coords)]
    else:
        colors = colors[:len(coords)]
    
    if labels is None:
        labels = [f"State {i}" for i in range(len(coords))]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Bloch sphere
    _draw_sphere(ax, show_grid=True)
    _draw_axes(ax, show_labels=True)
    
    # Draw states
    for i, (x, y, z) in enumerate(coords):
        color = colors[i % len(colors)]
        label = labels[i] if i < len(labels) else None
        
        if show_vectors:
            _draw_vector(ax, x, y, z, color=color, label=label)
        
        # Draw point
        ax.scatter([x], [y], [z], color=color, s=100, zorder=10)
    
    ax.set_title(title)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1, 1, 1])
    
    if labels:
        ax.legend()
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Bloch sphere plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# BLOCH SPHERE ANIMATION
# ============================================================================

def animate_bloch(
    states: List[Union[Ket, np.ndarray, List, Tuple[float, float, float]]],
    interval: int = 200,
    filename: Optional[str] = None,
    title: str = "Bloch Sphere Animation",
    figsize: Tuple[float, float] = (6, 6),
    color: str = 'purple'
):
    """
    Create an animation of states on the Bloch sphere
    
    Args:
        states: List of states to animate
        interval: Time between frames in milliseconds
        filename: Path to save the animation (if None, displays)
        title: Title of the animation
        figsize: Figure size
        color: Color of the state vector
        
    Example:
        >>> from psiqit.quantum import zero, plus, minus
        >>> states = [zero(), plus(), minus(), zero()]
        >>> animate_bloch(states, interval=500)
    """
    if not _has_matplotlib():
        logger.warning("matplotlib not available. Install with: pip install matplotlib")
        return
    
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.animation import FuncAnimation
        
        # Convert states to coordinates
        coords = []
        for state in states:
            if isinstance(state, (list, tuple)) and len(state) == 3:
                coords.append(state)
            else:
                x, y, z = state_to_bloch(state)
                coords.append((x, y, z))
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw static elements
        _draw_sphere(ax, show_grid=True)
        _draw_axes(ax, show_labels=True)
        
        # Initialize vector
        x0, y0, z0 = coords[0]
        vec = ax.quiver(0, 0, 0, x0, y0, z0, color=color, arrow_length_ratio=0.15, 
                        linewidth=3, zorder=10)
        point = ax.scatter([x0], [y0], [z0], color=color, s=100, zorder=10)
        
        ax.set_title(title)
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_box_aspect([1, 1, 1])
        
        # Update function
        def update(frame):
            x, y, z = coords[frame % len(coords)]
            
            # Update vector
            vec.remove()
            new_vec = ax.quiver(0, 0, 0, x, y, z, color=color, 
                                arrow_length_ratio=0.15, linewidth=3, zorder=10)
            
            # Update point
            point._offsets3d = ([x], [y], [z])
            
            # Update title with frame number
            ax.set_title(f"{title} - Frame {frame+1}/{len(coords)}")
            
            return new_vec, point
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(coords), interval=interval, blit=False)
        
        # Save or show
        if filename is not None:
            anim.save(filename, writer='pillow', fps=1000/interval)
            logger.info(f"Animation saved to {filename}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not create animation: {e}")
        # Fallback: plot the first state
        bloch_sphere(states[0], title=title)


# ============================================================================
# ADDITIONAL UTILITIES
# ============================================================================

def get_bloch_coordinates(state: Union[Ket, np.ndarray, List]) -> Dict[str, float]:
    """
    Get Bloch sphere coordinates with additional information
    
    Args:
        state: Single qubit state
        
    Returns:
        Dict: Bloch coordinates and angles
        
    Example:
        >>> from psiqit.quantum import plus
        >>> info = get_bloch_coordinates(plus())
        >>> print(info['x'])  # 1.0
        >>> print(info['theta'])  # π/2
    """
    x, y, z = state_to_bloch(state)
    theta, phi = bloch_to_spherical(x, y, z)
    
    return {
        'x': x,
        'y': y,
        'z': z,
        'theta': theta,
        'phi': phi,
        'r': np.sqrt(x**2 + y**2 + z**2)
    }


def bloch_distance(
    state1: Union[Ket, np.ndarray, List, Tuple[float, float, float]],
    state2: Union[Ket, np.ndarray, List, Tuple[float, float, float]]
) -> float:
    """
    Compute the Euclidean distance between two states on the Bloch sphere
    
    Args:
        state1: First state
        state2: Second state
        
    Returns:
        float: Distance between the states
        
    Example:
        >>> from psiqit.quantum import zero, one
        >>> dist = bloch_distance(zero(), one())  # 2.0
    """
    # Convert to coordinates
    if isinstance(state1, (list, tuple)) and len(state1) == 3:
        x1, y1, z1 = state1
    else:
        x1, y1, z1 = state_to_bloch(state1)
    
    if isinstance(state2, (list, tuple)) and len(state2) == 3:
        x2, y2, z2 = state2
    else:
        x2, y2, z2 = state_to_bloch(state2)
    
    return float(np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2))


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Coordinate conversions
    'state_to_bloch',
    'bloch_to_state',
    'spherical_to_bloch',
    'bloch_to_spherical',
    
    # Plotting
    'bloch_sphere',
    'plot_multiple_states',
    'animate_bloch',
    
    # Additional utilities
    'get_bloch_coordinates',
    'bloch_distance',
]
