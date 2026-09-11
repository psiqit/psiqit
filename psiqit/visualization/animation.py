# psiqit/visualization/animation.py

"""
Animation Module
Animate wavefunctions and Bloch sphere states
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union, Callable
from ..quantum.state import Ket
from ..utils.logger import logger
from ..visualization.bloch import state_to_bloch, bloch_sphere


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


def _has_mpl_toolkits() -> bool:
    """Check if mpl_toolkits is available"""
    try:
        import mpl_toolkits
        return True
    except ImportError:
        return False


def _has_animation() -> bool:
    """Check if matplotlib animation is available"""
    try:
        from matplotlib import animation
        return True
    except ImportError:
        return False


# ============================================================================
# ANIMATE WAVEFUNCTION
# ============================================================================

def animate_wavefunction(
    psi_func: Callable[[np.ndarray, float], np.ndarray],
    x_range: Tuple[float, float],
    t_range: Tuple[float, float],
    n_x: int = 200,
    n_frames: int = 100,
    interval: int = 50,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    show_real_imag: bool = True,
    show_probability: bool = True,
    ylim: Optional[Tuple[float, float]] = None,
    title: str = "Wavefunction Animation"
):
    """
    Animate a time-dependent wavefunction
    
    Args:
        psi_func: Wavefunction function ψ(x, t) returning complex array
        x_range: (x_min, x_max) range
        t_range: (t_min, t_max) range
        n_x: Number of spatial points
        n_frames: Number of animation frames
        interval: Time between frames in milliseconds
        save_path: Path to save animation (if None, displays)
        figsize: Figure size (width, height)
        show_real_imag: Show real and imaginary parts
        show_probability: Show probability density
        ylim: Y-axis limits (if None, auto)
        title: Title of the animation
        
    Example:
        >>> import numpy as np
        >>> def psi_func(x, t):
        ...     # Gaussian wavepacket
        ...     sigma = 1.0
        ...     k0 = 1.0
        ...     x0 = 0.0
        ...     return np.exp(-(x - x0 - t)**2/(2*sigma**2)) * np.exp(1j*(k0*x - t))
        >>> animate_wavefunction(psi_func, (-5, 5), (0, 10), n_frames=100)
    """
    if not _has_matplotlib():
        logger.warning("matplotlib not available. Install with: pip install matplotlib")
        return
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        # Create spatial grid
        x = np.linspace(x_range[0], x_range[1], n_x)
        t = np.linspace(t_range[0], t_range[1], n_frames)
        
        # Compute wavefunctions for all frames
        psi_frames = []
        for ti in t:
            psi = psi_func(x, ti)
            psi_frames.append(psi)
        
        # Determine y-axis limits
        if ylim is None:
            max_abs = 0.0
            for psi in psi_frames:
                max_abs = max(max_abs, np.max(np.abs(psi)))
            ylim = (-max_abs * 1.2, max_abs * 1.2)
        
        # Create figure
        if show_real_imag and show_probability:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
            axes = [ax1, ax2]
        elif show_real_imag or show_probability:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            axes = [ax]
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            axes = [ax]
        
        # Initialize plots
        lines = []
        frames_text = None
        
        for i, ax in enumerate(axes):
            if show_real_imag and i == 0:
                # Real and imaginary parts
                psi0 = psi_frames[0]
                line_real, = ax.plot(x, np.real(psi0), 'b-', label='Re(ψ)', linewidth=2)
                line_imag, = ax.plot(x, np.imag(psi0), 'r-', label='Im(ψ)', linewidth=2)
                lines.append((line_real, line_imag))
                ax.set_ylabel('Amplitude')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(ylim)
            elif show_probability and (i == len(axes) - 1 or not show_real_imag):
                # Probability density
                psi0 = psi_frames[0]
                prob = np.abs(psi0)**2
                line_prob, = ax.plot(x, prob, 'g-', label='|ψ|²', linewidth=2)
                lines.append((line_prob,))
                ax.set_ylabel('Probability')
                ax.legend()
                ax.grid(True, alpha=0.3)
                if ylim is not None:
                    ax.set_ylim(0, ylim[1])
            else:
                # Fallback: just show real part
                psi0 = psi_frames[0]
                line_real, = ax.plot(x, np.real(psi0), 'b-', label='Re(ψ)', linewidth=2)
                lines.append((line_real,))
                ax.set_ylabel('Amplitude')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(ylim)
            
            ax.set_xlabel('x')
            ax.set_xlim(x_range)
        
        # Set title
        fig.suptitle(f"{title} - t = {t[0]:.3f}")
        
        # Add time display
        if len(axes) > 0:
            time_text = axes[-1].text(0.02, 0.95, f"t = {t[0]:.3f}", 
                                      transform=axes[-1].transAxes,
                                      fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            time_text = None
        
        plt.tight_layout()
        
        # Update function
        def update(frame):
            psi = psi_frames[frame]
            current_t = t[frame]
            
            idx = 0
            for ax in axes:
                if show_real_imag and idx == 0:
                    line_real, line_imag = lines[idx]
                    line_real.set_ydata(np.real(psi))
                    line_imag.set_ydata(np.imag(psi))
                    idx += 1
                elif show_probability and (idx == len(axes) - 1 or not show_real_imag):
                    line_prob, = lines[idx]
                    line_prob.set_ydata(np.abs(psi)**2)
                    idx += 1
                else:
                    line_real, = lines[idx]
                    line_real.set_ydata(np.real(psi))
                    idx += 1
            
            # Update time text
            if time_text is not None:
                time_text.set_text(f"t = {current_t:.3f}")
            
            fig.suptitle(f"{title} - t = {current_t:.3f}")
            
            return [line for sublist in lines for line in sublist] + ([time_text] if time_text else [])
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
        
        # Save or show
        if save_path is not None:
            try:
                anim.save(save_path, writer='pillow', fps=1000/interval)
                logger.info(f"Animation saved to {save_path}")
            except Exception as e:
                logger.warning(f"Could not save animation: {e}")
                # Fallback: try with ffmpeg
                try:
                    anim.save(save_path.replace('.gif', '.mp4'), writer='ffmpeg', fps=1000/interval)
                    logger.info(f"Animation saved to {save_path.replace('.gif', '.mp4')}")
                except:
                    logger.warning("Could not save animation with ffmpeg either")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not create wavefunction animation: {e}")
        # Fallback: show first frame
        try:
            import matplotlib.pyplot as plt
            x = np.linspace(x_range[0], x_range[1], n_x)
            psi = psi_func(x, t_range[0])
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            
            ax1.plot(x, np.real(psi), 'b-', label='Re(ψ)')
            ax1.plot(x, np.imag(psi), 'r-', label='Im(ψ)')
            ax1.set_ylabel('Amplitude')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(x, np.abs(psi)**2, 'g-', label='|ψ|²')
            ax2.set_xlabel('x')
            ax2.set_ylabel('Probability')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            fig.suptitle(f"{title} - t = {t_range[0]:.3f}")
            plt.tight_layout()
            plt.show()
            plt.close()
        except:
            pass


# ============================================================================
# ANIMATE BLOCH SPHERE
# ============================================================================

def animate_bloch_sphere(
    states: List[Union[Ket, np.ndarray, List, Tuple[float, float, float]]],
    interval: int = 200,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8),
    title: str = "Bloch Sphere Animation",
    color: str = 'purple',
    show_trace: bool = False,
    trace_length: Optional[int] = None
):
    """
    Animate states on the Bloch sphere
    
    Args:
        states: List of states to animate
        interval: Time between frames in milliseconds
        save_path: Path to save animation (if None, displays)
        figsize: Figure size
        title: Title of the animation
        color: Color of the state vector
        show_trace: Show trace of previous states
        trace_length: Number of previous states to show in trace
        
    Example:
        >>> from psiqit.quantum import zero, plus, minus
        >>> states = [zero(), plus(), minus(), zero()]
        >>> animate_bloch_sphere(states, interval=500)
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
        
        # Draw Bloch sphere
        from .bloch import _draw_sphere, _draw_axes
        _draw_sphere(ax, show_grid=True)
        _draw_axes(ax, show_labels=True)
        
        # Initialize vector
        x0, y0, z0 = coords[0]
        vec = ax.quiver(0, 0, 0, x0, y0, z0, color=color, arrow_length_ratio=0.15, 
                        linewidth=3, zorder=10)
        point = ax.scatter([x0], [y0], [z0], color=color, s=150, zorder=10)
        
        # Initialize trace
        trace_points = []
        trace_lines = []
        if show_trace:
            trace_length = trace_length or len(coords)
            for i in range(min(trace_length, len(coords))):
                x, y, z = coords[i]
                if i > 0:
                    px, py, pz = coords[i-1]
                    line, = ax.plot([px, x], [py, y], [pz, z], 
                                   color=color, alpha=0.3, linewidth=1)
                    trace_lines.append(line)
                point_trace = ax.scatter([x], [y], [z], color=color, 
                                        s=20, alpha=0.3, zorder=5)
                trace_points.append(point_trace)
        
        ax.set_title(f"{title} - Frame 1/{len(coords)}")
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
            
            # Update trace
            if show_trace:
                # Clear old trace
                for line in trace_lines:
                    line.remove()
                for point_t in trace_points:
                    point_t.remove()
                trace_lines.clear()
                trace_points.clear()
                
                # Draw new trace
                start = max(0, frame - trace_length + 1)
                for i in range(start, frame + 1):
                    cx, cy, cz = coords[i % len(coords)]
                    if i > start:
                        px, py, pz = coords[i-1 % len(coords)]
                        line, = ax.plot([px, cx], [py, cy], [pz, cz], 
                                       color=color, alpha=0.3, linewidth=1)
                        trace_lines.append(line)
                    point_t = ax.scatter([cx], [cy], [cz], color=color, 
                                        s=20, alpha=0.3, zorder=5)
                    trace_points.append(point_t)
            
            ax.set_title(f"{title} - Frame {frame+1}/{len(coords)}")
            
            return [new_vec, point] + trace_lines + trace_points
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(coords), interval=interval, blit=False)
        
        # Save or show
        if save_path is not None:
            try:
                anim.save(save_path, writer='pillow', fps=1000/interval)
                logger.info(f"Bloch sphere animation saved to {save_path}")
            except Exception as e:
                logger.warning(f"Could not save animation: {e}")
                # Fallback: try with ffmpeg
                try:
                    anim.save(save_path.replace('.gif', '.mp4'), writer='ffmpeg', fps=1000/interval)
                    logger.info(f"Bloch sphere animation saved to {save_path.replace('.gif', '.mp4')}")
                except:
                    logger.warning("Could not save animation with ffmpeg either")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not create Bloch sphere animation: {e}")
        # Fallback: show first state
        try:
            from .bloch import bloch_sphere
            bloch_sphere(states[0] if states else None, title=title)
        except:
            pass


# ============================================================================
# ADDITIONAL ANIMATION UTILITIES
# ============================================================================

def animate_parametric_curve(
    curve_func: Callable[[float], Tuple[float, float, float]],
    t_range: Tuple[float, float],
    n_frames: int = 100,
    interval: int = 50,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8),
    title: str = "Parametric Curve on Bloch Sphere",
    color: str = 'purple',
    show_trace: bool = True
):
    """
    Animate a parametric curve on the Bloch sphere
    
    Args:
        curve_func: Function f(t) -> (x, y, z) on the Bloch sphere
        t_range: (t_min, t_max) range
        n_frames: Number of animation frames
        interval: Time between frames in milliseconds
        save_path: Path to save animation
        figsize: Figure size
        title: Title of the animation
        color: Color of the curve
        show_trace: Show the full trace of the curve
        
    Example:
        >>> def curve(t):
        ...     # Circular path on equator
        ...     return (np.cos(t), np.sin(t), 0.0)
        >>> animate_parametric_curve(curve, (0, 2*np.pi), n_frames=50)
    """
    # Generate points
    t_values = np.linspace(t_range[0], t_range[1], n_frames)
    points = [curve_func(t) for t in t_values]
    
    # Animate using animate_bloch_sphere
    animate_bloch_sphere(
        states=points,
        interval=interval,
        save_path=save_path,
        figsize=figsize,
        title=title,
        color=color,
        show_trace=show_trace,
        trace_length=n_frames if show_trace else None
    )


def animate_quantum_circuit(
    circuit,
    shots: int = 100,
    interval: int = 200,
    save_path: Optional[str] = None
):
    """
    Animate measurement outcomes of a quantum circuit
    
    Args:
        circuit: QuantumCircuit object
        shots: Number of measurement shots
        interval: Time between frames in milliseconds
        save_path: Path to save animation
        
    Example:
        >>> from psiqit.circuits import QuantumCircuit
        >>> circ = QuantumCircuit(2)
        >>> circ.h(0).cx(0, 1)
        >>> animate_quantum_circuit(circ, shots=100)
    """
    if not _has_matplotlib():
        logger.warning("matplotlib not available. Install with: pip install matplotlib")
        return
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        # Get measurement results
        result = circuit.measure(shots=shots)
        counts = result['counts']
        all_states = list(counts.keys())
        all_counts = list(counts.values())
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Initialize bars
        bars = ax.bar(all_states, [0] * len(all_states), color='skyblue', edgecolor='black')
        
        ax.set_xlabel('State')
        ax.set_ylabel('Counts')
        ax.set_title('Quantum Circuit Measurements')
        ax.set_ylim(0, max(all_counts) * 1.1)
        
        # Total shots
        total = sum(all_counts)
        ax.text(0.95, 0.95, f'Total: {total}', transform=ax.transAxes,
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Update function
        def update(frame):
            # Simulate incremental measurements
            cumulative_counts = {state: 0 for state in all_states}
            for i in range(min(frame + 1, shots)):
                # Simulate a measurement
                state = np.random.choice(all_states, p=np.array(all_counts)/total)
                cumulative_counts[state] = cumulative_counts.get(state, 0) + 1
            
            # Update bars
            for bar, state in zip(bars, all_states):
                bar.set_height(cumulative_counts.get(state, 0))
            
            ax.set_title(f'Quantum Circuit Measurements - Shot {min(frame+1, shots)}/{shots}')
            
            return bars
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=shots, interval=interval, blit=False)
        
        # Save or show
        if save_path is not None:
            anim.save(save_path, writer='pillow', fps=1000/interval)
            logger.info(f"Animation saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not create circuit animation: {e}")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'animate_wavefunction',
    'animate_bloch_sphere',
    'animate_parametric_curve',
    'animate_quantum_circuit',
]