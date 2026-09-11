# psiqit/visualization/circuit_drawer.py

"""
Circuit Drawer Module
Draw quantum circuits in ASCII and Unicode formats
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple
from ..circuits.circuit import QuantumCircuit
from ..utils.logger import logger


# ============================================================================
# DRAW CIRCUIT
# ============================================================================

def draw_circuit(
    circuit: QuantumCircuit,
    style: str = 'ascii',
    show_wires: bool = True,
    qubit_labels: Optional[List[str]] = None
) -> str:
    """
    Draw a quantum circuit in various styles
    
    Args:
        circuit: QuantumCircuit object
        style: 'ascii', 'unicode', or 'text'
        show_wires: Show wire lines
        qubit_labels: Custom labels for qubits
        
    Returns:
        str: Circuit diagram
        
    Example:
        >>> from psiqit.circuits import QuantumCircuit
        >>> circ = QuantumCircuit(2)
        >>> circ.h(0).cx(0, 1)
        >>> print(draw_circuit(circ))
        q0: ── H ── ● ──
        q1: ──────── X ──
    """
    if style == 'ascii':
        return _draw_ascii(circuit, show_wires, qubit_labels)
    elif style == 'unicode':
        return _draw_unicode(circuit, show_wires, qubit_labels)
    elif style == 'text':
        return circuit_to_text(circuit, show_wires)
    else:
        raise ValueError(f"Unknown style: {style}. Use 'ascii', 'unicode', or 'text'")


# ============================================================================
# ASCII DRAWER
# ============================================================================

def _draw_ascii(
    circuit: QuantumCircuit,
    show_wires: bool = True,
    qubit_labels: Optional[List[str]] = None
) -> str:
    """
    Draw circuit in ASCII format
    
    Args:
        circuit: QuantumCircuit object
        show_wires: Show wire lines
        qubit_labels: Custom labels for qubits
        
    Returns:
        str: ASCII circuit diagram
    """
    n_qubits = circuit.n_qubits
    gates = circuit.get_gates()
    
    # Default qubit labels
    if qubit_labels is None:
        qubit_labels = [f"q{i}" for i in range(n_qubits)]
    else:
        qubit_labels = qubit_labels[:n_qubits]
    
    # Initialize lines
    lines = ["" for _ in range(n_qubits)]
    
    # Add qubit labels
    max_label_len = max(len(label) for label in qubit_labels)
    for i in range(n_qubits):
        lines[i] = f"{qubit_labels[i]:>{max_label_len}} : "
    
    # Gate symbols
    gate_symbols = {
        'H': 'H',
        'X': 'X',
        'Y': 'Y',
        'Z': 'Z',
        'S': 'S',
        'T': 'T',
        'CNOT': '●--X',
        'CZ': '●--Z',
        'SWAP': 'X--X',
        'TOFFOLI': '●--●--X',
        'RX': 'Rx',
        'RY': 'Ry',
        'RZ': 'Rz',
        'CX': '●--X',
        'CU1': '●--P',
    }
    
    # Process gates
    for gate in gates:
        name = gate['name']
        qubits = gate['qubits']
        params = gate.get('params', [])
        
        # Determine gate width
        if name in ['H', 'X', 'Y', 'Z', 'S', 'T']:
            width = 3  # " H "
            symbol = f" {name} "
        elif name in ['RX', 'RY', 'RZ']:
            theta = params[0] if params else 0
            label = f"{name}({theta:.2f})"
            width = len(label) + 2
            symbol = f" {label} "
        elif name == 'CNOT' or name == 'CX':
            width = 5  # " ●--X "
            symbol = " ●--X "
        elif name == 'CZ':
            width = 5  # " ●--Z "
            symbol = " ●--Z "
        elif name == 'SWAP':
            width = 5  # " X--X "
            symbol = " X--X "
        elif name == 'TOFFOLI':
            width = 7  # " ●--●--X "
            symbol = " ●--●--X "
        elif name == 'CU1':
            theta = params[0] if params else 0
            label = f"P({theta:.2f})"
            width = len(label) + 4
            symbol = f" ●--{label} "
        else:
            label = name[:4]
            width = len(label) + 2
            symbol = f" {label} "
        
        # Apply gate to qubits
        for i in range(n_qubits):
            if i in qubits:
                # Add gate symbol
                if len(qubits) == 1:
                    lines[i] += symbol
                else:
                    # Multi-qubit gate
                    idx = qubits.index(i)
                    if idx == 0:
                        # Control qubit
                        lines[i] += " ● "
                    elif idx == len(qubits) - 1:
                        # Target qubit
                        lines[i] += " X "
                    else:
                        # Middle control
                        lines[i] += " ● "
                    
                    # Add wire connectors
                    if idx < len(qubits) - 1:
                        lines[i] += "--"
                    else:
                        lines[i] += "  "
            else:
                # Wire
                lines[i] += "── " * (width // 3 + 1) if show_wires else "   "
    
    # Clean up lines
    for i in range(n_qubits):
        lines[i] = lines[i].rstrip()
    
    return "\n".join(lines)


# ============================================================================
# UNICODE DRAWER
# ============================================================================

def _draw_unicode(
    circuit: QuantumCircuit,
    show_wires: bool = True,
    qubit_labels: Optional[List[str]] = None
) -> str:
    """
    Draw circuit in Unicode format (more visual)
    
    Args:
        circuit: QuantumCircuit object
        show_wires: Show wire lines
        qubit_labels: Custom labels for qubits
        
    Returns:
        str: Unicode circuit diagram
    """
    n_qubits = circuit.n_qubits
    gates = circuit.get_gates()
    
    # Unicode symbols
    HORIZONTAL = '─'
    VERTICAL = '│'
    CONTROL = '●'
    TARGET = '⊗'
    SWAP = '✕'
    
    # Default qubit labels
    if qubit_labels is None:
        qubit_labels = [f"q{i}" for i in range(n_qubits)]
    else:
        qubit_labels = qubit_labels[:n_qubits]
    
    # Initialize lines
    lines = ["" for _ in range(n_qubits)]
    
    # Add qubit labels
    max_label_len = max(len(label) for label in qubit_labels)
    for i in range(n_qubits):
        lines[i] = f"{qubit_labels[i]:>{max_label_len}} : "
    
    # Gate symbols
    gate_symbols = {
        'H': 'H',
        'X': 'X',
        'Y': 'Y',
        'Z': 'Z',
        'S': 'S',
        'T': 'T',
        'CNOT': f'{CONTROL}{HORIZONTAL}{HORIZONTAL}{TARGET}',
        'CZ': f'{CONTROL}{HORIZONTAL}{HORIZONTAL}Z',
        'SWAP': f'{SWAP}{HORIZONTAL}{HORIZONTAL}{SWAP}',
        'TOFFOLI': f'{CONTROL}{HORIZONTAL}{HORIZONTAL}{CONTROL}{HORIZONTAL}{HORIZONTAL}{TARGET}',
    }
    
    # Process gates
    for gate in gates:
        name = gate['name']
        qubits = gate['qubits']
        params = gate.get('params', [])
        
        # Determine gate symbol
        if name in ['H', 'X', 'Y', 'Z', 'S', 'T']:
            symbol = f" {name} "
            width = 3
        elif name in ['RX', 'RY', 'RZ']:
            theta = params[0] if params else 0
            symbol = f" {name}({theta:.2f}) "
            width = len(symbol)
        elif name == 'CNOT' or name == 'CX':
            symbol = f" {CONTROL}--{TARGET} "
            width = 5
        elif name == 'CZ':
            symbol = f" {CONTROL}--Z "
            width = 5
        elif name == 'SWAP':
            symbol = f" {SWAP}--{SWAP} "
            width = 5
        elif name == 'TOFFOLI':
            symbol = f" {CONTROL}--{CONTROL}--{TARGET} "
            width = 7
        else:
            symbol = f" {name[:4]} "
            width = len(symbol)
        
        # Apply gate to qubits
        for i in range(n_qubits):
            if i in qubits:
                # Add gate symbol
                if len(qubits) == 1:
                    lines[i] += symbol
                else:
                    # Multi-qubit gate
                    idx = qubits.index(i)
                    if idx == 0:
                        lines[i] += f" {CONTROL} "
                    elif idx == len(qubits) - 1:
                        lines[i] += f" {TARGET} "
                    else:
                        lines[i] += f" {CONTROL} "
                    
                    # Add wire connectors
                    if idx < len(qubits) - 1:
                        lines[i] += "--"
                    else:
                        lines[i] += "  "
            else:
                # Wire
                if show_wires:
                    lines[i] += f" {HORIZONTAL}{HORIZONTAL} "
                else:
                    lines[i] += "    "
    
    # Clean up lines
    for i in range(n_qubits):
        lines[i] = lines[i].rstrip()
    
    return "\n".join(lines)


# ============================================================================
# CIRCUIT TO TEXT
# ============================================================================

def circuit_to_text(
    circuit: QuantumCircuit,
    show_wires: bool = True
) -> str:
    """
    Convert circuit to a simple text representation
    
    Args:
        circuit: QuantumCircuit object
        show_wires: Show wire lines
        
    Returns:
        str: Text representation
        
    Example:
        >>> print(circuit_to_text(circuit))
        Circuit: 2 qubits, 2 gates
        Gate 1: H on [0]
        Gate 2: CNOT on [0, 1]
    """
    gates = circuit.get_gates()
    
    lines = [
        f"Circuit: {circuit.n_qubits} qubits, {len(gates)} gates, depth={circuit.depth}",
        "-" * 40
    ]
    
    for i, gate in enumerate(gates, 1):
        name = gate['name']
        qubits = gate['qubits']
        params = gate.get('params', [])
        
        if params:
            param_str = f", params={params}"
        else:
            param_str = ""
        
        lines.append(f"Gate {i}: {name} on {qubits}{param_str}")
    
    return "\n".join(lines)


# ============================================================================
# CIRCUIT STATISTICS
# ============================================================================

def circuit_statistics(circuit: QuantumCircuit) -> Dict[str, Any]:
    """
    Get statistics about a quantum circuit
    
    Args:
        circuit: QuantumCircuit object
        
    Returns:
        Dict: Circuit statistics
        
    Example:
        >>> stats = circuit_statistics(circuit)
        >>> print(stats['gate_counts'])
        {'H': 1, 'CNOT': 1}
    """
    gates = circuit.get_gates()
    
    # Gate counts
    gate_counts = {}
    gate_types = set()
    
    for gate in gates:
        name = gate['name']
        gate_counts[name] = gate_counts.get(name, 0) + 1
        gate_types.add(name)
    
    # Count multi-qubit gates
    multi_qubit_gates = 0
    single_qubit_gates = 0
    
    for gate in gates:
        if len(gate['qubits']) > 1:
            multi_qubit_gates += 1
        else:
            single_qubit_gates += 1
    
    # Count controlled gates
    controlled_gates = sum(1 for gate in gates if gate['name'] in ['CNOT', 'CX', 'CZ', 'TOFFOLI', 'CU1'])
    
    # Count rotation gates
    rotation_gates = sum(1 for gate in gates if gate['name'] in ['RX', 'RY', 'RZ'])
    
    # Count measurement gates
    measurement_gates = sum(1 for gate in gates if gate['name'] == 'MEASURE')
    
    # Circuit depth by qubit
    qubit_depths = {i: 0 for i in range(circuit.n_qubits)}
    for gate in gates:
        for qubit in gate['qubits']:
            qubit_depths[qubit] = qubit_depths.get(qubit, 0) + 1
    
    # Qubit usage
    qubit_usage = {}
    for i in range(circuit.n_qubits):
        qubit_usage[i] = qubit_depths.get(i, 0)
    
    return {
        'n_qubits': circuit.n_qubits,
        'total_gates': len(gates),
        'depth': circuit.depth,
        'gate_counts': gate_counts,
        'gate_types': sorted(gate_types),
        'single_qubit_gates': single_qubit_gates,
        'multi_qubit_gates': multi_qubit_gates,
        'controlled_gates': controlled_gates,
        'rotation_gates': rotation_gates,
        'measurement_gates': measurement_gates,
        'qubit_depths': qubit_depths,
        'qubit_usage': qubit_usage,
        'max_depth_per_qubit': max(qubit_depths.values()) if qubit_depths else 0,
        'avg_depth_per_qubit': sum(qubit_depths.values()) / len(qubit_depths) if qubit_depths else 0
    }


# ============================================================================
# ADDITIONAL DRAWING UTILITIES
# ============================================================================

def draw_circuit_latex(
    circuit: QuantumCircuit,
    style: str = 'quantikz'
) -> str:
    """
    Generate LaTeX code for a circuit diagram
    
    Args:
        circuit: QuantumCircuit object
        style: 'quantikz' or 'qcircuit'
        
    Returns:
        str: LaTeX code
        
    Example:
        >>> print(draw_circuit_latex(circuit))
        \\begin{quantikz}
        \\lstick{q0} & \\gate{H} & \\ctrl{1} & \\qw \\\\
        \\lstick{q1} & \\qw & \\targ{} & \\qw \\\\
        \\end{quantikz}
    """
    # Import from interface.latex
    try:
        from ..interface.latex import circuit_to_latex
        return circuit_to_latex(circuit, style=style)
    except ImportError:
        logger.warning("Could not import circuit_to_latex from interface.latex")
        return circuit_to_text(circuit)


def draw_circuit_qiskit(
    circuit: QuantumCircuit,
    style: str = 'mpl'
):
    """
    Draw circuit using Qiskit (if available)
    
    Args:
        circuit: QuantumCircuit object
        style: 'mpl', 'text', or 'latex'
        
    Returns:
        Qiskit circuit drawing
        
    Example:
        >>> draw_circuit_qiskit(circuit, style='mpl')
    """
    try:
        from qiskit import QuantumCircuit as QiskitCircuit
        
        # Convert to Qiskit circuit
        qiskit_circ = QiskitCircuit(circuit.n_qubits)
        
        for gate in circuit.get_gates():
            name = gate['name']
            qubits = gate['qubits']
            params = gate.get('params', [])
            
            if name == 'H':
                for q in qubits:
                    qiskit_circ.h(q)
            elif name == 'X':
                for q in qubits:
                    qiskit_circ.x(q)
            elif name == 'Y':
                for q in qubits:
                    qiskit_circ.y(q)
            elif name == 'Z':
                for q in qubits:
                    qiskit_circ.z(q)
            elif name == 'S':
                for q in qubits:
                    qiskit_circ.s(q)
            elif name == 'T':
                for q in qubits:
                    qiskit_circ.t(q)
            elif name in ['RX', 'RY', 'RZ']:
                for q, theta in zip(qubits, params):
                    if name == 'RX':
                        qiskit_circ.rx(theta, q)
                    elif name == 'RY':
                        qiskit_circ.ry(theta, q)
                    else:
                        qiskit_circ.rz(theta, q)
            elif name in ['CNOT', 'CX']:
                for i in range(0, len(qubits), 2):
                    qiskit_circ.cx(qubits[i], qubits[i+1])
            elif name == 'CZ':
                for i in range(0, len(qubits), 2):
                    qiskit_circ.cz(qubits[i], qubits[i+1])
            elif name == 'SWAP':
                for i in range(0, len(qubits), 2):
                    qiskit_circ.swap(qubits[i], qubits[i+1])
            elif name == 'TOFFOLI':
                for i in range(0, len(qubits), 3):
                    qiskit_circ.ccx(qubits[i], qubits[i+1], qubits[i+2])
        
        return qiskit_circ.draw(style)
        
    except ImportError:
        logger.warning("Qiskit not available. Install with: pip install qiskit")
        return draw_circuit(circuit, style='ascii')


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'draw_circuit',
    '_draw_ascii',
    '_draw_unicode',
    'circuit_to_text',
    'circuit_statistics',
    'draw_circuit_latex',
    'draw_circuit_qiskit',
]
