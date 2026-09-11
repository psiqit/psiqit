"""
Command Line Interface Module
CLI tools for quantum circuit simulation and algorithm execution
"""

import argparse
import sys
import json
import re
from typing import List, Tuple, Optional, Dict, Any
from ..circuits.circuit import QuantumCircuit
from ..circuits.qubit import Qubit
from ..algorithms.grover import grover_search
from ..algorithms.qft import qft
from ..quantum.state import Ket, zero, one, plus
from ..utils.logger import logger


# ============================================================================
# GATE PARSING
# ============================================================================

def parse_gates(gates_str: str) -> List[Tuple[str, List[int], List[float]]]:
    """
    Parse a string representation of quantum gates
    
    Format: "gate_name(qubits)[params]"
    Example: "H(0),CNOT(0,1),RX(0)[1.57]"
    """
    if not gates_str:
        return []
    
    gates = []
    # الگوی Regex کل گیت را یکجا پیدا می‌کند و کاماهای داخل پرانتز را نادیده می‌گیرد
    pattern = r'([A-Z]+)\s*\(\s*([^)]*)\s*\)\s*(?:\[\s*([^\]]*)\s*\])?'
    matches = re.findall(pattern, gates_str)
    
    for name, qubits_str, params_str in matches:
        # Parse qubits
        qubits = []
        if qubits_str.strip():
            try:
                qubits = [int(q.strip()) for q in qubits_str.split(',') if q.strip()]
            except ValueError:
                logger.warning(f"Could not parse qubits: {qubits_str}")
                continue
        
        # Parse parameters
        params = []
        if params_str and params_str.strip():
            try:
                params = [float(p.strip()) for p in params_str.split(',') if p.strip()]
            except ValueError:
                logger.warning(f"Could not parse params: {params_str}")
        
        gates.append((name, qubits, params))
    
    return gates
# ============================================================================
# RUN CIRCUIT
# ============================================================================

def run_circuit(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run a quantum circuit from command line arguments
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dict: Circuit execution results
        
    Example:
        >>> args = argparse.Namespace(n_qubits=2, gates="H(0),CNOT(0,1)", shots=1024)
        >>> result = run_circuit(args)
    """
    logger.info(f"Running circuit with {args.n_qubits} qubits")
    
    # Create circuit
    circ = QuantumCircuit(args.n_qubits)
    
    # Parse and apply gates
    gates = parse_gates(args.gates)
    
    for gate_name, qubits, params in gates:
        if gate_name == 'H':
            for q in qubits:
                circ.h(q)
        elif gate_name == 'X':
            for q in qubits:
                circ.x(q)
        elif gate_name == 'Y':
            for q in qubits:
                circ.y(q)
        elif gate_name == 'Z':
            for q in qubits:
                circ.z(q)
        elif gate_name == 'S':
            for q in qubits:
                circ.s(q)
        elif gate_name == 'T':
            for q in qubits:
                circ.t(q)
        elif gate_name == 'RX':
            for q, theta in zip(qubits, params):
                circ.rx(q, theta)
        elif gate_name == 'RY':
            for q, theta in zip(qubits, params):
                circ.ry(q, theta)
        elif gate_name == 'RZ':
            for q, theta in zip(qubits, params):
                circ.rz(q, theta)
        elif gate_name == 'CNOT':
            for i in range(0, len(qubits), 2):
                circ.cx(qubits[i], qubits[i+1])
        elif gate_name == 'CZ':
            for i in range(0, len(qubits), 2):
                circ.cz(qubits[i], qubits[i+1])
        elif gate_name == 'SWAP':
            for i in range(0, len(qubits), 2):
                circ.swap(qubits[i], qubits[i+1])
        elif gate_name == 'TOFFOLI':
            for i in range(0, len(qubits), 3):
                circ.toffoli(qubits[i], qubits[i+1], qubits[i+2])
        else:
            logger.warning(f"Unknown gate: {gate_name}")
    
    # Run simulation
    state = circ.run()
    
    # Measure
    if args.shots > 0:
        result = circ.measure(shots=args.shots)
    else:
        result = {'state': state.data.tolist(), 'shots': 0}
    
    # Add circuit info
    result['n_qubits'] = args.n_qubits
    result['depth'] = circ.depth
    result['gates'] = len(circ.get_gates())
    
    return result


# ============================================================================
# RUN ALGORITHM
# ============================================================================

def run_algorithm(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run a quantum algorithm from command line arguments
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dict: Algorithm execution results
        
    Example:
        >>> args = argparse.Namespace(algorithm='grover', n_qubits=3, target=5)
        >>> result = run_algorithm(args)
    """
    logger.info(f"Running algorithm: {args.algorithm}")
    
    if args.algorithm == 'grover':
        result = grover_search(
            n_qubits=args.n_qubits,
            target=args.target,
            shots=args.shots or 1024
        )
        return result
    
    elif args.algorithm == 'qft':
        from ..algorithms.qft import qft_circuit
        circ = qft_circuit(args.n_qubits)
        state = circ.run()
        return {
            'algorithm': 'qft',
            'n_qubits': args.n_qubits,
            'state': state.data.tolist(),
            'depth': circ.depth
        }
    
    else:
        logger.error(f"Unknown algorithm: {args.algorithm}")
        return {'error': f"Unknown algorithm: {args.algorithm}"}


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    """
    Main entry point for the PSIQIT command line interface
    
    Example:
        $ psiqit circuit --n-qubits 2 --gates "H(0),CNOT(0,1)" --shots 1024
        $ psiqit algorithm grover --n-qubits 3 --target 5
    """
    parser = argparse.ArgumentParser(
        prog='psiqit',
        description='PSIQIT - Python Scientific Quantum Information Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # ========================================================================
    # CIRCUIT COMMAND
    # ========================================================================
    circuit_parser = subparsers.add_parser('circuit', help='Run a quantum circuit')
    circuit_parser.add_argument(
        '--n-qubits', '-n',
        type=int,
        required=True,
        help='Number of qubits'
    )
    circuit_parser.add_argument(
        '--gates', '-g',
        type=str,
        required=True,
        help='Gates to apply (e.g., "H(0),CNOT(0,1),RX(0)[1.57]")'
    )
    circuit_parser.add_argument(
        '--shots', '-s',
        type=int,
        default=1024,
        help='Number of measurement shots (default: 1024)'
    )
    circuit_parser.add_argument(
        '--output', '-o',
        type=str,
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    circuit_parser.add_argument(
        '--draw', '-d',
        action='store_true',
        help='Draw the circuit diagram'
    )
    
    # ========================================================================
    # ALGORITHM COMMAND
    # ========================================================================
    algorithm_parser = subparsers.add_parser('algorithm', help='Run a quantum algorithm')
    algorithm_parser.add_argument(
        'algorithm',
        type=str,
        choices=['grover', 'qft'],
        help='Algorithm to run'
    )
    algorithm_parser.add_argument(
        '--n-qubits', '-n',
        type=int,
        required=True,
        help='Number of qubits'
    )
    algorithm_parser.add_argument(
        '--target', '-t',
        type=int,
        default=0,
        help='Target state for Grover search'
    )
    algorithm_parser.add_argument(
        '--shots', '-s',
        type=int,
        default=1024,
        help='Number of measurement shots (default: 1024)'
    )
    algorithm_parser.add_argument(
        '--output', '-o',
        type=str,
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    
    # ========================================================================
    # VERSION COMMAND
    # ========================================================================
    parser.add_argument(
        '--version', '-v',
        action='store_true',
        help='Show version information'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show version
    if args.version:
        from ..version import __version__
        print(f"PSIQIT version {__version__}")
        print("Python Scientific Quantum Information Toolkit")
        return
    
    # No command
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    try:
        if args.command == 'circuit':
            result = run_circuit(args)
            
            # Draw circuit
            if args.draw:
                from ..circuits.circuit import QuantumCircuit
                circ = QuantumCircuit(args.n_qubits)
                gates = parse_gates(args.gates)
                for gate_name, qubits, params in gates:
                    if gate_name == 'H':
                        for q in qubits:
                            circ.h(q)
                    elif gate_name == 'CNOT':
                        for i in range(0, len(qubits), 2):
                            circ.cx(qubits[i], qubits[i+1])
                    # Add other gates...
                print("\nCircuit Diagram:")
                print(circ.draw())
                print()
            
            # Output results
            if args.output == 'json':
                print(json.dumps(result, indent=2))
            else:
                print(f"\nCircuit Results:")
                print(f"  Qubits: {result['n_qubits']}")
                print(f"  Depth: {result['depth']}")
                print(f"  Gates: {result['gates']}")
                if 'counts' in result:
                    print(f"  Measurement Counts:")
                    for state, count in sorted(result['counts'].items()):
                        print(f"    {state}: {count}")
                if 'state' in result:
                    print(f"  State Vector:")
                    print(f"    {result['state']}")
        
        elif args.command == 'algorithm':
            result = run_algorithm(args)
            
            if args.output == 'json':
                print(json.dumps(result, indent=2))
            else:
                print(f"\nAlgorithm Results:")
                print(f"  Algorithm: {args.algorithm}")
                for key, value in result.items():
                    if key not in ['state']:
                        print(f"  {key}: {value}")
                if 'state' in result:
                    print(f"  State Vector: {result['state']}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# ============================================================================
# ENTRY POINT
# ============================================================================

def cli_main():
    """Entry point for console script"""
    main()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'parse_gates',
    'run_circuit',
    'run_algorithm',
    'main',
    'cli_main',
]


# ============================================================================
# MAIN GUARD
# ============================================================================

if __name__ == '__main__':
    main()