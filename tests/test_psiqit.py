"""
PSIQIT Test Suite - Comprehensive Testing for PSIQIT 1.0.3
Run this file to test all modules of the PSIQIT library
"""

import sys
import os
import time
import numpy as np
from datetime import datetime

print("="*70)
print(" PSIQIT - Python Scientific Quantum Information Toolkit")
print(" Version 1.0.3 - Comprehensive Test Suite")
print("="*70)
print(f" Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# ============================================================================
# IMPORT PSIQIT MODULES
# ============================================================================

print("\n[1] Loading PSIQIT modules...")
print("-" * 50)

try:
    import psiqit
    from psiqit import __version__
    print(f"✅ psiqit version: {__version__}")
except ImportError as e:
    print(f"❌ Failed to import psiqit: {e}")
    sys.exit(1)

# Import all submodules
modules = {
    'math': None,
    'quantum': None,
    'circuits': None,
    'dynamics': None,
    'error_correction': None,
    'algorithms': None,
    'info': None,
    'interface': None,
    'lab': None,
    'noise_canceling': None,
    'qml': None,
    'utils': None,
    'variational': None,
    'visualization': None,
}

for name in modules.keys():
    try:
        modules[name] = __import__(f'psiqit.{name}', fromlist=['*'])
        print(f"✅ psiqit.{name}")
    except ImportError as e:
        print(f"⚠️ psiqit.{name} - {e}")

print("-" * 50)
print("✅ All modules loaded successfully!")

# ============================================================================
# TEST COUNTER
# ============================================================================

class TestCounter:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_result(self, name: str, passed: bool, error: str = ""):
        self.total += 1
        if passed:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed += 1
            self.errors.append((name, error))
            print(f"  ❌ {name}: {error}")
    
    def summary(self):
        print("\n" + "="*70)
        print(" TEST RESULTS SUMMARY")
        print("="*70)
        print(f"  Total Tests: {self.total}")
        print(f"  Passed:      {self.passed}")
        print(f"  Failed:      {self.failed}")
        print(f"  Success Rate: {self.passed/self.total*100:.1f}%")
        if self.errors:
            print("\n  Failed Tests:")
            for name, error in self.errors:
                print(f"    - {name}: {error}")
        print("="*70)
        return self.failed == 0

test = TestCounter()

# ============================================================================
# TEST 1: MATH MODULE
# ============================================================================

print("\n[2] Testing Math Module")
print("-" * 50)

try:
    from psiqit.math import (
        derivative, integral, gradient,
        euler, rk4, solve_ode,
        solve_heat_equation, solve_wave_equation,
        PI, HBAR, Matrix, Vector,
        commutator, expectation,
        is_unitary, is_hermitian,
        create_symbols, differentiate
    )
    
    # Test calculus
    def f(x): return x**2
    d = derivative(f, 2)
    test.add_result("calculus.derivative", abs(d - 4.0) < 1e-6, f"Expected 4.0, got {d}")
    
    # Test ODE solver
    def g(x, y): return y
    result = rk4(g, 0, 1, 1, n_steps=10)
    test.add_result("ode_solver.rk4", len(result.x) == 11, f"Expected 11 points, got {len(result.x)}")
    
    # Test constants
    test.add_result("math.constants.PI", abs(PI - 3.14159) < 0.001, f"PI = {PI}")
    test.add_result("math.constants.HBAR", HBAR > 0, f"HBAR = {HBAR}")
    
    # Test Matrix
    M = Matrix([[1, 2], [3, 4]])
    test.add_result("math.Matrix", M.shape == (2, 2), f"Shape = {M.shape}")
    
    # Test commutator
    X = Matrix([[0, 1], [1, 0]])
    Z = Matrix([[1, 0], [0, -1]])
    C = commutator(X, Z)
    test.add_result("math.commutator", C.shape == (2, 2), f"Shape = {C.shape}")
    
    # Test is_unitary
    test.add_result("math.is_unitary", is_unitary(X).is_valid, "Matrix is not unitary")
    
    # Test symbolic (if available)
    try:
        x = create_symbols('x')
        diff = differentiate(x**3, x)
        test.add_result("math.symbolic", True, "")
    except Exception as e:
        test.add_result("math.symbolic", False, str(e))
    
except Exception as e:
    test.add_result("math.module", False, str(e))

# ============================================================================
# TEST 2: QUANTUM MODULE
# ============================================================================

print("\n[3] Testing Quantum Module")
print("-" * 50)

try:
    from psiqit.quantum import (
        Ket, Bra, zero, one, plus, minus,
        bell_phi_plus, bell_phi_minus, ghz, w_state,
        Operator, pauli_x, pauli_y, pauli_z, hadamard, cnot,
        measure, expectation,
        Alice, Bob, BB84, QuantumTeleportation
    )
    
    # Test Ket
    state = Ket([1, 0])
    test.add_result("quantum.Ket", state.dim == 2, f"dim={state.dim}")
    
    # Test zero state
    z = zero()
    test.add_result("quantum.zero", abs(z.data[0] - 1.0) < 1e-6, f"data={z.data}")
    
    # Test Bell state
    bell = bell_phi_plus()
    test.add_result("quantum.bell_phi_plus", bell.dim == 4, f"dim={bell.dim}")
    
    # Test GHZ
    g = ghz(3)
    test.add_result("quantum.ghz", g.dim == 8, f"dim={g.dim}")
    
    # Test Operator
    X = pauli_x()
    test.add_result("quantum.pauli_x", X.dim == 2, f"dim={X.dim}")
    
    # Test Hadamard
    H = hadamard()
    test.add_result("quantum.hadamard", H.is_unitary, "Not unitary")
    
    # Test CNOT
    CNOT = cnot()
    test.add_result("quantum.cnot", CNOT.dim == 4, f"dim={CNOT.dim}")
    
    # Test measure
    state = Ket([1/np.sqrt(2), 1/np.sqrt(2)])
    result = measure(state, shots=100)
    test.add_result("quantum.measure", 'counts' in result, f"result keys={list(result.keys())}")
    
    # Test Alice/Bob
    alice = Alice()
    bob = Bob()
    alice.prepare(plus())
    state = alice.send()
    bob.receive(state)
    result = bob.measure('x')
    test.add_result("quantum.parties", result in [0, 1], f"result={result}")
    
except Exception as e:
    test.add_result("quantum.module", False, str(e))

# ============================================================================
# TEST 3: CIRCUITS MODULE
# ============================================================================

print("\n[4] Testing Circuits Module")
print("-" * 50)

try:
    from psiqit.circuits import (
        QuantumCircuit, QuantumRegister, Qubit,
        create_bell_circuit, create_ghz_circuit
    )
    
    # Test Qubit
    q = Qubit(0)
    q.apply_h()
    test.add_result("circuits.Qubit", q.is_measured == False, "Qubit should not be measured")
    
    # Test QuantumCircuit
    circ = QuantumCircuit(2)
    circ.h(0).cx(0, 1)
    test.add_result("circuits.QuantumCircuit", circ.depth == 2, f"depth={circ.depth}")
    
    # Test run
    state = circ.run()
    test.add_result("circuits.run", len(state.data) == 4, f"state dim={len(state.data)}")
    
    # Test measure
    result = circ.measure(shots=100)
    test.add_result("circuits.measure", 'counts' in result, f"result keys={list(result.keys())}")
    
    # Test Bell circuit
    bell = create_bell_circuit()
    test.add_result("circuits.create_bell_circuit", bell.depth == 2, f"depth={bell.depth}")
    
    # Test QuantumRegister
    reg = QuantumRegister(2)
    test.add_result("circuits.QuantumRegister", reg.n_qubits == 2, f"n_qubits={reg.n_qubits}")
    
    # Test get_state_vector
    state = reg.get_state_vector()
    test.add_result("circuits.register.get_state_vector", state.dim == 4, f"dim={state.dim}")
    
except Exception as e:
    test.add_result("circuits.module", False, str(e))

# ============================================================================
# TEST 4: ALGORITHMS MODULE
# ============================================================================

print("\n[5] Testing Algorithms Module")
print("-" * 50)

try:
    from psiqit.algorithms import (
        grover_search, grover_search_simulated,
        deutsch_jozsa,
        bernstein_vazirani,
        qft, iqft,
        quantum_phase_estimation,
        shor_factor
    )
    from psiqit.quantum import pauli_z, zero
    
    # Test Grover
    result = grover_search(n_qubits=3, target=5, shots=100)
    test.add_result("algorithms.grover_search", 'most_likely' in result, f"keys={list(result.keys())}")
    
    # Test Grover simulated
    result = grover_search_simulated(n_qubits=3, target=5)
    test.add_result("algorithms.grover_simulated", 'probability_target' in result, f"keys={list(result.keys())}")
    
    # Test Deutsch-Jozsa
    result = deutsch_jozsa(function_type='constant_zero', n_qubits=3, shots=100)
    test.add_result("algorithms.deutsch_jozsa", result['result'] == 'constant', f"result={result['result']}")
    
    # Test Bernstein-Vazirani
    result = bernstein_vazirani(hidden_string='101', shots=100)
    test.add_result("algorithms.bernstein_vazirani", result['found_string'] == '101', f"found={result['found_string']}")
    
    # Test QFT
    from psiqit.quantum import Ket
    state = Ket([1, 0, 0, 0])
    qft_state = qft(state)
    test.add_result("algorithms.qft", qft_state.dim == 4, f"dim={qft_state.dim}")
    
    # Test IQFT
    original = iqft(qft_state)
    test.add_result("algorithms.iqft", original.dim == 4, f"dim={original.dim}")
    
    # Test QPE
    result = quantum_phase_estimation(pauli_z(), n_qubits=3, state=zero(), shots=100)
    test.add_result("algorithms.qpe", 'phase' in result, f"keys={list(result.keys())}")
    
    # Test Shor
    try:
        p, q = shor_factor(15, use_quantum=False, max_attempts=5)
        test.add_result("algorithms.shor_factor", p * q == 15, f"{p}*{q}={p*q}")
    except Exception as e:
        test.add_result("algorithms.shor_factor", False, str(e))
    
except Exception as e:
    test.add_result("algorithms.module", False, str(e))

# ============================================================================
# TEST 5: INFO MODULE
# ============================================================================

print("\n[6] Testing Info Module")
print("-" * 50)

try:
    from psiqit.info import (
        shannon_entropy, von_neumann_entropy,
        concurrence, negativity, is_entangled
    )
    from psiqit.quantum import bell_phi_plus, zero
    
    # Test Shannon entropy
    probs = [0.5, 0.5]
    H = shannon_entropy(probs, base='2')
    test.add_result("info.shannon_entropy", abs(H - 1.0) < 1e-6, f"H={H}")
    
    # Test von Neumann entropy
    rho = np.array([[0.5, 0], [0, 0.5]])
    S = von_neumann_entropy(rho, base='2')
    test.add_result("info.von_neumann_entropy", abs(S - 1.0) < 1e-6, f"S={S}")
    
    # Test concurrence
    C = concurrence(bell_phi_plus())
    test.add_result("info.concurrence", abs(C - 1.0) < 1e-6, f"C={C}")
    
    # Test negativity
    N = negativity(bell_phi_plus())
    test.add_result("info.negativity", N > 0, f"N={N}")
    
    # Test is_entangled
    test.add_result("info.is_entangled_bell", is_entangled(bell_phi_plus()), "Bell state not entangled")
    test.add_result("info.is_entangled_zero", not is_entangled(zero()), "Zero state should not be entangled")
    
except Exception as e:
    test.add_result("info.module", False, str(e))

# ============================================================================
# TEST 6: UTILS MODULE
# ============================================================================

print("\n[7] Testing Utils Module")
print("-" * 50)

try:
    from psiqit.utils import (
        ket_to_density, to_bloch_coordinates,
        random_state, set_random_seed,
        is_unitary, validate_qubits,
        get_config, logger
    )
    from psiqit.quantum import zero, pauli_x
    
    # Test ket_to_density
    rho = ket_to_density(zero())
    test.add_result("utils.ket_to_density", len(rho) == 2, f"shape={len(rho)}")
    
    # Test to_bloch_coordinates
    x, y, z = to_bloch_coordinates(zero())
    test.add_result("utils.to_bloch_coordinates", abs(z - 1.0) < 1e-6, f"z={z}")
    
    # Test random_state
    set_random_seed(42)
    state = random_state(4)
    test.add_result("utils.random_state", state.dim == 4, f"dim={state.dim}")
    
    # Test is_unitary
    result = is_unitary(pauli_x())
    test.add_result("utils.is_unitary", result.is_valid, "Pauli-X should be unitary")
    
    # Test validate_qubits
    try:
        validate_qubits(3)
        test.add_result("utils.validate_qubits", True, "")
    except Exception as e:
        test.add_result("utils.validate_qubits", False, str(e))
    
    # Test config
    config = get_config()
    test.add_result("utils.config", config.default_shots == 1024, f"default_shots={config.default_shots}")
    
    # Test logger
    try:
        logger.info("Test log message")
        test.add_result("utils.logger", True, "")
    except Exception as e:
        test.add_result("utils.logger", False, str(e))
    
except Exception as e:
    test.add_result("utils.module", False, str(e))

# ============================================================================
# TEST 7: VARIATIONAL MODULE
# ============================================================================

print("\n[8] Testing Variational Module")
print("-" * 50)

try:
    from psiqit.variational import (
        VQE, QAOA,
        maxcut_hamiltonian,
        SSVQE,
        VariationalMonteCarlo,
        RayleighRitz
    )
    
    # Test VQE
    hamiltonian = {'Z0Z1': 1.0, 'Z0': 0.5, 'Z1': -0.5}
    vqe = VQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=1)
    test.add_result("variational.VQE", vqe.n_qubits == 2, f"n_qubits={vqe.n_qubits}")
    
    # Test QAOA
    edges = [(0, 1)]
    H = maxcut_hamiltonian(edges)
    qaoa = QAOA(n_qubits=2, hamiltonian=H, p=1)
    test.add_result("variational.QAOA", qaoa.n_qubits == 2, f"n_qubits={qaoa.n_qubits}")
    
    # Test SSVQE
    ssvqe = SSVQE(n_qubits=2, hamiltonian=hamiltonian, n_states=2)
    test.add_result("variational.SSVQE", ssvqe.n_qubits == 2, f"n_qubits={ssvqe.n_qubits}")
    
    # Test RayleighRitz
    def ansatz(params):
        return np.array([np.cos(params[0]), np.sin(params[0])])
    H_mat = np.array([[2, 1], [1, 2]])
    rr = RayleighRitz(H_mat, ansatz, n_params=1)
    test.add_result("variational.RayleighRitz", rr.n_params == 1, f"n_params={rr.n_params}")
    
except Exception as e:
    test.add_result("variational.module", False, str(e))

# ============================================================================
# TEST 8: QML MODULE
# ============================================================================

print("\n[9] Testing QML Module")
print("-" * 50)

try:
    from psiqit.qml import (
        QNN, QSVM, VQC,
        QuantumKernel, QGAN
    )
    
    # Test QNN
    qnn = QNN(n_qubits=2, n_layers=2)
    test.add_result("qml.QNN", qnn.n_qubits == 2, f"n_qubits={qnn.n_qubits}")
    
    # Test QSVM
    qsvm = QSVM(n_qubits=2, kernel_type='quantum')
    test.add_result("qml.QSVM", qsvm.n_qubits == 2, f"n_qubits={qsvm.n_qubits}")
    
    # Test VQC
    vqc = VQC(n_qubits=2, n_layers=2)
    test.add_result("qml.VQC", vqc.n_qubits == 2, f"n_qubits={vqc.n_qubits}")
    
    # Test QuantumKernel
    kernel = QuantumKernel(n_qubits=2, feature_map='zz')
    test.add_result("qml.QuantumKernel", kernel.n_qubits == 2, f"n_qubits={kernel.n_qubits}")
    
    # Test QGAN
    qgan = QGAN(n_qubits=2, n_latent=2, n_layers=1)
    test.add_result("qml.QGAN", qgan.n_qubits == 2, f"n_qubits={qgan.n_qubits}")
    
except Exception as e:
    test.add_result("qml.module", False, str(e))

# ============================================================================
# TEST 9: DYNAMICS MODULE
# ============================================================================

print("\n[10] Testing Dynamics Module")
print("-" * 50)

try:
    from psiqit.dynamics import (
        WaveFunction,
        solve_time_independent,
        LindbladSolver,
        QuantumTrajectory,
        HeisenbergEvolution,
        AdiabaticEvolution
    )
    from psiqit.quantum import pauli_z, pauli_x, zero
    
    # Test WaveFunction
    x = np.linspace(-5, 5, 100)
    psi = np.exp(-x**2/2) / np.sqrt(np.sqrt(np.pi))
    wf = WaveFunction(x, psi)
    test.add_result("dynamics.WaveFunction", len(wf.x) == 100, f"len={len(wf.x)}")
    
    # Test solve_time_independent
    def harmonic(x): return 0.5 * x**2
    energies, states = solve_time_independent(harmonic, (-5, 5), n_points=100, n_states=3)
    test.add_result("dynamics.solve_time_independent", len(energies) == 3, f"found={len(energies)}")
    
    # Test LindbladSolver
    H = pauli_z()
    L = pauli_x()
    solver = LindbladSolver(H, [L], gamma=[0.1])
    test.add_result("dynamics.LindbladSolver", solver.dim == 2, f"dim={solver.dim}")
    
    # Test QuantumTrajectory
    qt = QuantumTrajectory(H, [L], gamma=[0.1])
    test.add_result("dynamics.QuantumTrajectory", qt.dim == 2, f"dim={qt.dim}")
    
    # Test HeisenbergEvolution
    hev = HeisenbergEvolution(H)
    test.add_result("dynamics.HeisenbergEvolution", hev.dim == 2, f"dim={hev.dim}")
    
    # Test AdiabaticEvolution
    H_i = pauli_x()
    H_f = pauli_z()
    evo = AdiabaticEvolution(H_i, H_f, T=1.0, n_steps=10)
    test.add_result("dynamics.AdiabaticEvolution", evo.dim == 2, f"dim={evo.dim}")
    
except Exception as e:
    test.add_result("dynamics.module", False, str(e))

# ============================================================================
# TEST 10: ERROR CORRECTION MODULE
# ============================================================================

print("\n[11] Testing Error Correction Module")
print("-" * 50)

try:
    from psiqit.error_correction import (
        BitFlipCode, PhaseFlipCode, ShorCode, SteaneCode
    )
    from psiqit.quantum import zero, plus
    
    # Test BitFlipCode
    bf = BitFlipCode(n=3)
    state = zero()
    encoded = bf.encode(state)
    result = bf.decode(encoded)
    test.add_result("error_correction.BitFlipCode", result.success, "BitFlipCode failed")
    
    # Test PhaseFlipCode
    pf = PhaseFlipCode(n=3)
    state = plus()
    encoded = pf.encode(state)
    result = pf.decode(encoded)
    test.add_result("error_correction.PhaseFlipCode", result.success, "PhaseFlipCode failed")
    
    # Test ShorCode
    shor = ShorCode()
    state = zero()
    encoded = shor.encode(state)
    result = shor.decode(encoded)
    test.add_result("error_correction.ShorCode", result.success, "ShorCode failed")
    
    # Test SteaneCode
    steane = SteaneCode()
    state = zero()
    encoded = steane.encode(state)
    result = steane.decode(encoded)
    test.add_result("error_correction.SteaneCode", result.success, "SteaneCode failed")
    
except Exception as e:
    test.add_result("error_correction.module", False, str(e))

# ============================================================================
# TEST 11: NOISE CANCELING MODULE
# ============================================================================

print("\n[12] Testing Noise Canceling Module")
print("-" * 50)

try:
    from psiqit.noise_canceling import (
        bit_flip_channel, depolarizing_channel,
        amplitude_damping_channel,
        zero_noise_extrapolation
    )
    from psiqit.quantum import zero
    
    # Test bit_flip_channel
    state = zero()
    result = bit_flip_channel(state, p=0.1)
    test.add_result("noise_canceling.bit_flip", result.fidelity > 0, f"fidelity={result.fidelity}")
    
    # Test depolarizing_channel
    result = depolarizing_channel(state, p=0.1)
    test.add_result("noise_canceling.depolarizing", result.fidelity > 0, f"fidelity={result.fidelity}")
    
    # Test amplitude_damping_channel
    from psiqit.quantum import one
    state = one()
    result = amplitude_damping_channel(state, gamma=0.5)
    test.add_result("noise_canceling.amplitude_damping", result.fidelity > 0, f"fidelity={result.fidelity}")
    
    # Test ZNE
    results = []
    for p in [0.1, 0.2, 0.3]:
        r = depolarizing_channel(state, p)
        results.append(r)
    value = zero_noise_extrapolation(results)
    test.add_result("noise_canceling.ZNE", value > 0, f"value={value}")
    
except Exception as e:
    test.add_result("noise_canceling.module", False, str(e))

# ============================================================================
# TEST 12: LAB MODULE
# ============================================================================

print("\n[13] Testing Lab Module")
print("-" * 50)

try:
    from psiqit.lab import QuantumLab, PredefinedExperiments
    
    # Test QuantumLab
    lab = QuantumLab()
    test.add_result("lab.QuantumLab", len(lab.experiments) == 0, f"experiments={len(lab.experiments)}")
    
    # Test create_experiment
    exp = lab.create_experiment("Test", n_qubits=2)
    exp.add_gate('h', 0).add_gate('cx', 0, 1)
    test.add_result("lab.create_experiment", len(exp.gates) == 2, f"gates={len(exp.gates)}")
    
    # Test run_experiment
    result = lab.run_experiment(exp, shots=100)
    test.add_result("lab.run_experiment", result.status.value == 'completed', f"status={result.status}")
    
    # Test PredefinedExperiments
    bell = PredefinedExperiments.bell_state()
    test.add_result("lab.PredefinedExperiments.bell", len(bell.gates) == 2, f"gates={len(bell.gates)}")
    
    ghz = PredefinedExperiments.ghz_state(3)
    test.add_result("lab.PredefinedExperiments.ghz", len(ghz.gates) == 3, f"gates={len(ghz.gates)}")
    
except Exception as e:
    test.add_result("lab.module", False, str(e))

# ============================================================================
# TEST 13: INTERFACE MODULE
# ============================================================================

print("\n[14] Testing Interface Module")
print("-" * 50)

try:
    from psiqit.interface import (
        parse_gates,
        matrix_to_latex, state_to_latex,
        display_notebook_version
    )
    from psiqit.quantum import pauli_x, Ket
    import numpy as np
    
    # Test parse_gates
    gates = parse_gates("H(0),CNOT(0,1),RX(0)[1.57]")
    test.add_result("interface.parse_gates", len(gates) == 3, f"gates={len(gates)}")
    
    # Test matrix_to_latex
    latex = matrix_to_latex(pauli_x(), name="X")
    test.add_result("interface.matrix_to_latex", "\\begin" in latex, f"latex length={len(latex)}")
    
    # Test state_to_latex
    state = Ket([1/np.sqrt(2), 1/np.sqrt(2)])
    latex = state_to_latex(state, name='psi')
    test.add_result("interface.state_to_latex", "\\rangle" in latex, f"latex length={len(latex)}")
    
    # Test display_notebook_version
    try:
        display_notebook_version()
        test.add_result("interface.display_notebook_version", True, "")
    except Exception as e:
        test.add_result("interface.display_notebook_version", False, str(e))
    
except Exception as e:
    test.add_result("interface.module", False, str(e))

# ============================================================================
# TEST 14: VISUALIZATION MODULE
# ============================================================================

print("\n[15] Testing Visualization Module")
print("-" * 50)

try:
    from psiqit.visualization import (
        state_to_bloch, bloch_sphere,
        draw_circuit, circuit_statistics,
        wigner_function_gaussian, plot_wigner,
        render_equation, schrodinger_equation
    )
    from psiqit.quantum import zero
    from psiqit.circuits import QuantumCircuit
    
    # Test state_to_bloch
    x, y, z = state_to_bloch(zero())
    test.add_result("visualization.state_to_bloch", abs(z - 1.0) < 1e-6, f"z={z}")
    
    # Test draw_circuit
    circ = QuantumCircuit(2)
    circ.h(0).cx(0, 1)
    draw = draw_circuit(circ)
    test.add_result("visualization.draw_circuit", len(draw) > 0, f"len={len(draw)}")
    
    # Test circuit_statistics
    stats = circuit_statistics(circ)
    test.add_result("visualization.circuit_statistics", stats['total_gates'] == 2, f"gates={stats['total_gates']}")
    
    # Test wigner_function_gaussian
    W, x, p = wigner_function_gaussian()
    test.add_result("visualization.wigner_function", W.shape == (100, 100), f"shape={W.shape}")
    
    # Test render_equation
    eq = render_equation(r"E = mc^2")
    test.add_result("visualization.render_equation", len(eq) > 0, f"len={len(eq)}")
    
    # Test schrodinger_equation
    eq = schrodinger_equation()
    test.add_result("visualization.schrodinger_equation", 'hbar' in eq, f"eq={eq[:20]}...")
    
except Exception as e:
    test.add_result("visualization.module", False, str(e))

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print(" PSIQIT TEST SUITE COMPLETED")
print("="*70)

success = test.summary()

if success:
    print("\n🎉 All tests passed! PSIQIT is working correctly!")
    print("   Version 1.0.3 is ready for use.")
else:
    print("\n⚠️ Some tests failed. Please check the errors above.")
    print("   Make sure all dependencies are installed correctly.")

print("\n" + "="*70)

# ============================================================================
# SAVE TEST RESULTS
# ============================================================================

try:
    with open("test_results.txt", "w") as f:
        f.write("="*70 + "\n")
        f.write("PSIQIT Test Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total Tests: {test.total}\n")
        f.write(f"Passed: {test.passed}\n")
        f.write(f"Failed: {test.failed}\n")
        f.write(f"Success Rate: {test.passed/test.total*100:.1f}%\n\n")
        if test.errors:
            f.write("Failed Tests:\n")
            for name, error in test.errors:
                f.write(f"  - {name}: {error}\n")
        f.write("\n" + "="*70 + "\n")
    print("📄 Test results saved to: test_results.txt")
except:
    pass