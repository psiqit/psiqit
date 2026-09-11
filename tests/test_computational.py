"""
PSIQIT Comprehensive Test Suite
Complete testing of all modules and functionality
Version 1.0.3

این فایل تمام ماژول‌های PSIQIT را تست می‌کند:
- Module Imports
- Quantum States
- Quantum Operators
- Quantum Circuits
- Algorithms
- Information Theory
- Variational Methods
- Dynamics
- Utilities
- Visualization
- Lab
- Error Correction
- Noise Canceling
- Interface
- Numerical Accuracy

اجرا:
    python test_comprehensive.py
"""

import sys
import time
import numpy as np
from datetime import datetime

# ============================================================================
# TEST COUNTER
# ============================================================================

class TestCounter:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.start_time = time.time()
    
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
        elapsed = time.time() - self.start_time
        print("\n" + "="*70)
        print(" TEST RESULTS SUMMARY")
        print("="*70)
        print(f"  Total Tests: {self.total}")
        print(f"  Passed:      {self.passed}")
        print(f"  Failed:      {self.failed}")
        print(f"  Success Rate: {self.passed/self.total*100:.1f}%")
        print(f"  Time:        {elapsed:.2f}s")
        if self.errors:
            print("\n  Failed Tests:")
            for name, error in self.errors:
                print(f"    - {name}: {error}")
        print("="*70)
        return self.failed == 0


test = TestCounter()

# ============================================================================
# HEADER
# ============================================================================

print("="*70)
print(" PSIQIT - Comprehensive Test Suite")
print(" Version 1.0.3")
print("="*70)
print(f" Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# ============================================================================
# TEST 1: MODULE IMPORTS
# ============================================================================

print("\n[1] Testing Module Imports")
print("-" * 50)

try:
    import psiqit
    from psiqit import __version__
    test.add_result("psiqit import", True)
except Exception as e:
    test.add_result("psiqit import", False, str(e))

try:
    from psiqit.quantum import (
        Ket, Bra, zero, one, plus, minus, basis,
        bell_phi_plus, bell_phi_minus, bell_psi_plus, bell_psi_minus,
        ghz, w_state, random_state,
        Operator, pauli_x, pauli_y, pauli_z,
        hadamard, phase, s_gate, t_gate,
        rx, ry, rz, cnot, cz, swap, toffoli, fredkin,
        measure, expectation, variance, projector, pauli_string,
        POVM, ProjectiveMeasurement,
        Alice, Bob, BB84, QuantumTeleportation
    )
    test.add_result("quantum module", True)
except Exception as e:
    test.add_result("quantum module", False, str(e))

try:
    from psiqit.circuits import (
        QuantumCircuit, QuantumRegister, Qubit,
        create_bell_circuit, create_ghz_circuit, create_w_circuit,
        BeamSplitter, PhaseShifter, OpticalCircuit
    )
    test.add_result("circuits module", True)
except Exception as e:
    test.add_result("circuits module", False, str(e))

try:
    from psiqit.algorithms import (
        grover_search, grover_search_simulated,
        deutsch_jozsa, deutsch_algorithm,
        bernstein_vazirani,
        qft, iqft, qft_matrix,
        quantum_phase_estimation,
        shor_factor, shor_algorithm
    )
    test.add_result("algorithms module", True)
except Exception as e:
    test.add_result("algorithms module", False, str(e))

try:
    from psiqit.info import (
        shannon_entropy, von_neumann_entropy, renyi_entropy,
        concurrence, negativity, entanglement_entropy,
        schmidt_decomposition, is_entangled,
        purity, mutual_information
    )
    test.add_result("info module", True)
except Exception as e:
    test.add_result("info module", False, str(e))

try:
    from psiqit.variational import (
        VQE, QAOA, SSVQE, ADAPTVQE,
        maxcut_hamiltonian, qubo_to_hamiltonian,
        VariationalMonteCarlo, RayleighRitz
    )
    test.add_result("variational module", True)
except Exception as e:
    test.add_result("variational module", False, str(e))

try:
    from psiqit.dynamics import (
        WaveFunction, solve_time_independent, solve_time_dependent,
        LindbladSolver, QuantumTrajectory,
        HeisenbergEvolution, AdiabaticEvolution,
        TrotterEvolution, ChebyshevEvolution
    )
    test.add_result("dynamics module", True)
except Exception as e:
    test.add_result("dynamics module", False, str(e))

try:
    from psiqit.utils import (
        ket_to_density, to_bloch_coordinates,
        random_state as utils_random_state, set_random_seed,
        is_unitary, validate_qubits,
        get_config, logger
    )
    test.add_result("utils module", True)
except Exception as e:
    test.add_result("utils module", False, str(e))

try:
    from psiqit.visualization import (
        state_to_bloch, bloch_sphere,
        draw_circuit, circuit_statistics,
        render_equation, schrodinger_equation
    )
    test.add_result("visualization module", True)
except Exception as e:
    test.add_result("visualization module", False, str(e))

try:
    from psiqit.lab import QuantumLab, PredefinedExperiments
    test.add_result("lab module", True)
except Exception as e:
    test.add_result("lab module", False, str(e))

try:
    from psiqit.error_correction import BitFlipCode, PhaseFlipCode, ShorCode, SteaneCode
    test.add_result("error_correction module", True)
except Exception as e:
    test.add_result("error_correction module", False, str(e))

try:
    from psiqit.noise_canceling import (
        bit_flip_channel, phase_flip_channel, depolarizing_channel,
        amplitude_damping_channel, zero_noise_extrapolation
    )
    test.add_result("noise_canceling module", True)
except Exception as e:
    test.add_result("noise_canceling module", False, str(e))

try:
    from psiqit.interface import parse_gates, matrix_to_latex, state_to_latex
    test.add_result("interface module", True)
except Exception as e:
    test.add_result("interface module", False, str(e))

# ============================================================================
# TEST 2: QUANTUM STATES
# ============================================================================

print("\n[2] Testing Quantum States")
print("-" * 50)

state = zero()
test.add_result("zero state", abs(state.data[0] - 1.0) < 1e-6)

state = one()
test.add_result("one state", abs(state.data[1] - 1.0) < 1e-6)

state = plus()
test.add_result("plus state", abs(state.data[0] - 1/np.sqrt(2)) < 1e-6)

state = minus()
test.add_result("minus state", abs(state.data[0] - 1/np.sqrt(2)) < 1e-6 and abs(state.data[1] + 1/np.sqrt(2)) < 1e-6)

bell = bell_phi_plus()
test.add_result("bell_phi_plus", bell.dim == 4)

bell = bell_phi_minus()
test.add_result("bell_phi_minus", bell.dim == 4)

bell = bell_psi_plus()
test.add_result("bell_psi_plus", bell.dim == 4)

bell = bell_psi_minus()
test.add_result("bell_psi_minus", bell.dim == 4)

ghz_state = ghz(3)
test.add_result("ghz(3)", ghz_state.dim == 8)

w = w_state(3)
test.add_result("w_state(3)", w.dim == 8)

ket1 = Ket([1, 0])
ket2 = Ket([0, 1])
inner = ket1.inner(ket2)
test.add_result("inner product", abs(inner) < 1e-10)

fid = abs(ket1.inner(ket1))**2
test.add_result("fidelity", abs(fid - 1.0) < 1e-6)

set_random_seed(42)
rnd = random_state(4)
test.add_result("random_state", rnd.dim == 4)

result = rnd.measure(shots=100)
test.add_result("measurement", 'counts' in result)

basis_state = basis(4, 2)
test.add_result("basis state", basis_state.dim == 4 and abs(basis_state.data[2] - 1.0) < 1e-6)

# ============================================================================
# TEST 3: QUANTUM OPERATORS
# ============================================================================

print("\n[3] Testing Quantum Operators")
print("-" * 50)

X = pauli_x()
test.add_result("pauli_x", X.is_unitary)

Y = pauli_y()
test.add_result("pauli_y", Y.is_unitary)

Z = pauli_z()
test.add_result("pauli_z", Z.is_unitary)

H = hadamard()
test.add_result("hadamard", H.is_unitary)

S = s_gate()
test.add_result("s_gate", S.is_unitary)

T = t_gate()
test.add_result("t_gate", T.is_unitary)

Rx = rx(np.pi/2)
test.add_result("rx", Rx.is_unitary)

Ry = ry(np.pi/2)
test.add_result("ry", Ry.is_unitary)

Rz = rz(np.pi/2)
test.add_result("rz", Rz.is_unitary)

CNOT = cnot()
test.add_result("cnot", CNOT.dim == 4)

CZ = cz()
test.add_result("cz", CZ.dim == 4)

SWAP = swap()
test.add_result("swap", SWAP.dim == 4)

Tof = toffoli()
test.add_result("toffoli", Tof.dim == 8)

Fred = fredkin()
test.add_result("fredkin", Fred.dim == 8)

comm = X.commutator(Z)
test.add_result("commutator", comm.dim == 2)

XZ = np.kron(X.data, Z.data)
test.add_result("tensor_product", XZ.shape == (4, 4))

state = Ket([1, 0])
exp = expectation(Z, state)
test.add_result("expectation", abs(exp - 1.0) < 1e-6)

P = projector(state)
test.add_result("projector", P.dim == 2)

PS = pauli_string('XYZ')
test.add_result("pauli_string", PS.dim == 8)

# ============================================================================
# TEST 4: QUANTUM CIRCUITS
# ============================================================================

print("\n[4] Testing Quantum Circuits")
print("-" * 50)

circ = QuantumCircuit(2)
test.add_result("circuit creation", circ.n_qubits == 2)

circ.h(0)
circ.cx(0, 1)
test.add_result("gates application", circ.depth == 2)

state = circ.run()
test.add_result("circuit run", len(state.data) == 4)

result = circ.measure(shots=100)
test.add_result("circuit measure", 'counts' in result)

circ2 = create_bell_circuit()
state = circ2.run()
bell_state = bell_phi_plus()
diff = np.linalg.norm(state.data - bell_state.data)
test.add_result("Bell circuit", diff < 1e-10)

# GHZ circuit - با اصلاح _expand_2q_gate
circ3 = create_ghz_circuit(3)
state = circ3.run()
ghz_state = ghz(3)
diff = np.linalg.norm(state.data - ghz_state.data)
test.add_result("GHZ circuit", diff < 1e-6)

# W circuit - با اصلاح create_w_circuit
circ4 = create_w_circuit(3)
state = circ4.run()
w_state_actual = w_state(3)
diff = np.linalg.norm(state.data - w_state_actual.data)
test.add_result("W circuit", diff < 1e-6)

reg = QuantumRegister(2)
test.add_result("register creation", reg.n_qubits == 2)
test.add_result("register state", len(reg.get_state_vector().data) == 4)

q = Qubit(0)
q.apply_h()
test.add_result("qubit", q.is_measured == False)

# ============================================================================
# TEST 5: ALGORITHMS
# ============================================================================

print("\n[5] Testing Quantum Algorithms")
print("-" * 50)

# Grover simulated (reliable)
result = grover_search_simulated(n_qubits=3, target=5)
test.add_result("grover_search_simulated", result['most_likely'] == 5)

# Grover circuit
result = grover_search(n_qubits=3, target=5, shots=1000)
test.add_result("grover_search_circuit", result['probability_target'] > 0.5)

# Deutsch-Jozsa - constant
result = deutsch_jozsa(function_type='constant_zero', n_qubits=2, shots=100)
test.add_result("deutsch_jozsa_constant", result['result'] == 'constant')

# Deutsch-Jozsa - balanced
result = deutsch_jozsa(function_type='balanced', n_qubits=2, shots=100)
test.add_result("deutsch_jozsa_balanced", result['result'] == 'balanced')

# Deutsch algorithm
result = deutsch_algorithm(function_type='constant_zero', shots=100)
test.add_result("deutsch_constant", result['result'] == 'constant')

result = deutsch_algorithm(function_type='balanced', shots=100)
test.add_result("deutsch_balanced", result['result'] == 'balanced')

# Bernstein-Vazirani
result = bernstein_vazirani(hidden_string='101', shots=1000)
test.add_result("bernstein_vazirani", result['found_string'] == '101')

# QFT
state = Ket([1, 0, 0, 0])
qft_state = qft(state)
test.add_result("qft", qft_state.dim == 4)

original = iqft(qft_state)
test.add_result("iqft", np.allclose(original.data, state.data, atol=1e-10))

Q = qft_matrix(3)
test.add_result("qft_matrix", Q.shape == (8, 8))

# QPE
from psiqit.quantum import pauli_z
result = quantum_phase_estimation(pauli_z(), n_qubits=3, state=zero(), shots=100)
test.add_result("qpe", 'phase' in result)

# Shor (classical)
try:
    p, q = shor_factor(15, use_quantum=False, max_attempts=5)
    test.add_result("shor_factor", p * q == 15)
except:
    test.add_result("shor_factor", False, "Failed to factor 15")

# ============================================================================
# TEST 6: INFORMATION THEORY
# ============================================================================

print("\n[6] Testing Information Theory")
print("-" * 50)

probs = [0.5, 0.5]
H_ent = shannon_entropy(probs, base='2')
test.add_result("shannon_entropy", abs(H_ent - 1.0) < 1e-6)

rho = np.array([[0.5, 0], [0, 0.5]])
S = von_neumann_entropy(rho, base='2')
test.add_result("von_neumann_entropy", abs(S - 1.0) < 1e-6)

S2 = renyi_entropy(rho, alpha=2, base='2')
test.add_result("renyi_entropy", S2 > 0)

C = concurrence(bell_phi_plus())
test.add_result("concurrence", abs(C - 1.0) < 1e-6)

N = negativity(bell_phi_plus())
test.add_result("negativity", N > 0)

p = purity(rho)
test.add_result("purity", abs(p - 0.5) < 1e-6)

test.add_result("is_entangled_bell", is_entangled(bell_phi_plus()))
test.add_result("is_entangled_zero", not is_entangled(zero()))

schmidt = schmidt_decomposition(bell_phi_plus(), [2, 2])
test.add_result("schmidt_decomposition", schmidt['rank'] == 2)

MI = mutual_information(np.eye(4)/4, 2, 2, base='2')
test.add_result("mutual_information", MI >= 0)

# ============================================================================
# TEST 7: VARIATIONAL METHODS
# ============================================================================

print("\n[7] Testing Variational Methods")
print("-" * 50)

hamiltonian = {'Z0Z1': 1.0, 'Z0': 0.5, 'Z1': -0.5}
vqe = VQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=1)
test.add_result("VQE creation", vqe.n_qubits == 2)

energy = vqe.evaluate_energy()
test.add_result("VQE energy", isinstance(energy, float))

edges = [(0, 1)]
H_mc = maxcut_hamiltonian(edges)
qaoa = QAOA(n_qubits=2, hamiltonian=H_mc, p=1)
test.add_result("QAOA creation", qaoa.n_qubits == 2)

test.add_result("maxcut_hamiltonian", len(H_mc) == 1)

qubo = {(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0}
H_qubo = qubo_to_hamiltonian(qubo, 2)
test.add_result("qubo_to_hamiltonian", len(H_qubo) > 0)

ssvqe = SSVQE(n_qubits=2, hamiltonian=hamiltonian, n_states=2)
test.add_result("SSVQE", ssvqe.n_qubits == 2)

def ansatz(params):
    return np.array([np.cos(params[0]), np.sin(params[0])])
H_mat = np.array([[2, 1], [1, 2]])
rr = RayleighRitz(H_mat, ansatz, n_params=1)
test.add_result("RayleighRitz", rr.n_params == 1)

# ============================================================================
# TEST 8: DYNAMICS
# ============================================================================

print("\n[8] Testing Dynamics")
print("-" * 50)

x = np.linspace(-5, 5, 100)
psi = np.exp(-x**2/2) / np.sqrt(np.sqrt(np.pi))
wf = WaveFunction(x, psi)
test.add_result("WaveFunction", len(wf.x) == 100)

def harmonic(x): return 0.5 * x**2
energies, states = solve_time_independent(harmonic, (-5, 5), n_points=100, n_states=3)
test.add_result("solve_time_independent", len(energies) == 3)

L = pauli_x()
solver = LindbladSolver(pauli_z(), [L], gamma=[0.1])
test.add_result("LindbladSolver", solver.dim == 2)

qt = QuantumTrajectory(pauli_z(), [L], gamma=[0.1])
test.add_result("QuantumTrajectory", qt.dim == 2)

hev = HeisenbergEvolution(pauli_z())
test.add_result("HeisenbergEvolution", hev.dim == 2)

evo = AdiabaticEvolution(pauli_x(), pauli_z(), T=1.0, n_steps=10)
test.add_result("AdiabaticEvolution", evo.dim == 2)

trotter = TrotterEvolution(pauli_x() + pauli_z(), n_steps=10)
test.add_result("TrotterEvolution", trotter.dim == 2)

# ============================================================================
# TEST 9: UTILITIES
# ============================================================================

print("\n[9] Testing Utilities")
print("-" * 50)

rho = ket_to_density(zero())
test.add_result("ket_to_density", np.array(rho).shape == (2, 2))

x, y, z = to_bloch_coordinates(zero())
test.add_result("to_bloch_coordinates", abs(z - 1.0) < 1e-6)

result = is_unitary(pauli_x())
test.add_result("is_unitary", result.is_valid)

try:
    validate_qubits(3)
    test.add_result("validate_qubits", True)
except:
    test.add_result("validate_qubits", False, "Failed")

config = get_config()
test.add_result("get_config", config.default_shots == 1024)

try:
    set_random_seed(42)
    test.add_result("set_random_seed", True)
except:
    test.add_result("set_random_seed", False, "Failed")

# ============================================================================
# TEST 10: VISUALIZATION
# ============================================================================

print("\n[10] Testing Visualization")
print("-" * 50)

circ = QuantumCircuit(2)
circ.h(0).cx(0, 1)
draw = draw_circuit(circ)
test.add_result("draw_circuit", len(draw) > 0)

stats = circuit_statistics(circ)
test.add_result("circuit_statistics", stats['total_gates'] == 2)

x, y, z = state_to_bloch(zero())
test.add_result("state_to_bloch", abs(z - 1.0) < 1e-6)

eq = render_equation(r"E = mc^2")
test.add_result("render_equation", len(eq) > 0)

eq = schrodinger_equation()
test.add_result("schrodinger_equation", 'hbar' in eq)

# ============================================================================
# TEST 11: LAB
# ============================================================================

print("\n[11] Testing Lab Module")
print("-" * 50)

lab = QuantumLab()
test.add_result("QuantumLab", len(lab.experiments) == 0)

exp = lab.create_experiment("Test", n_qubits=2)
exp.add_gate('h', 0).add_gate('cx', 0, 1)
test.add_result("create_experiment", len(exp.gates) == 2)

result = lab.run_experiment(exp, shots=100)
test.add_result("run_experiment", result.status.value == 'completed')

bell = PredefinedExperiments.bell_state()
test.add_result("PredefinedExperiments.bell", len(bell.gates) == 2)

ghz_exp = PredefinedExperiments.ghz_state(3)
test.add_result("PredefinedExperiments.ghz", len(ghz_exp.gates) == 3)

w_exp = PredefinedExperiments.w_state(3)
test.add_result("PredefinedExperiments.w", len(w_exp.gates) > 0)

# ============================================================================
# TEST 12: ERROR CORRECTION
# ============================================================================

print("\n[12] Testing Error Correction")
print("-" * 50)

bf = BitFlipCode(n=3)
state = zero()
encoded = bf.encode(state)
result = bf.decode(encoded)
test.add_result("BitFlipCode", result.success)

pf = PhaseFlipCode(n=3)
state = plus()
encoded = pf.encode(state)
result = pf.decode(encoded)
test.add_result("PhaseFlipCode", result.success)

shor = ShorCode()
state = zero()
encoded = shor.encode(state)
result = shor.decode(encoded)
test.add_result("ShorCode", result.success)

steane = SteaneCode()
state = zero()
encoded = steane.encode(state)
result = steane.decode(encoded)
test.add_result("SteaneCode", result.success)

# ============================================================================
# TEST 13: NOISE CANCELING
# ============================================================================

print("\n[13] Testing Noise Canceling")
print("-" * 50)

state = zero()
result = bit_flip_channel(state, p=0.1)
test.add_result("bit_flip_channel", result.fidelity > 0)

result = phase_flip_channel(state, p=0.1)
test.add_result("phase_flip_channel", result.fidelity > 0)

result = depolarizing_channel(state, p=0.1)
test.add_result("depolarizing_channel", result.fidelity > 0)

state = one()
result = amplitude_damping_channel(state, gamma=0.5)
test.add_result("amplitude_damping_channel", result.fidelity > 0)

results = []
for p in [0.1, 0.2, 0.3]:
    r = depolarizing_channel(state, p)
    results.append(r)
value = zero_noise_extrapolation(results)
test.add_result("zero_noise_extrapolation", value > 0)

# ============================================================================
# TEST 14: INTERFACE
# ============================================================================

print("\n[14] Testing Interface")
print("-" * 50)

gates = parse_gates("H(0),CNOT(0,1),RX(0)[1.5708]")
test.add_result("parse_gates", len(gates) == 3)

latex = matrix_to_latex(pauli_x(), name="X")
test.add_result("matrix_to_latex", "\\begin" in latex)

state = Ket([1/np.sqrt(2), 1/np.sqrt(2)])
latex = state_to_latex(state, name='psi')
test.add_result("state_to_latex", "\\rangle" in latex)

# ============================================================================
# TEST 15: NUMERICAL ACCURACY
# ============================================================================

print("\n[15] Testing Numerical Accuracy")
print("-" * 50)

U = np.array([[1, 0], [0, 1]], dtype=complex)
test.add_result("identity unitary", np.allclose(U @ U.conj().T, np.eye(2)))

H_test = np.array([[1, 0], [0, -1]], dtype=complex)
test.add_result("Hermitian", np.allclose(H_test, H_test.conj().T))

rho = np.array([[0.5, 0], [0, 0.5]])
test.add_result("trace preservation", abs(np.trace(rho) - 1.0) < 1e-6)

state = Ket([1, 0])
probs = np.abs(state.data)**2
test.add_result("probability normalization", abs(np.sum(probs) - 1.0) < 1e-6)

from psiqit.algorithms import grover_probability
prob = grover_probability(3, 2)
test.add_result("Grover probability", prob > 0.9)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print(" PSIQIT COMPREHENSIVE TEST SUITE COMPLETED")
print("="*70)

success = test.summary()

if success:
    print("\n🎉 All tests passed! PSIQIT is fully functional!")
    print("   Version 1.0.3 is ready for production use.")
else:
    print("\n⚠️ Some tests failed. Please check the errors above.")
    print("   Review the failed tests and fix the issues.")

print("\n" + "="*70)

# ============================================================================
# SAVE TEST RESULTS
# ============================================================================

try:
    with open("test_comprehensive_results.txt", "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("PSIQIT Comprehensive Test Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Version: 1.0.3\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total Tests: {test.total}\n")
        f.write(f"Passed: {test.passed}\n")
        f.write(f"Failed: {test.failed}\n")
        f.write(f"Success Rate: {test.passed/test.total*100:.1f}%\n")
        f.write(f"Time: {time.time() - test.start_time:.2f}s\n\n")
        if test.errors:
            f.write("Failed Tests:\n")
            for name, error in test.errors:
                f.write(f"  - {name}: {error}\n")
        f.write("\n" + "="*70 + "\n")
    print("📄 Test results saved to: test_comprehensive_results.txt")
except:
    pass