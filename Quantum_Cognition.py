"""
Quantum_Cognition.py — Quantum Probability Theory for Cognitive Modeling.

Implements the Busemeyer–Bruza quantum cognition framework using exact
quantum probability mathematics — not metaphor, not vocabulary borrowing.

Mathematical foundations:
  • Cognitive states as unit vectors |ψ⟩ ∈ ℂⁿ (Hilbert space)
  • Born rule: P(outcome i) = |⟨eᵢ|ψ⟩|²
  • Projective measurement: |ψ'⟩ = P|ψ⟩ / ‖P|ψ⟩‖  (state collapse)
  • Quantum interference: P(A∪B) = P(A) + P(B) + 2·Re(⟨ψ|P_A P_B|ψ⟩)
  • Order effects:  P(A→B) ≠ P(B→A)  iff  [P_A, P_B] ≠ 0
  • Mixed states (density matrix): ρ = Σᵢ pᵢ|ψᵢ⟩⟨ψᵢ|
  • Decoherence: ρᵢⱼ(t) = ρᵢⱼ(0)·exp(−γt), i≠j  (Lindblad dephasing)
  • Unitary evolution: U(t) = e^(−iHt), H Hermitian  (Schrödinger)
  • Quantum walk: O(N) spreading in N steps vs O(√N) classical
  • Entanglement: Schmidt decomposition, entanglement entropy

Cognitive phenomena explained:
  • Conjunction fallacy  (Tversky & Kahneman 1983)
  • Question order effects  (Wang & Busemeyer 2013)
  • Disjunction effect / Sure-Thing violation  (Tversky & Shafir 1992)
  • Similarity-based memory interference
  • Context-dependent concept activation

Integration: QuantumCognitionEngine plugs directly into A_Brain.PreciousBrain.

References:
  Busemeyer & Bruza (2012). Quantum Models of Cognition and Decision.
  Pothos & Busemeyer (2009). A quantum probability explanation for violations
    of rational decision theory. Proc. R. Soc. B 276, 2171–2178.
  Khrennikov (2010). Ubiquitous Quantum Structure. Springer.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ── Utility: Hermitian matrix exponential ─────────────────────────────────────

def _hermitian_expm(A: np.ndarray) -> np.ndarray:
    """e^A for Hermitian A via eigendecomposition (exact, no scipy needed)."""
    eigvals, eigvecs = np.linalg.eigh(A)
    return (eigvecs * np.exp(eigvals)) @ eigvecs.conj().T


def _hermitian_logm(A: np.ndarray) -> np.ndarray:
    """log(A) for Hermitian positive-definite A via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-300)
    return (eigvecs * np.log(eigvals)) @ eigvecs.conj().T


# ══════════════════════════════════════════════════════════════════════════════
# 1.  QUANTUM STATE  |ψ⟩ ∈ ℂⁿ
# ══════════════════════════════════════════════════════════════════════════════

class QuantumState:
    """
    Pure quantum cognitive state |ψ⟩ — a unit vector in ℂⁿ.

    Represents the brain simultaneously entertaining a superposition of
    cognitive hypotheses. Each component ψᵢ is a complex probability
    amplitude; |ψᵢ|² is the Born-rule probability for outcome i.

    Key properties
    ──────────────
    • Normalization:  ‖|ψ⟩‖ = 1   (probabilities sum to 1)
    • Inner product:  ⟨φ|ψ⟩ = Σᵢ φᵢ* ψᵢ   (complex overlap)
    • Fidelity:       F = |⟨φ|ψ⟩|²  ∈ [0,1]  (state similarity)
    """

    __slots__ = ('_psi', '_n')

    def __init__(self, amplitudes: np.ndarray):
        self._psi = np.asarray(amplitudes, dtype=np.complex128).ravel()
        self._n   = len(self._psi)
        norm = np.sqrt(np.real(self._psi.conj() @ self._psi))
        if norm < 1e-14:
            raise ValueError("Zero vector cannot be a quantum state.")
        self._psi /= norm

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_real(cls, v: np.ndarray) -> 'QuantumState':
        """Embed a real neural activation vector into ℂⁿ."""
        return cls(v.astype(np.complex128))

    @classmethod
    def uniform(cls, n: int) -> 'QuantumState':
        """(1/√n) Σᵢ |i⟩ — maximum-uncertainty / maximum-superposition state."""
        return cls(np.ones(n, dtype=np.complex128) / np.sqrt(n))

    @classmethod
    def basis(cls, n: int, i: int) -> 'QuantumState':
        """Computational basis state |i⟩."""
        psi = np.zeros(n, dtype=np.complex128)
        psi[i] = 1.0
        return cls(psi)

    # ── Core operations ───────────────────────────────────────────────────────

    @property
    def vector(self) -> np.ndarray:
        return self._psi.copy()

    @property
    def dim(self) -> int:
        return self._n

    def inner(self, other: 'QuantumState') -> complex:
        """⟨self|other⟩ = Σᵢ self_ψᵢ* · other_ψᵢ"""
        return complex(self._psi.conj() @ other._psi)

    def probabilities(self) -> np.ndarray:
        """Born rule: P(i) = |ψᵢ|²   for all i."""
        return np.real(self._psi.conj() * self._psi)

    def measure(self, P: np.ndarray) -> Tuple[float, 'QuantumState']:
        """
        Projective measurement with projector P (P² = P, P = P†).

        Returns (probability, post-measurement state):
          prob       = ⟨ψ|P|ψ⟩
          collapsed  = P|ψ⟩ / ‖P|ψ⟩‖
        """
        projected = P @ self._psi
        prob = float(np.real(self._psi.conj() @ projected))
        if prob < 1e-14:
            return 0.0, self
        return prob, QuantumState(projected / np.sqrt(prob))

    def evolve(self, U: np.ndarray) -> 'QuantumState':
        """Unitary evolution: |ψ'⟩ = U|ψ⟩."""
        return QuantumState(U @ self._psi)

    def density_matrix(self) -> np.ndarray:
        """ρ = |ψ⟩⟨ψ| — outer product (pure state density matrix)."""
        return np.outer(self._psi, self._psi.conj())

    def fidelity(self, other: 'QuantumState') -> float:
        """F = |⟨self|other⟩|² ∈ [0,1]"""
        return float(abs(self.inner(other)) ** 2)

    def expected_value(self, observable: np.ndarray) -> float:
        """⟨O⟩ = ⟨ψ|O|ψ⟩"""
        return float(np.real(self._psi.conj() @ observable @ self._psi))

    def __repr__(self) -> str:
        p = self.probabilities()
        top = np.argsort(p)[-3:][::-1]
        s = ', '.join(f'|{i}⟩:{p[i]:.3f}' for i in top)
        return f'QuantumState(n={self._n}, top3=[{s}])'


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DENSITY MATRIX  ρ  (mixed states + decoherence)
# ══════════════════════════════════════════════════════════════════════════════

class DensityMatrix:
    """
    Mixed quantum cognitive state ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|.

    A pure state has Tr(ρ²) = 1 (zero entropy, maximum coherence).
    A maximally mixed state has ρ = I/n (maximum entropy, no coherence).

    Decoherence (environmental interaction) drives off-diagonal elements
    ρᵢⱼ → 0, converting quantum superposition into classical probability.

    Von Neumann entropy: S = −Tr(ρ log ρ) = −Σᵢ λᵢ log λᵢ
      S = 0   → pure quantum state
      S = logn → maximally mixed (classical ignorance)
    """

    def __init__(self, matrix: np.ndarray):
        rho = np.asarray(matrix, dtype=np.complex128)
        # Force Hermitian
        rho = (rho + rho.conj().T) / 2
        # Force positive semidefinite
        eigvals, eigvecs = np.linalg.eigh(rho)
        eigvals = np.maximum(eigvals, 0.0)
        rho = (eigvecs * eigvals) @ eigvecs.conj().T
        # Force unit trace
        tr = np.real(np.trace(rho))
        self._rho = rho / tr if tr > 1e-14 else rho
        self._n   = rho.shape[0]

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_pure(cls, state: QuantumState) -> 'DensityMatrix':
        return cls(state.density_matrix())

    @classmethod
    def from_ensemble(cls, states: List[QuantumState],
                      weights: List[float]) -> 'DensityMatrix':
        """ρ = Σᵢ wᵢ |ψᵢ⟩⟨ψᵢ|  (wᵢ auto-normalized)."""
        w = np.asarray(weights, dtype=float)
        w /= w.sum()
        rho = sum(wi * s.density_matrix() for wi, s in zip(w, states))
        return cls(rho)

    @classmethod
    def maximally_mixed(cls, n: int) -> 'DensityMatrix':
        """ρ = I/n — maximum ignorance."""
        return cls(np.eye(n, dtype=np.complex128) / n)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def matrix(self) -> np.ndarray:
        return self._rho.copy()

    @property
    def dim(self) -> int:
        return self._n

    def purity(self) -> float:
        """Tr(ρ²) ∈ [1/n, 1].  1 = pure, 1/n = maximally mixed."""
        return float(np.real(np.trace(self._rho @ self._rho)))

    def von_neumann_entropy(self) -> float:
        """S = −Tr(ρ log ρ) = −Σᵢ λᵢ log λᵢ  (nats, base-e)."""
        eigvals = np.linalg.eigvalsh(self._rho)
        eigvals = eigvals[eigvals > 1e-300]
        return float(-np.sum(eigvals * np.log(eigvals)))

    def coherence(self) -> float:
        """
        l₁-norm quantum coherence: C = Σᵢ≠ⱼ |ρᵢⱼ|.
        Measures total off-diagonal amplitude — how much superposition remains.
        C = 0 means fully dephased (classical mixture).
        """
        off = self._rho - np.diag(np.diag(self._rho))
        return float(np.sum(np.abs(off)))

    def relative_purity(self) -> float:
        """Normalized purity ∈ [0,1]: (Tr(ρ²) − 1/n) / (1 − 1/n)."""
        p   = self.purity()
        min_p = 1.0 / self._n
        if abs(1.0 - min_p) < 1e-14:
            return 1.0
        return float(np.clip((p - min_p) / (1.0 - min_p), 0.0, 1.0))

    # ── Operations ────────────────────────────────────────────────────────────

    def measure(self, P: np.ndarray) -> Tuple[float, 'DensityMatrix']:
        """
        Projective measurement outcome P:
          prob  = Tr(Pρ)
          ρ'    = PρP† / Tr(Pρ)
        """
        prob = float(np.real(np.trace(P @ self._rho)))
        if prob < 1e-14:
            return 0.0, self
        rho_post = P @ self._rho @ P.conj().T / prob
        return prob, DensityMatrix(rho_post)

    def evolve(self, U: np.ndarray) -> 'DensityMatrix':
        """Unitary evolution: ρ' = U ρ U†."""
        return DensityMatrix(U @ self._rho @ U.conj().T)

    def decohere(self, gamma: float, dt: float) -> 'DensityMatrix':
        """
        Pure dephasing channel (Lindblad limit):
          ρᵢⱼ(t+dt) = ρᵢⱼ(t) · exp(−γ · dt)   for i ≠ j
          ρᵢᵢ unchanged  (populations are conserved).

        gamma : dephasing rate (s⁻¹ in cognitive time units)
        dt    : elapsed cognitive time
        """
        decay = np.exp(-gamma * dt)
        mask  = np.full((self._n, self._n), decay, dtype=np.complex128)
        np.fill_diagonal(mask, 1.0)
        return DensityMatrix(self._rho * mask)

    def partial_trace_A(self, dim_a: int) -> 'DensityMatrix':
        """
        Partial trace over subsystem A (dim_a) → reduced ρ_B.

        For ρ ∈ ℂ^(dₐ·d_b) × ℂ^(dₐ·d_b):
          ρ_B[j,l] = Σᵢ ρ[(i,j),(i,l)]

        Computed exactly via reshape + np.trace with axis selection.
        """
        dim_b = self._n // dim_a
        rho_r = self._rho.reshape(dim_a, dim_b, dim_a, dim_b)
        rho_b = np.trace(rho_r, axis1=0, axis2=2)
        return DensityMatrix(rho_b)

    def expected_value(self, O: np.ndarray) -> float:
        """⟨O⟩ = Tr(ρ O)."""
        return float(np.real(np.trace(self._rho @ O)))

    def entanglement_entropy(self, dim_a: int) -> float:
        """S(ρ_B) — entanglement entropy via partial trace."""
        return self.partial_trace_A(dim_a).von_neumann_entropy()

    def __repr__(self) -> str:
        return (f'DensityMatrix(n={self._n}, '
                f'purity={self.purity():.4f}, '
                f'S={self.von_neumann_entropy():.4f}, '
                f'C={self.coherence():.4f})')


# ══════════════════════════════════════════════════════════════════════════════
# 3.  QUANTUM INTERFERENCE  (cognitive biases from non-classical probability)
# ══════════════════════════════════════════════════════════════════════════════

class QuantumInterference:
    """
    Static methods for computing quantum interference effects in cognition.

    Three experimentally-confirmed phenomena:

    1. Conjunction Fallacy (Tversky & Kahneman 1983)
       "Linda is more likely to be a feminist bank teller than a bank teller."
       Classical impossible: P(A∩B) ≤ min(P(A), P(B)).
       Quantum possible: interference term inflates P(A∩B).

    2. Question Order Effect (Wang & Busemeyer 2013)
       Surveys show P(A then B) ≠ P(B then A).
       Quantum cause: [P_A, P_B] ≠ 0 — incompatible observables.

    3. Disjunction Effect (Tversky & Shafir 1992)
       Sure-Thing Principle violated: act in both cases, don't act when unknown.
       Quantum cause: interference between win/lose reduces act probability.
    """

    @staticmethod
    def rank1_projector(v: np.ndarray) -> np.ndarray:
        """P = |v⟩⟨v| / ⟨v|v⟩  — rank-1 projector onto direction v."""
        v = np.asarray(v, dtype=np.complex128).ravel()
        v = v / np.linalg.norm(v)
        return np.outer(v, v.conj())

    @staticmethod
    def subspace_projector(V: np.ndarray) -> np.ndarray:
        """P = V(V†V)⁻¹V† — orthogonal projector onto column span of V."""
        V = np.asarray(V, dtype=np.complex128)
        Q, _ = np.linalg.qr(V)
        k = Q.shape[1]
        return Q[:, :k] @ Q[:, :k].conj().T

    @staticmethod
    def conjunction(
        state: QuantumState,
        proj_a: np.ndarray,
        proj_b: np.ndarray,
    ) -> Dict:
        """
        Compute conjunction probabilities (classical vs quantum).

        Classical (Kolmogorov): P(A∩B) ≤ min(P(A), P(B))
        Quantum sequential measurement A then B:
          P(A→B) = ‖P_B P_A |ψ⟩‖²

        Interference term T = P_quantum(A∩B) − P(A)·P(B)
          T > 0 → overestimation bias (positive interference)
          T < 0 → underestimation bias (negative interference)
        """
        ψ = state.vector

        p_a = float(np.real(ψ.conj() @ proj_a @ ψ))
        p_b = float(np.real(ψ.conj() @ proj_b @ ψ))

        # Quantum sequential P(A then B) = ‖P_B P_A |ψ⟩‖²
        pa_psi = proj_a @ ψ
        pa_norm = np.linalg.norm(pa_psi)
        if pa_norm < 1e-14:
            p_ab_q = 0.0
        else:
            pa_psi_n = pa_psi / pa_norm
            p_ab_q = p_a * float(np.real(pa_psi_n.conj() @ proj_b @ pa_psi_n))

        p_ab_classical = p_a * p_b  # independence baseline

        # Quantum interference term from anti-commutator
        t = float(np.real(ψ.conj() @ (proj_a @ proj_b + proj_b @ proj_a) @ ψ)) / 2
        interference = t - p_ab_classical

        return {
            'P(A)':              p_a,
            'P(B)':              p_b,
            'P(A∩B)_classical':  p_ab_classical,
            'P(A→B)_quantum':    p_ab_q,
            'interference_term': interference,
            'conjunction_fallacy': p_ab_q > min(p_a, p_b),
        }

    @staticmethod
    def order_effect(
        state: QuantumState,
        proj_a: np.ndarray,
        proj_b: np.ndarray,
    ) -> Dict:
        """
        Compute P(A→B) vs P(B→A) and test classical compatibility.

        P(A→B) = ‖P_B P_A |ψ⟩‖²
        P(B→A) = ‖P_A P_B |ψ⟩‖²
        Order effect δ = P(A→B) − P(B→A)
          δ = 0  iff  [P_A, P_B] = 0  (classically compatible questions)
          δ ≠ 0  quantum signature of incompatible observables

        QQ-equality (Wang & Busemeyer): P(A→B) + P(Ā→B) = P(B→A) + P(B→Ā)
        Tested here as a quantum validity check.
        """
        ψ = state.vector
        n = len(ψ)
        I = np.eye(n, dtype=np.complex128)

        p_ab = float(np.real(
            ψ.conj() @ proj_a.conj().T @ proj_b.conj().T @ proj_b @ proj_a @ ψ
        ))
        p_ba = float(np.real(
            ψ.conj() @ proj_b.conj().T @ proj_a.conj().T @ proj_a @ proj_b @ ψ
        ))

        # Complement projectors
        comp_a = I - proj_a
        comp_b = I - proj_b

        # QQ-equality test
        p_ab_bar   = float(np.real(ψ.conj() @ comp_a.T.conj() @ proj_b.T.conj() @ proj_b @ comp_a @ ψ))
        p_ba_bar   = float(np.real(ψ.conj() @ comp_b.T.conj() @ proj_a.T.conj() @ proj_a @ comp_b @ ψ))
        qq_lhs     = p_ab + p_ab_bar
        qq_rhs     = p_ba + p_ba_bar
        qq_satisfied = abs(qq_lhs - qq_rhs) < 1e-8

        # Commutator Frobenius norm: ‖[P_A, P_B]‖_F
        comm      = proj_a @ proj_b - proj_b @ proj_a
        comm_norm = float(np.linalg.norm(comm, 'fro'))

        return {
            'P(A→B)':            p_ab,
            'P(B→A)':            p_ba,
            'order_effect':      p_ab - p_ba,
            'commutator_norm':   comm_norm,
            'compatible':        comm_norm < 1e-8,
            'QQ_equality_holds': qq_satisfied,
        }

    @staticmethod
    def disjunction_effect(
        state: QuantumState,
        proj_win:  np.ndarray,
        proj_lose: np.ndarray,
        proj_act:  np.ndarray,
    ) -> Dict:
        """
        Test violation of the Sure-Thing Principle.

        Classical logic:
          P(act|win) > 0.5 AND P(act|lose) > 0.5  →  P(act|unknown) > 0.5
        Quantum: interference between win/lose superposition can violate this.

        When P(act|unknown) < min(P(act|win), P(act|lose)),
        destructive interference suppresses rational action — the Disjunction Effect.
        """
        ψ = state.vector

        def conditional(P_cond: np.ndarray, P_event: np.ndarray) -> float:
            p_c = float(np.real(ψ.conj() @ P_cond @ ψ))
            if p_c < 1e-14:
                return 0.0
            ψ_post = P_cond @ ψ / np.sqrt(p_c)
            return float(np.real(ψ_post.conj() @ P_event @ ψ_post))

        p_act_win  = conditional(proj_win,  proj_act)
        p_act_lose = conditional(proj_lose, proj_act)
        p_act_unk  = float(np.real(ψ.conj() @ proj_act @ ψ))

        lb = min(p_act_win, p_act_lose)
        ub = max(p_act_win, p_act_lose)
        violated = not (lb - 1e-8 <= p_act_unk <= ub + 1e-8)

        return {
            'P(act|win)':           p_act_win,
            'P(act|lose)':          p_act_lose,
            'P(act|unknown)':       p_act_unk,
            'sure_thing_violated':  violated,
            'interference_suppression': lb - p_act_unk if violated else 0.0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  QUANTUM MEMORY BANK  (superposed associative memory)
# ══════════════════════════════════════════════════════════════════════════════

class QuantumMemoryBank:
    """
    Quantum associative memory — patterns stored as superposed quantum states.

    Classical Hopfield: weights W = (1/K) Σ_μ ξ^μ (ξ^μ)ᵀ
    Quantum version:    superposition |M⟩ = (1/√K) Σ_μ s_μ e^(iφ_μ) |ξ^μ⟩

    Encoding:
      Each pattern |ξ^μ⟩ is stored with a phase φ_μ (emotion-tagged) and
      a strength s_μ. The full memory is a superposition of all patterns.

    Retrieval:
      Direct:       F_μ = |⟨ξ^μ|q⟩|²  (fidelity with each pattern)
      Interference: F_int = |⟨M|q⟩|²   (fidelity with full superposition)
      F_int > F_direct → constructive interference aids recall
      F_int < F_direct → destructive interference suppresses recall

    Entanglement (Schmidt decomposition):
      Reshape |M⟩ as √dₐ × √d_b matrix, compute SVD.
      E_S = −Σᵢ λᵢ² log λᵢ²  (entanglement entropy of memory superposition)
      High E_S → memories are strongly entangled (rich associative structure).
    """

    def __init__(self, dim: int, capacity: int = 512):
        self._dim      = dim
        self._capacity = capacity
        self._patterns:  List[np.ndarray] = []
        self._phases:    List[float]      = []
        self._labels:    List[str]        = []
        self._strengths: List[float]      = []
        self._superposition: Optional[np.ndarray] = None
        self._dirty = True

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(
        self,
        pattern:  np.ndarray,
        label:    str   = '',
        strength: float = 1.0,
        phase:    Optional[float] = None,
    ) -> int:
        """
        Store pattern as normalized quantum state |ξ⟩.
        Phase φ is the quantum label (random if not specified).
        Returns index of stored pattern; -1 if pattern is zero.
        """
        v = np.asarray(pattern, dtype=np.complex128).ravel()
        if len(v) < self._dim:
            v = np.pad(v, (0, self._dim - len(v)))
        elif len(v) > self._dim:
            v = v[:self._dim]
        norm = np.linalg.norm(v)
        if norm < 1e-14:
            return -1
        v = v / norm

        if phase is None:
            phase = float(2 * np.pi * np.random.random())

        self._patterns.append(v)
        self._phases.append(float(phase))
        self._labels.append(label)
        self._strengths.append(float(strength))
        self._dirty = True

        # Evict weakest pattern if over capacity
        if len(self._patterns) > self._capacity:
            i = int(np.argmin(self._strengths))
            for lst in (self._patterns, self._phases, self._labels, self._strengths):
                lst.pop(i)

        return len(self._patterns) - 1

    # ── Superposition ─────────────────────────────────────────────────────────

    def _build_superposition(self):
        """
        |M⟩ = (1/Z) Σ_μ s_μ e^(iφ_μ) |ξ^μ⟩
        where Z normalizes the result.
        """
        if not self._patterns:
            self._superposition = None
            self._dirty = False
            return
        M = np.zeros(self._dim, dtype=np.complex128)
        for v, phi, s in zip(self._patterns, self._phases, self._strengths):
            M += s * np.exp(1j * phi) * v
        norm = np.linalg.norm(M)
        self._superposition = M / norm if norm > 1e-14 else M
        self._dirty = False

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def recall(
        self,
        query:           np.ndarray,
        top_k:           int  = 5,
        use_interference: bool = True,
    ) -> List[Dict]:
        """
        Retrieve top_k patterns closest to query.

        Each result contains:
          label, fidelity (direct Born-rule), interference_fidelity,
          interference_gain = F_int − F_direct (positive = constructive).
        """
        if not self._patterns:
            return []

        q = np.asarray(query, dtype=np.complex128).ravel()
        if len(q) < self._dim:
            q = np.pad(q, (0, self._dim - len(q)))
        elif len(q) > self._dim:
            q = q[:self._dim]
        norm = np.linalg.norm(q)
        if norm < 1e-14:
            return []
        q = q / norm

        # Direct fidelity: |⟨ξ^μ|q⟩|²
        fidelities = np.array([
            float(abs(v.conj() @ q) ** 2)
            for v in self._patterns
        ])

        if use_interference and self._dirty:
            self._build_superposition()

        k     = min(top_k, len(self._patterns))
        idxs  = np.argsort(fidelities)[-k:][::-1]
        results = []

        for rank, idx in enumerate(idxs):
            r = {
                'label':    self._labels[idx],
                'fidelity': float(fidelities[idx]),
                'strength': float(self._strengths[idx]),
                'phase':    float(self._phases[idx]),
                'rank':     rank,
            }
            if use_interference and self._superposition is not None:
                f_int = float(abs(self._superposition.conj() @ q) ** 2)
                r['interference_fidelity'] = f_int
                r['interference_gain']     = f_int - float(fidelities[idx])
            results.append(r)

        return results

    # ── Entanglement & capacity ───────────────────────────────────────────────

    def entanglement_entropy(self) -> float:
        """
        Entanglement entropy of the memory superposition via Schmidt decomposition.

        |M⟩ reshaped as (√dim × √dim) matrix; Schmidt coefficients from SVD.
        E_S = −Σᵢ λᵢ² log λᵢ²  (0 = product state, log(k) = maximally entangled)
        """
        if self._dirty:
            self._build_superposition()
        if self._superposition is None:
            return 0.0

        sqrt_dim = int(np.sqrt(self._dim))
        if sqrt_dim * sqrt_dim != self._dim:
            # Non-square: split as (2 × dim//2)
            split_a, split_b = 2, self._dim // 2
        else:
            split_a = split_b = sqrt_dim

        M  = self._superposition[:split_a * split_b].reshape(split_a, split_b)
        sv = np.linalg.svd(M, compute_uv=False)
        lam2 = (sv ** 2)
        lam2_sum = lam2.sum()
        if lam2_sum < 1e-14:
            return 0.0
        lam2 = lam2 / lam2_sum
        lam2 = lam2[lam2 > 1e-300]
        return float(-np.sum(lam2 * np.log(lam2)))

    def gram_matrix(self) -> np.ndarray:
        """G[μ,ν] = |⟨ξ^μ|ξ^ν⟩|² — pairwise fidelities between stored patterns."""
        K = len(self._patterns)
        if K == 0:
            return np.zeros((0, 0))
        P = np.stack(self._patterns)           # (K, dim)
        inner = P @ P.conj().T                 # (K, K)
        return np.real(inner.conj() * inner)   # element-wise |·|²

    def effective_capacity(self) -> float:
        """
        Effective number of distinguishable memories.
        Computed as inverse participation ratio of gram matrix eigenspectrum:
          C_eff = 1 / Σᵢ (λᵢ / Σλ)²
        """
        K = len(self._patterns)
        if K < 2:
            return float(K)
        G    = self.gram_matrix()
        eigs = np.maximum(np.linalg.eigvalsh(G), 0)
        eigs /= eigs.sum() + 1e-14
        return float(1.0 / (np.sum(eigs ** 2) + 1e-14))

    def __len__(self) -> int:
        return len(self._patterns)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  QUANTUM WALK  (spreading activation with quantum interference)
# ══════════════════════════════════════════════════════════════════════════════

class QuantumWalk:
    """
    Discrete-time quantum walk on a cycle graph of N nodes.

    Classical random walk: Gaussian spreading — std ∝ √t
    Quantum walk:          Ballistic spreading — std ∝ t  (quadratic speedup)

    The quantum walk models how semantic activation spreads through a
    concept network, with interference creating peaks and troughs
    (constructive: concept highlighted; destructive: concept suppressed).

    State space: coin ⊗ position = ℂ² ⊗ ℂᴺ
      |0,x⟩ → leftward coin, position x
      |1,x⟩ → rightward coin, position x

    Coin: Hadamard  H₂ = (1/√2)[[1,1],[1,−1]]
    Shift: |0,x⟩ → |0,(x−1) mod N⟩,  |1,x⟩ → |1,(x+1) mod N⟩
    Step: U = S · (H₂ ⊗ Iₙ)
    """

    def __init__(self, n_nodes: int):
        self._n    = n_nodes
        self._dim  = 2 * n_nodes
        self._U    = self._build_step_operator()
        self._psi  = self._symmetric_start(0)

    def _build_step_operator(self) -> np.ndarray:
        N = self._n
        H2 = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        C  = np.kron(H2, np.eye(N, dtype=np.complex128))

        S = np.zeros((2 * N, 2 * N), dtype=np.complex128)
        for x in range(N):
            S[0 * N + (x - 1) % N, 0 * N + x] = 1.0   # coin=0: move left
            S[1 * N + (x + 1) % N, 1 * N + x] = 1.0   # coin=1: move right
        return S @ C

    def _symmetric_start(self, node: int) -> np.ndarray:
        """(|0,node⟩ + i|1,node⟩)/√2 — symmetric start avoids drift."""
        psi = np.zeros(self._dim, dtype=np.complex128)
        psi[0 * self._n + node % self._n] = 1.0 / np.sqrt(2)
        psi[1 * self._n + node % self._n] = 1j  / np.sqrt(2)
        return psi

    def reset(self, start_node: Optional[int] = None):
        node = (start_node or 0) % self._n
        self._psi = self._symmetric_start(node)

    def step(self, n_steps: int = 1):
        """Advance the walk by n_steps: |ψ⟩ → Uⁿ|ψ⟩."""
        for _ in range(n_steps):
            self._psi = self._U @ self._psi

    def position_distribution(self) -> np.ndarray:
        """P(x) = |⟨0,x|ψ⟩|² + |⟨1,x|ψ⟩|²  — marginal over coin."""
        N = self._n
        return np.abs(self._psi[:N]) ** 2 + np.abs(self._psi[N:]) ** 2

    def interference_pattern(self) -> np.ndarray:
        """
        Cᵢ(x) = 2|Re(ψ₀*(x) · ψ₁(x))| — coin interference per node.
        Positive peak = constructive (concept amplified).
        """
        N = self._n
        return 2.0 * np.abs(np.real(self._psi[:N].conj() * self._psi[N:]))

    def spreading_entropy(self) -> float:
        """Shannon entropy of P(x) — measures how spread the activation is."""
        p = self.position_distribution()
        p = p[p > 1e-300]
        return float(-np.sum(p * np.log(p)))

    def position_std(self) -> float:
        """Standard deviation of position distribution (linear, not wrapped)."""
        p    = self.position_distribution()
        xs   = np.arange(self._n, dtype=float)
        mean = float(xs @ p)
        return float(np.sqrt(xs @ (p * xs) - mean ** 2))

    def quantum_advantage(self, n_steps: int) -> float:
        """
        Ratio of quantum to classical spreading (std-based).

        Quantum walk: std ∝ t  (ballistic)
        Classical walk: std ∝ √t  (diffusive)
        Advantage = σ_quantum / √(n_steps)
        Expected ≈ 1/√2 · √(n_steps) → advantage ≈ √(n_steps) / √2
        Normalised so advantage=1 means no quantum benefit.
        """
        classical_std = np.sqrt(max(n_steps, 1))
        quantum_std   = self.position_std()
        return float(quantum_std / (classical_std + 1e-14))

    def top_activated_nodes(self, k: int = 5) -> np.ndarray:
        return np.argsort(self.position_distribution())[-k:][::-1]


# ══════════════════════════════════════════════════════════════════════════════
# 6.  COGNITIVE HAMILTONIAN  (continuous quantum evolution)
# ══════════════════════════════════════════════════════════════════════════════

class CognitiveHamiltonian:
    """
    Hermitian operator H governing continuous quantum cognitive evolution.

    The Hamiltonian encodes the cognitive energy landscape:
      • Diagonal: Eᵢ = energy of cognitive basis state |i⟩
      • Off-diagonal: Hᵢⱼ = coupling strength between states i and j
      • Low-energy eigenstates = stable cognitive attractors
      • High-energy states = unstable, transient cognitive patterns

    Schrödinger equation: d|ψ⟩/dt = −iH|ψ⟩  (ℏ = 1)
    Solution: |ψ(t)⟩ = e^(−iHt)|ψ(0)⟩  (unitary, norm-preserving)
    """

    def __init__(self, n: int):
        self._n = n
        self._H: Optional[np.ndarray] = None

    @classmethod
    def from_activation(cls, activation: np.ndarray) -> 'CognitiveHamiltonian':
        """
        Build H from a neural activation vector a:
          H = (|a⟩⟨a| + |a⟩⟨a|†) / 2  →  symmetrized rank-1 operator
        Represents a Hamiltonian whose ground state = normalized activation.
        """
        a = np.asarray(activation, dtype=np.complex128).ravel()
        n = len(a)
        h = cls(n)
        a_norm = a / (np.linalg.norm(a) + 1e-14)
        H_raw  = np.outer(a_norm, a_norm.conj())
        h._H   = (H_raw + H_raw.conj().T) / 2
        return h

    @classmethod
    def from_weight_matrix(cls, W: np.ndarray) -> 'CognitiveHamiltonian':
        """
        Build H from neural weight matrix W (symmetrized, eigenvalue-normalized).
        H = (W + W†)/2  then scaled so eigenvalues ∈ [−π, π].
        """
        n = W.shape[0]
        h = cls(n)
        sym   = (W + W.conj().T) / 2
        eigs  = np.linalg.eigvalsh(sym)
        spread = eigs.max() - eigs.min()
        if spread > 1e-14:
            sym = (sym - eigs.min() * np.eye(n)) / spread * (2 * np.pi) - np.pi * np.eye(n)
        h._H = sym
        return h

    @classmethod
    def from_gue(cls, n: int, seed: Optional[int] = None) -> 'CognitiveHamiltonian':
        """Random Hermitian from the Gaussian Unitary Ensemble (GUE)."""
        rng = np.random.default_rng(seed)
        A   = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        h   = cls(n)
        h._H = (A + A.conj().T) / (2.0 * np.sqrt(2 * n))
        return h

    def unitary(self, t: float) -> np.ndarray:
        """U(t) = e^(−iHt) — exact via eigendecomposition of H."""
        if self._H is None:
            return np.eye(self._n, dtype=np.complex128)
        return _hermitian_expm(-1j * self._H * t)

    def energy_levels(self) -> np.ndarray:
        """Eigenvalues of H — the cognitive energy spectrum."""
        if self._H is None:
            return np.zeros(self._n)
        return np.linalg.eigvalsh(self._H)

    def ground_state(self) -> QuantumState:
        """Lowest-energy eigenstate — the dominant cognitive attractor."""
        if self._H is None:
            return QuantumState.basis(self._n, 0)
        _, vecs = np.linalg.eigh(self._H)
        return QuantumState(vecs[:, 0])

    def spectral_gap(self) -> float:
        """E₁ − E₀: energy gap between ground and first excited state."""
        eigs = self.energy_levels()
        return float(eigs[1] - eigs[0]) if len(eigs) >= 2 else 0.0

    def thermodynamic_entropy(self, beta: float = 1.0) -> float:
        """
        Von Neumann entropy of thermal state ρ = e^(−βH) / Tr(e^(−βH)).
        beta = 1/kT — inverse cognitive 'temperature'.
        """
        if self._H is None:
            return float(np.log(self._n))
        eigs   = self.energy_levels()
        boltz  = np.exp(-beta * (eigs - eigs.min()))
        boltz /= boltz.sum()
        boltz  = boltz[boltz > 1e-300]
        return float(-np.sum(boltz * np.log(boltz)))


# ══════════════════════════════════════════════════════════════════════════════
# 7.  QUANTUM COGNITION ENGINE  (master integration layer)
# ══════════════════════════════════════════════════════════════════════════════

class QuantumCognitionEngine:
    """
    Full quantum cognition engine — integrates all quantum modules.

    Hooks into A_Brain.PreciousBrain:
      encode_neural(activations)    → embed neural state → QuantumState
      quantum_decide(options)       → Born-rule decision with interference
      quantum_memorize(pattern)     → phase-encoded quantum memory
      quantum_recall(query)         → interference-based retrieval
      decohere(dt)                  → environmental decoherence step
      evolve(dt)                    → unitary Hamiltonian evolution
      spread_activation(node, t)   → quantum walk on concept graph
      quantum_preciousness()        → 5-axis quantum score (0–100)
      get_state_report()            → full metrics snapshot
    """

    # Emotion → encoding phase (radians)
    _EMOTION_PHASES = {
        'happiness': 0.0,         'love':    np.pi / 4,
        'surprise':  np.pi / 2,   'neutral': np.pi,
        'sadness':   3 * np.pi / 4, 'fear':  5 * np.pi / 6,
        'anger':     7 * np.pi / 6, 'disgust': 3 * np.pi / 2,
    }

    def __init__(
        self,
        hilbert_dim:      int   = 64,
        decoherence_rate: float = 0.05,
        n_walk_nodes:     int   = 32,
        memory_capacity:  int   = 512,
    ):
        self._dim   = hilbert_dim
        self._gamma = decoherence_rate

        # Quantum state (pure and mixed representations kept in sync)
        self.state:   QuantumState  = QuantumState.uniform(hilbert_dim)
        self.density: DensityMatrix = DensityMatrix.from_pure(self.state)

        # Sub-modules
        self.memory:      QuantumMemoryBank    = QuantumMemoryBank(hilbert_dim, memory_capacity)
        self.walk:        QuantumWalk          = QuantumWalk(n_walk_nodes)
        self.hamiltonian: CognitiveHamiltonian = CognitiveHamiltonian.from_gue(hilbert_dim, seed=42)

        # Accumulators for preciousness scoring
        self._n_decisions        = 0
        self._total_interference = 0.0  # sum of |interference_term| across decisions
        self._n_decoherence      = 0
        self._t                  = 0.0  # cognitive time

    # ── Neural state encoding ─────────────────────────────────────────────────

    def encode_neural(self, activations: np.ndarray) -> QuantumState:
        """
        Embed a real neural activation vector into ℂⁿ via analytic signal.

        Method: construct the analytic signal z = a + i·H[a]
          where H[a] is the discrete Hilbert transform of a (via FFT phase shift).
          This embeds the activation into the complex plane while preserving
          all frequency content; similar activations yield high fidelity.

        The resulting |ψ⟩ has the property:
          |⟨ψ_a|ψ_b⟩|² ≈ cosine_similarity(a, b)² for real inputs.
        """
        a = np.asarray(activations, dtype=float).ravel()
        # Pad or truncate to Hilbert dim
        if len(a) < self._dim:
            a = np.pad(a, (0, self._dim - len(a)))
        else:
            a = a[:self._dim]

        # Discrete Hilbert transform via FFT
        A   = np.fft.rfft(a)
        hf  = np.ones(len(A), dtype=np.complex128)
        hf[1:] = -1j   # multiply positive frequencies by −i → +π/2 phase shift
        Ha  = np.fft.irfft(A * hf, n=self._dim)

        # Analytic signal embedding
        psi_c = a.astype(np.complex128) + 1j * Ha

        self.state   = QuantumState(psi_c)
        self.density = DensityMatrix.from_pure(self.state)

        # Update Hamiltonian from new cognitive state
        self.hamiltonian = CognitiveHamiltonian.from_activation(psi_c)

        return self.state

    # ── Quantum decision making ───────────────────────────────────────────────

    def quantum_decide(
        self,
        options:       List[np.ndarray],
        option_labels: Optional[List[str]] = None,
    ) -> Dict:
        """
        Make a decision between N options using Born-rule measurement.

        Each option vector |oᵢ⟩ defines a rank-1 projector Pᵢ = |oᵢ⟩⟨oᵢ|.
        Decision probability: P(i) = |⟨oᵢ|ψ⟩|²  (Born rule)

        Interference between options i,j (quantum cross-term):
          I[i,j] = 2·Re(⟨ψ|oᵢ⟩ · ⟨oᵢ|oⱼ⟩ · ⟨oⱼ|ψ⟩)
          Non-zero when options are non-orthogonal — pure quantum effect.

        Order effect tested between first two options.
        State collapses to chosen option after decision.
        """
        n_opts = len(options)
        if option_labels is None:
            option_labels = [f'option_{i}' for i in range(n_opts)]

        # Build normalized option vectors and projectors
        opt_vecs = []
        for o in options:
            v = np.asarray(o, dtype=np.complex128).ravel()
            if len(v) < self._dim:
                v = np.pad(v, (0, self._dim - len(v)))
            elif len(v) > self._dim:
                v = v[:self._dim]
            nrm = np.linalg.norm(v)
            opt_vecs.append(v / nrm if nrm > 1e-14 else v)

        projectors = [np.outer(v, v.conj()) for v in opt_vecs]
        ψ          = self.state.vector
        amplitudes = np.array([v.conj() @ ψ for v in opt_vecs])  # ⟨oᵢ|ψ⟩
        probs      = np.real(amplitudes.conj() * amplitudes)      # |⟨oᵢ|ψ⟩|²

        # Quantum interference matrix I[i,j] = 2·Re(α_i* · ⟨oᵢ|oⱼ⟩ · α_j)
        gram = np.array([[opt_vecs[i].conj() @ opt_vecs[j]
                          for j in range(n_opts)] for i in range(n_opts)])
        I_mat = np.zeros((n_opts, n_opts))
        for i in range(n_opts):
            for j in range(n_opts):
                if i != j:
                    I_mat[i, j] = 2.0 * float(np.real(
                        np.conj(amplitudes[i]) * gram[i, j] * amplitudes[j]
                    ))

        # Order effect (first two options)
        order = {}
        if n_opts >= 2:
            order = QuantumInterference.order_effect(
                self.state, projectors[0], projectors[1]
            )

        # Conjunction fallacy test (first two options)
        conjunction = {}
        if n_opts >= 2:
            conjunction = QuantumInterference.conjunction(
                self.state, projectors[0], projectors[1]
            )

        # Normalize probabilities and choose
        probs_pos = np.maximum(probs, 0.0)
        s = probs_pos.sum()
        probs_norm = probs_pos / s if s > 1e-14 else np.ones(n_opts) / n_opts
        chosen = int(np.argmax(probs_norm))

        # State collapse to chosen option
        prob_c, collapsed = self.state.measure(projectors[chosen])
        self.state   = collapsed
        self.density = DensityMatrix.from_pure(self.state)

        total_interference = float(np.sum(np.abs(I_mat)))
        self._total_interference += total_interference
        self._n_decisions += 1

        return {
            'chosen_index':        chosen,
            'chosen_label':        option_labels[chosen],
            'probability':         float(probs_norm[chosen]),
            'all_probabilities':   probs_norm.tolist(),
            'all_labels':          option_labels,
            'interference_matrix': I_mat.tolist(),
            'total_interference':  total_interference,
            'order_effect':        order,
            'conjunction':         conjunction,
            'amplitudes_abs':      np.abs(amplitudes).tolist(),
        }

    # ── Quantum memory ────────────────────────────────────────────────────────

    def quantum_memorize(
        self,
        pattern:  np.ndarray,
        label:    str   = '',
        emotion:  str   = 'neutral',
        strength: float = 1.0,
    ) -> Dict:
        """
        Encode pattern into quantum memory with emotion-tagged phase.

        Different emotions are mapped to distinct phases φ ∈ [0, 2π),
        so emotional context creates separable interference patterns.
        Happiness (φ=0) and fear (φ=5π/6) will constructively interfere
        with themselves but may destructively interfere with each other.
        """
        phase = self._EMOTION_PHASES.get(emotion, np.pi)
        idx   = self.memory.encode(pattern, label=label, strength=strength, phase=phase)

        return {
            'index':            idx,
            'label':            label,
            'emotion':          emotion,
            'phase_rad':        phase,
            'n_stored':         len(self.memory),
            'effective_capacity': self.memory.effective_capacity(),
            'entanglement_entropy': self.memory.entanglement_entropy(),
        }

    def quantum_recall(
        self,
        query: np.ndarray,
        top_k: int = 5,
    ) -> Dict:
        """
        Retrieve memories via Born-rule fidelity + interference superposition.

        Returns both direct (per-pattern) and interference (full superposition)
        fidelities. The difference reveals constructive vs destructive interference.
        """
        matches = self.memory.recall(query, top_k=top_k, use_interference=True)
        return {
            'matches':            matches,
            'n_stored':           len(self.memory),
            'effective_capacity': self.memory.effective_capacity(),
            'entanglement':       self.memory.entanglement_entropy(),
        }

    # ── Decoherence ───────────────────────────────────────────────────────────

    def decohere(self, dt: float = 1.0) -> Dict:
        """
        Apply Lindblad dephasing to the density matrix.

        Simulates environmental interaction: off-diagonal coherences decay
        exponentially, converting quantum superposition into classical mixture.
        Purity decreases, von Neumann entropy increases.
        """
        p_before = self.density.purity()
        c_before = self.density.coherence()

        self.density = self.density.decohere(self._gamma, dt)
        self._t     += dt
        self._n_decoherence += 1

        return {
            'dt':              dt,
            'purity_before':   p_before,
            'purity_after':    self.density.purity(),
            'purity_loss':     p_before - self.density.purity(),
            'coherence_after': self.density.coherence(),
            'entropy':         self.density.von_neumann_entropy(),
            'cognitive_time':  self._t,
        }

    # ── Unitary evolution ─────────────────────────────────────────────────────

    def evolve(self, dt: float = 0.05) -> QuantumState:
        """
        Continuous cognitive evolution: |ψ(t+dt)⟩ = e^(−iHdt)|ψ(t)⟩.

        Rotates the cognitive state through Hilbert space guided by
        the current Hamiltonian, moving toward attractor states.
        Also applies to the density matrix: ρ' = U ρ U†.
        """
        U            = self.hamiltonian.unitary(dt)
        self.state   = self.state.evolve(U)
        self.density = self.density.evolve(U)
        self._t     += dt
        return self.state

    # ── Quantum walk ──────────────────────────────────────────────────────────

    def spread_activation(
        self,
        start_concept: int,
        n_steps:       int = 12,
    ) -> Dict:
        """
        Run a quantum walk from a concept node for n_steps.

        Quantum walk spreads O(t) in t steps vs O(√t) classically.
        Returns probability distribution, interference pattern,
        and quantum advantage ratio over classical baseline.
        """
        self.walk.reset(start_concept)
        self.walk.step(n_steps)

        dist    = self.walk.position_distribution()
        interf  = self.walk.interference_pattern()
        adv     = self.walk.quantum_advantage(n_steps)

        return {
            'start_concept':     start_concept,
            'n_steps':           n_steps,
            'distribution':      dist.tolist(),
            'interference':      interf.tolist(),
            'top_nodes':         self.walk.top_activated_nodes(5).tolist(),
            'spreading_entropy': self.walk.spreading_entropy(),
            'quantum_advantage': adv,
        }

    # ── Preciousness & reporting ──────────────────────────────────────────────

    def quantum_preciousness(self) -> Dict:
        """
        Five-axis quantum preciousness score (each axis 0–20, total 0–100).

        1. coherence_score      — l₁-norm coherence / max possible (superposition)
        2. entanglement_score   — memory entanglement entropy / log(dim)
        3. interference_score   — accumulated quantum interference across decisions
        4. decoherence_resist   — relative purity (resistance to classical collapse)
        5. walk_spread          — quantum spreading advantage over classical walk
        """
        # 1. Coherence
        # Max l₁ coherence for a pure state in ℂⁿ = n−1 (uniform superposition)
        coh_max = float(self._dim - 1)
        coh     = self.density.coherence()
        coherence_score = float(np.tanh(3.0 * coh / (coh_max + 1e-14)) * 20)

        # 2. Entanglement (memory)
        ent_H = self.memory.entanglement_entropy()
        # Max entanglement depends on Schmidt rank
        sqrt_dim = int(np.sqrt(self._dim))
        ent_max  = float(np.log(min(sqrt_dim, max(1, len(self.memory)))))
        entanglement_score = float(np.tanh(ent_H / (ent_max + 1e-14)) * 20) if ent_max > 0 else 0.0

        # 3. Interference exploitation
        if self._n_decisions > 0:
            avg_int = self._total_interference / self._n_decisions
            interference_score = float(np.tanh(avg_int * 5.0) * 20)
        else:
            interference_score = 0.0

        # 4. Decoherence resistance (relative purity)
        decoherence_resist = self.density.relative_purity() * 20.0

        # 5. Quantum walk advantage — canonical 16-step measurement
        # advantage ≈ σ_q / √16 = σ_q / 4
        # Expected quantum: σ_q ≈ 16/√2 ≈ 11.3 → advantage ≈ 2.83
        # Score normalised so advantage ≥ 2 → full marks
        _tmp = QuantumWalk(self.walk._n)
        _tmp.step(16)
        walk_adv   = _tmp.quantum_advantage(n_steps=16)
        walk_score = float(np.tanh((walk_adv - 1.0) / 1.5) * 20)

        total = (coherence_score + entanglement_score + interference_score
                 + decoherence_resist + walk_score)

        return {
            'coherence':          float(coherence_score),
            'entanglement':       float(entanglement_score),
            'interference':       float(interference_score),
            'decoherence_resist': float(decoherence_resist),
            'walk_spread':        float(walk_score),
            'total':              float(total),
            'max':                100.0,
        }

    def get_state_report(self) -> Dict:
        """Complete quantum state snapshot."""
        return {
            'hilbert_dim':          self._dim,
            'cognitive_time':       self._t,
            'purity':               self.density.purity(),
            'von_neumann_entropy':  self.density.von_neumann_entropy(),
            'l1_coherence':         self.density.coherence(),
            'n_memories':           len(self.memory),
            'memory_entanglement':  self.memory.entanglement_entropy(),
            'effective_capacity':   self.memory.effective_capacity(),
            'n_decisions':          self._n_decisions,
            'n_decoherence_steps':  self._n_decoherence,
            'decoherence_rate':     self._gamma,
            'hamiltonian_gap':      self.hamiltonian.spectral_gap(),
            'quantum_preciousness': self.quantum_preciousness(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 8.  DEMONSTRATION
# ══════════════════════════════════════════════════════════════════════════════

def demo_quantum_cognition():
    """
    Demonstrate all six quantum cognitive phenomena with real mathematics.
    """
    np.random.seed(0)
    print("\n" + "=" * 72)
    print("  QUANTUM COGNITION ENGINE — Mathematical Demonstration")
    print("=" * 72)

    engine = QuantumCognitionEngine(hilbert_dim=64, decoherence_rate=0.08,
                                    n_walk_nodes=32, memory_capacity=128)

    # ── DEMO 1: State encoding + Hamiltonian evolution ────────────────────────
    print("\n── 1. QUANTUM STATE ENCODING & EVOLUTION ─────────────────────────────")
    a1 = np.random.randn(64)
    a2 = a1 * 0.9 + np.random.randn(64) * 0.1   # similar to a1
    a3 = np.random.randn(64)                      # unrelated

    s1 = engine.encode_neural(a1)
    s2 = QuantumState.from_real(a2.astype(complex));  s2 = QuantumState(a2.astype(complex) + 1j*np.imag(s1.vector))
    s3 = QuantumState.from_real(a3.astype(complex))

    print(f"   State: {s1}")
    print(f"   Fidelity(similar):   {s1.fidelity(QuantumState(a2.astype(complex))):.4f}  (expect high)")
    print(f"   Fidelity(unrelated): {s1.fidelity(QuantumState(a3.astype(complex))):.4f}  (expect low)")

    engine.evolve(dt=0.1)
    print(f"   After Hamiltonian evolution dt=0.1:")
    print(f"   Purity={engine.density.purity():.6f}  "
          f"Coherence={engine.density.coherence():.4f}  "
          f"S={engine.density.von_neumann_entropy():.6f}")

    # ── DEMO 2: Conjunction Fallacy ───────────────────────────────────────────
    print("\n── 2. CONJUNCTION FALLACY ────────────────────────────────────────────")
    qa = np.random.randn(64); qa /= np.linalg.norm(qa)
    qb = 0.7 * qa + 0.3 * np.random.randn(64); qb /= np.linalg.norm(qb)
    P_a = QuantumInterference.rank1_projector(qa.astype(complex))
    P_b = QuantumInterference.rank1_projector(qb.astype(complex))

    conj = QuantumInterference.conjunction(engine.state, P_a, P_b)
    print(f"   P(A)                 = {conj['P(A)']:.4f}")
    print(f"   P(B)                 = {conj['P(B)']:.4f}")
    print(f"   P(A∩B) classical     = {conj['P(A∩B)_classical']:.4f}")
    print(f"   P(A→B) quantum       = {conj['P(A→B)_quantum']:.4f}")
    print(f"   Interference term    = {conj['interference_term']:+.4f}")
    print(f"   Conjunction fallacy  = {conj['conjunction_fallacy']}")

    # ── DEMO 3: Order Effects ────────────────────────────────────────────────
    print("\n── 3. QUESTION ORDER EFFECTS ─────────────────────────────────────────")
    order = QuantumInterference.order_effect(engine.state, P_a, P_b)
    print(f"   P(A→B)               = {order['P(A→B)']:.4f}")
    print(f"   P(B→A)               = {order['P(B→A)']:.4f}")
    print(f"   Order effect δ       = {order['order_effect']:+.4f}")
    print(f"   Commutator ‖[Pₐ,P_b]‖= {order['commutator_norm']:.4f}")
    print(f"   Compatible?          = {order['compatible']}")
    print(f"   QQ-equality holds?   = {order['QQ_equality_holds']}")

    # ── DEMO 4: Quantum Memory ───────────────────────────────────────────────
    print("\n── 4. QUANTUM ASSOCIATIVE MEMORY ─────────────────────────────────────")
    emotions = ['happiness', 'fear', 'surprise', 'neutral', 'love', 'anger']
    for i, emo in enumerate(emotions):
        pat = np.random.randn(64)
        engine.quantum_memorize(pat, label=f'exp_{i}', emotion=emo, strength=1.0)
    print(f"   Stored {len(engine.memory)} memories")
    print(f"   Effective capacity:  {engine.memory.effective_capacity():.2f}  "
          f"(classical max: {len(engine.memory)})")
    print(f"   Entanglement entropy:{engine.memory.entanglement_entropy():.4f}  "
          f"(0=product, log(n)=max entangled)")

    query = np.random.randn(64)
    recall = engine.quantum_recall(query, top_k=3)
    print(f"   Recall top-3 matches:")
    for m in recall['matches']:
        gain = m.get('interference_gain', 0.0)
        print(f"     [{m['rank']+1}] {m['label']:<10} "
              f"fidelity={m['fidelity']:.4f}  "
              f"interference_gain={gain:+.4f}")

    # ── DEMO 5: Decoherence ──────────────────────────────────────────────────
    print("\n── 5. ENVIRONMENTAL DECOHERENCE ──────────────────────────────────────")
    purities = [engine.density.purity()]
    entropies = [engine.density.von_neumann_entropy()]
    for _ in range(5):
        engine.decohere(dt=1.0)
        purities.append(engine.density.purity())
        entropies.append(engine.density.von_neumann_entropy())
    print(f"   Purity:  {' → '.join(f'{p:.3f}' for p in purities)}")
    print(f"   Entropy: {' → '.join(f'{s:.3f}' for s in entropies)}")
    print(f"   Quantum → Classical transition demonstrated")

    # ── DEMO 6: Quantum Walk ─────────────────────────────────────────────────
    print("\n── 6. QUANTUM WALK (spreading activation) ────────────────────────────")
    walk_res = engine.spread_activation(start_concept=0, n_steps=16)
    print(f"   Spreading entropy:   {walk_res['spreading_entropy']:.4f}")
    print(f"   Quantum advantage:   {walk_res['quantum_advantage']:.3f}× over classical")
    print(f"   Top activated nodes: {walk_res['top_nodes']}")

    # ── DEMO 7: Quantum Decision ─────────────────────────────────────────────
    print("\n── 7. QUANTUM DECISION MAKING ────────────────────────────────────────")
    opts = [np.random.randn(64) for _ in range(4)]
    decision = engine.quantum_decide(opts, ['code', 'explain', 'debug', 'refactor'])
    print(f"   Chosen: {decision['chosen_label']}  "
          f"(P={decision['probability']:.4f})")
    print(f"   All probs: {', '.join(f'{l}:{p:.3f}' for l, p in zip(decision['all_labels'], decision['all_probabilities']))}")
    print(f"   Total interference: {decision['total_interference']:.4f}")

    # ── Final preciousness score ──────────────────────────────────────────────
    print("\n── QUANTUM PRECIOUSNESS SCORE ────────────────────────────────────────")
    qp = engine.quantum_preciousness()
    print(f"   Coherence          : {qp['coherence']:.2f} / 20")
    print(f"   Entanglement       : {qp['entanglement']:.2f} / 20")
    print(f"   Interference       : {qp['interference']:.2f} / 20")
    print(f"   Decoherence resist : {qp['decoherence_resist']:.2f} / 20")
    print(f"   Walk spread        : {qp['walk_spread']:.2f} / 20")
    print(f"   ─────────────────────────────────────")
    print(f"   QUANTUM TOTAL      : {qp['total']:.2f} / 100")
    print("\n" + "=" * 72 + "\n")

    return engine


if __name__ == "__main__":
    demo_quantum_cognition()
