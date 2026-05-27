"""
QHPCLM.py — Quantum Hierarchical Predictive Coding Language Model

Upgrades HPCLM with full quantum mechanics from Quantum_Cognition.py.

  Classical HPCLM                      Quantum QHPCLM
  ─────────────────────────────────────────────────────────────────
  real activations  a ∈ ℝⁿ            complex quantum state |ψ⟩ ∈ ℂⁿ
  softmax(W·a) → P(next token)         Born rule  |⟨eₜ|ψ⟩|² → P(next token)
  Hebbian  ΔW = α·outer(a, x)          Quantum ΔH = iα(|eₜ₊₁⟩⟨ψₜ| − |ψₜ⟩⟨eₜ₊₁|)
  weight decay                          Lindblad decoherence  ρᵢⱼ·e^{−γt}
  k-WTA sparsity                        quantum state = k-WTA replaced by unitary
  free energy  ‖e‖²                    quantum free energy  1 − P(correct next token)

Architecture for token t:
  |eₜ⟩   = analytic_embed(E[t])            (complex quantum embedding)
  |ψ₁⟩   = U₁|eₜ⟩                          (L1 Hamiltonian evolution)
  |ψ₂⟩   = U₂|ψ₁⟩                          (L2 evolution of L1 state)
  |ψ₃⟩   = U₃|ψ₂⟩                          (L3 evolution of L2 state)
  P(t')  = |⟨eₜ'|ψ₁⟩|² / Z               (Born rule over vocab — NO softmax)

H update (quantum Delta rule):
  Goal:   U₁|eₜ⟩ → |eₜ₊₁⟩               (predict next token embedding)
  ΔH₁  = iα(|eₜ₊₁⟩⟨ψ₁| − |ψ₁⟩⟨eₜ₊₁|)  (Hermitian, local, no backprop)

Three levels (γ controls decoherence / forgetting speed):
  L1 Phonemic  γ=0.02  — character/token patterns   (fast decoherence)
  L2 Lexical   γ=0.03  — word / subword structure
  L3 Semantic  γ=0.01  — conceptual context          (slow decoherence)
"""

import numpy as np
from typing import Dict, List, Tuple
import time

from Quantum_Cognition import (
    DensityMatrix,
    QuantumMemoryBank,
    QuantumState,
    QuantumWalk,
)
from HPCLM import CharTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Exact unitary  e^{−iHt}  for Hermitian H
# ──────────────────────────────────────────────────────────────────────────────

def unitary(H: np.ndarray, t: float = 1.0) -> np.ndarray:
    """
    e^{−iHt} for Hermitian H, exact via eigendecomposition.

    H = V D V†  (D real diagonal)
    e^{−iHt} = V e^{−iDt} V†

    Never truncates a series — this is the exact quantum time evolution.
    """
    eigs, vecs = np.linalg.eigh(H)
    return (vecs * np.exp(-1j * eigs * t)) @ vecs.conj().T


def hermitian_norm(H: np.ndarray) -> np.ndarray:
    """
    Re-symmetrise and scale H so eigenvalues ∈ [−π, π].
    Prevents phase aliasing (U wrapping) and H diverging under repeated updates.
    """
    H    = (H + H.conj().T) / 2
    eigs = np.linalg.eigvalsh(H)
    spr  = eigs.max() - eigs.min()
    if spr > 1e-14:
        H = (H - eigs.min() * np.eye(H.shape[0])) / spr * (2 * np.pi) - np.pi * np.eye(H.shape[0])
    return H


# ──────────────────────────────────────────────────────────────────────────────
# Analytic signal encoder  (real → complex quantum state via FFT)
# ──────────────────────────────────────────────────────────────────────────────

def analytic_embed(v: np.ndarray, dim: int) -> np.ndarray:
    """
    Embed real vector v into ℂ^dim via analytic signal (Hilbert transform).

    z = v + i·H[v],  H[v] = discrete Hilbert transform via FFT phase shift.

    Property: |⟨z_a|z_b⟩|² ≈ cosine²(a,b) — semantically similar tokens
    produce high-fidelity quantum states naturally.
    """
    if len(v) < dim:
        v = np.pad(v, (0, dim - len(v)))
    else:
        v = v[:dim]
    v  = v.astype(float)
    A  = np.fft.rfft(v)
    hf = np.ones(len(A), dtype=complex)
    hf[1:] = -1j                                   # +π/2 phase on positive freqs
    Hv = np.fft.irfft(A * hf, n=dim)
    z  = v.astype(complex) + 1j * Hv
    nrm = np.linalg.norm(z)
    return z / nrm if nrm > 1e-14 else z


# ──────────────────────────────────────────────────────────────────────────────
# Quantum Predictive Coding Level
# ──────────────────────────────────────────────────────────────────────────────

class QuantumPCLevel:
    """
    One quantum level: evolves an input state through a learnable Hamiltonian.

    State:      |ψ⟩ = U|e⟩  ∈ ℂ^dim       (Hamiltonian-transformed embedding)
    Hamiltonian H (Hermitian, dim×dim)    — governs unitary U = e^{−iH}
    Density     ρ                         — decoherence tracker

    H update (quantum Delta rule, called from QHPCLM with supervision):
      Goal: U|eₜ⟩ → |target⟩
      ΔH = iα(|target⟩⟨ψ| − |ψ⟩⟨target|)  — Hermitian, local, no backprop
    """

    def __init__(
        self,
        dim:              int,
        decoherence_rate: float = 0.02,
        u_refresh:        int   = 8,
        seed:             int   = 0,
    ):
        self.dim       = dim
        self.gamma     = decoherence_rate
        self.u_refresh = u_refresh   # recompute U every N H-updates (lazy caching)

        # GUE Hamiltonian — random Hermitian, eigenvalues normalised to [−π, π]
        rng  = np.random.default_rng(seed)
        A    = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        self.H: np.ndarray = hermitian_norm(
            ((A + A.conj().T) / (2.0 * np.sqrt(2 * dim))).real
        )

        self._psi    = QuantumState.uniform(dim).vector    # (dim,) complex
        self.density = DensityMatrix.from_pure(QuantumState(self._psi))

        self._U:             np.ndarray = None
        self._update_count:  int        = 0
        self._density_dirty: bool       = False

    def _get_U(self) -> np.ndarray:
        """Return cached unitary, recomputing if cache is empty."""
        if self._U is None:
            self._U = unitary(self.H, t=1.0)
        return self._U

    def evolve(self, x_psi: np.ndarray) -> np.ndarray:
        """
        |ψ⟩ = normalise(U|x⟩) where U = e^{−iH}.
        Uses cached U; density matrix is lazily recomputed only on property access.
        """
        x_psi = np.asarray(x_psi, dtype=complex).ravel()
        nrm   = np.linalg.norm(x_psi)
        if nrm > 1e-14:
            x_psi = x_psi / nrm

        out = self._get_U() @ x_psi
        nrm = np.linalg.norm(out)
        out = out / nrm if nrm > 1e-14 else out

        self._psi         = out
        self._density_dirty = True   # recompute density matrix only when requested

        return out

    def update_H(self, target_psi: np.ndarray, lr: float):
        """
        Quantum Delta rule: push H so U|current_input⟩ → |target_psi⟩.

        ΔH = iα(|target⟩⟨ψ| − |ψ⟩⟨target|)
        Hermitian: (ΔH)† = ΔH ✓ (anti-Hermitian × i)

        U is refreshed every u_refresh calls (lazy caching) — acceptable
        approximation since lr is small and H changes slowly.
        """
        target = np.asarray(target_psi, dtype=complex).ravel()
        nrm    = np.linalg.norm(target)
        if nrm > 1e-14:
            target = target / nrm
        dH = 1j * lr * (
            np.outer(target, self._psi.conj()) - np.outer(self._psi, target.conj())
        )
        self.H = hermitian_norm(self.H + dH)
        self._update_count += 1
        if self._update_count % self.u_refresh == 0:
            self._U = unitary(self.H, t=1.0)

    def reset(self):
        self._psi           = QuantumState.uniform(self.dim).vector
        self.density        = DensityMatrix.from_pure(QuantumState(self._psi))
        self._U             = unitary(self.H, t=1.0)
        self._density_dirty = False

    def _ensure_density(self):
        if self._density_dirty:
            self.density = DensityMatrix.from_pure(
                QuantumState(self._psi)
            ).decohere(self.gamma, 1.0)
            self._density_dirty = False

    @property
    def purity(self) -> float:
        self._ensure_density()
        return self.density.purity()

    @property
    def coherence(self) -> float:
        self._ensure_density()
        return self.density.coherence()


# ──────────────────────────────────────────────────────────────────────────────
# QHPCLM — full quantum model
# ──────────────────────────────────────────────────────────────────────────────

class QHPCLM:
    """
    Quantum Hierarchical Predictive Coding Language Model.

    Forward pass for token t:
      |e_t⟩  = analytic_embed(E[t])           embedding → complex state
      |ψ₁⟩   = U₁ · |e_t⟩                     L1 Hamiltonian evolution
      |ψ₂⟩   = U₂ · |ψ₁⟩                      L2 evolution
      |ψ₃⟩   = U₃ · |ψ₂⟩                      L3 evolution

    Next-token prediction (Born rule, NO softmax):
      P(t') = |⟨E_psi[t'] | ψ₁⟩|² / Z

    Learning (quantum Delta rule on H₁ using next-token supervision):
      ΔH₁ = iα(|e_{t+1}⟩⟨ψ₁| − |ψ₁⟩⟨e_{t+1}|)

    After training: U₁ maps embed(t) → embed(t+1) for common sequences.
    Born rule naturally assigns high probability to the next character.
    """

    def __init__(
        self,
        tokenizer:        CharTokenizer,
        hilbert_dim:      int   = 64,
        decoherence_rate: float = 0.02,
        lr:               float = 0.01,
        mem_capacity:     int   = 256,
        walk_nodes:       int   = 32,
        seed:             int   = 42,
    ):
        self.tok         = tokenizer
        self.V           = tokenizer.vocab_size
        self.hilbert_dim = hilbert_dim
        self.lr          = lr

        rng = np.random.RandomState(seed)

        # Real token embeddings → complex analytic quantum states (frozen)
        E_real = rng.randn(self.V, hilbert_dim).astype(np.float32)
        norms  = np.linalg.norm(E_real, axis=1, keepdims=True) + 1e-8
        E_real /= norms
        self.E_psi: np.ndarray = np.stack([
            analytic_embed(E_real[t], hilbert_dim) for t in range(self.V)
        ])  # (V, hilbert_dim) complex, each row unit-normalised

        # Three quantum levels
        self.L1 = QuantumPCLevel(hilbert_dim, decoherence_rate,       seed=seed+1)
        self.L2 = QuantumPCLevel(hilbert_dim, decoherence_rate * 1.5, seed=seed+2)
        self.L3 = QuantumPCLevel(hilbert_dim, decoherence_rate * 0.5, seed=seed+3)

        # Quantum memory — stores high-surprise states
        self.qmemory = QuantumMemoryBank(hilbert_dim, capacity=mem_capacity)

        # Quantum walk — context spreading (O(t) vs classical O(√t))
        self.qwalk = QuantumWalk(walk_nodes)

        self.step_count   = 0
        self._qfe_history:  List[float] = []
        self._pur_history:  List[float] = []

    # ── embedding ─────────────────────────────────────────────────────────────

    def _embed(self, token_id: int) -> np.ndarray:
        return self.E_psi[token_id].copy()

    # ── Born rule ─────────────────────────────────────────────────────────────

    def _born_probs(self, psi: np.ndarray) -> np.ndarray:
        """
        P(t') = |⟨E_psi[t'] | ψ⟩|² / Z   (POVM normalisation).

        Amplitudes: E_psi @ ψ.conj()  → (V,) complex
        Probs:      |amplitudes|²      → (V,) real ≥ 0, normalised.
        """
        amps  = self.E_psi @ psi.conj()
        probs = np.real(amps.conj() * amps)
        probs = np.maximum(probs, 0.0)
        s     = probs.sum()
        return probs / s if s > 1e-14 else np.ones(self.V) / self.V

    # ── core step ─────────────────────────────────────────────────────────────

    def step(self, token_id: int, next_token_id: int = -1, learn: bool = True) -> Dict:
        """
        Full quantum forward pass for one token.

        token_id      — current token (drives Hamiltonian evolution)
        next_token_id — supervision target for H update (-1 = no update)
        """
        e_t = self._embed(token_id)                        # |e_t⟩

        # Evolve through quantum hierarchy
        psi1 = self.L1.evolve(e_t)                         # U₁|e_t⟩
        psi2 = self.L2.evolve(psi1)                        # U₂|ψ₁⟩
        _    = self.L3.evolve(psi2)                        # U₃|ψ₂⟩

        # Born rule prediction from L1 state
        probs = self._born_probs(psi1)

        # Quantum free energy = 1 − P(correct next token)
        if next_token_id >= 0:
            qfe = 1.0 - float(probs[next_token_id])
        else:
            qfe = 1.0 - float(probs.max())

        # Quantum Delta rule: update H₁ so U₁|e_t⟩ → |e_{t+1}⟩
        if learn and next_token_id >= 0:
            e_next = self._embed(next_token_id)
            self.L1.update_H(e_next, self.lr)
            # L2: push U₂|ψ₁⟩ toward what next L1 state will be
            # (approximate: use e_next as proxy for next L1 output)
            self.L2.update_H(e_next, self.lr * 0.5)

        # Store high-surprise states in quantum memory
        if learn and qfe > 0.7:
            self.qmemory.encode(e_t, label=self.tok.id2ch.get(token_id, '?'),
                                strength=qfe)

        # Quantum walk: spread activation from current token
        self.qwalk.reset(token_id % self.qwalk._n)
        self.qwalk.step(4)

        self._qfe_history.append(qfe)
        self._pur_history.append(self.L1.purity)
        self.step_count += 1

        top_id = int(np.argmax(probs))
        return {
            'probs':       probs,
            'top_token':   top_id,
            'top_char':    self.tok.id2ch.get(top_id, '?'),
            'confidence':  float(probs[top_id]),
            'qfe':         qfe,
            'purity_l1':   self.L1.purity,
            'coherence_l1':self.L1.coherence,
            'walk_entropy':float(self.qwalk.spreading_entropy()),
            'n_memories':  len(self.qmemory),
        }

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, text: str, verbose: bool = True) -> Dict:
        ids     = self.tok.encode(text, add_bos=True, add_eos=True)
        n       = len(ids)
        correct = 0
        t0      = time.time()

        for i, tok_id in enumerate(ids):
            next_id = ids[i + 1] if i + 1 < n else -1
            result  = self.step(tok_id, next_token_id=next_id, learn=True)
            if next_id >= 0 and result['top_token'] == next_id:
                correct += 1

            if verbose and n > 1 and (i + 1) % max(1, n // 4) == 0:
                acc = correct / max(i, 1)
                print(f"  [{i+1:4d}/{n}]  acc={acc:.3f}  "
                      f"conf={result['confidence']:.4f}  "
                      f"QFE={result['qfe']:.4f}  "
                      f"purity={result['purity_l1']:.4f}")

        elapsed  = time.time() - t0
        accuracy = correct / max(n - 1, 1)
        return {
            'tokens':    n,
            'accuracy':  accuracy,
            'elapsed':   elapsed,
            'tok_per_s': n / elapsed,
            'final_qfe': self._qfe_history[-1] if self._qfe_history else 1.0,
        }

    # ── generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt:             str,
        max_tokens:         int   = 100,
        temperature:        float = 1.0,
        top_k:              int   = 5,
        repetition_penalty: float = 0.60,
        repetition_window:  int   = 20,
        learn:              bool  = False,
    ) -> str:
        ids = self.tok.encode(prompt, add_bos=True, add_eos=False)
        for tok_id in ids:
            self.step(tok_id, learn=learn)

        out_ids: List[int] = []
        last_id = ids[-1]

        # Special token ids that should never appear in generated output
        special = {CharTokenizer.PAD, CharTokenizer.UNK, CharTokenizer.BOS}

        for step_i in range(max_tokens):
            result = self.step(last_id, learn=learn)
            probs  = result['probs'].astype(np.float64)

            # Mask all special tokens from generated output; max_tokens controls length
            for sid in special:
                probs[sid] = 0.0
            probs[CharTokenizer.EOS] = 0.0

            # Repetition penalty: proportional to frequency in recent window
            if repetition_penalty < 1.0 and out_ids:
                from collections import Counter
                counts = Counter(out_ids[-repetition_window:])
                for prev_id, cnt in counts.items():
                    probs[prev_id] *= repetition_penalty ** cnt

            s = probs.sum()
            if s < 1e-12:
                probs = np.ones(self.V) / self.V
            else:
                probs /= s

            # Temperature scaling on log-probabilities
            logits  = np.log(probs + 1e-12) / max(temperature, 1e-6)
            logits -= logits.max()
            probs   = np.exp(logits)

            if top_k > 0:
                threshold = float(np.sort(probs)[-min(top_k, len(probs))])
                probs = np.where(probs >= threshold, probs, 0.0)

            s = probs.sum()
            probs = probs / s if s > 1e-12 else np.ones(self.V) / self.V

            next_id = int(np.random.choice(self.V, p=probs))
            if next_id == CharTokenizer.EOS:
                break
            out_ids.append(next_id)
            last_id = next_id

        return self.tok.decode(out_ids)

    # ── helpers ───────────────────────────────────────────────────────────────

    def reset_context(self):
        self.L1.reset()
        self.L2.reset()
        self.L3.reset()
        self.qwalk = QuantumWalk(self.qwalk._n)

    def param_count(self) -> int:
        return (3 * self.hilbert_dim ** 2 * 2 +
                self.V * self.hilbert_dim * 2)

    def _qp_score(self) -> Dict:
        coh_max = float(self.hilbert_dim - 1)
        coh     = self.L1.coherence
        c_score = float(np.tanh(3.0 * coh / (coh_max + 1e-14)) * 20)
        ent     = self.qmemory.entanglement_entropy()
        ent_max = float(np.log(max(1, min(8, len(self.qmemory)))))
        e_score = float(np.tanh(ent / (ent_max + 1e-14)) * 20) if ent_max > 0 else 0.0
        interf  = float(abs(self.L1._psi.conj() @ self.L2._psi) ** 2) * 20
        d_score = self.L1.purity * 20.0
        _w      = QuantumWalk(self.qwalk._n)
        _w.step(16)
        w_score = float(np.tanh((_w.quantum_advantage(16) - 1.0) / 1.5) * 20)
        return {'coherence': c_score, 'entanglement': e_score,
                'interference': interf, 'decohere_resist': d_score,
                'walk_spread': w_score,
                'total': c_score + e_score + interf + d_score + w_score}

    def summary(self):
        avg_qfe = float(np.mean(self._qfe_history[-100:])) if self._qfe_history else 1.0
        avg_pur = float(np.mean(self._pur_history[-100:])) if self._pur_history else 1.0
        qp      = self._qp_score()
        print("\n" + "=" * 64)
        print("  QHPCLM — Quantum Hierarchical Predictive Coding LM")
        print("=" * 64)
        print(f"  Hilbert dim    : {self.hilbert_dim}")
        print(f"  Vocab size     : {self.V}")
        print(f"  Parameters     : {self.param_count():,}")
        print(f"  Steps trained  : {self.step_count:,}")
        print(f"  Avg QFE  (100) : {avg_qfe:.4f}  (0=perfect, 1=random)")
        print(f"  Avg purity(100): {avg_pur:.4f}  (1=pure quantum)")
        print(f"  Q-memories     : {len(self.qmemory)}")
        print(f"\n  Level       γ        purity   coherence")
        print(f"  L1 phonem.  {self.L1.gamma:.4f}   {self.L1.purity:.4f}   {self.L1.coherence:.4f}")
        print(f"  L2 lexical  {self.L2.gamma:.4f}   {self.L2.purity:.4f}   {self.L2.coherence:.4f}")
        print(f"  L3 semant.  {self.L3.gamma:.4f}   {self.L3.purity:.4f}   {self.L3.coherence:.4f}")
        print(f"\n  Quantum Preciousness : {qp['total']:.1f} / 100")
        print(f"    Coherence {qp['coherence']:.1f}  Entanglement {qp['entanglement']:.1f}"
              f"  Interference {qp['interference']:.1f}")
        print(f"    Decohere-resist {qp['decohere_resist']:.1f}  Walk-spread {qp['walk_spread']:.1f}")
        print("=" * 64)


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

DEMO_TEXT = """\
The brain is not a computer. It is a prediction machine.
Every perception is a controlled hallucination.
Consciousness arises from the brain's attempt to minimise surprise.
The world we see is not reality — it is the brain's best guess.
Free energy drives all thought. Memory is not storage; it is reconstruction.
Learning is prediction error minimisation. Intelligence models and predicts.
"""


def run_demo():
    print("=" * 64)
    print("  QHPCLM — Quantum Neural Language Model")
    print("  Born rule · Unitary evolution · Lindblad decoherence")
    print("  Quantum walk · Phase-tagged memory · No softmax")
    print("=" * 64)

    tok = CharTokenizer().fit(DEMO_TEXT)
    print(f"  Vocab size  : {tok.vocab_size} characters")

    model   = QHPCLM(tok, hilbert_dim=64, decoherence_rate=0.02, lr=0.05, seed=42)
    from HPCLM import HPCLM
    classic = HPCLM(tok, embed_dim=64, l1_dim=128, l2_dim=256, l3_dim=512, lr=0.02)

    print(f"  Q params    : {model.param_count():,}")
    print(f"  C params    : {classic.param_count():,}\n")

    print("── Training (10 epochs each) ────────────────────────────────")
    print(f"  {'Ep':>3}  {'Q-acc':>6}  {'Q-QFE':>7}  {'Q-pur':>6}  {'C-acc':>6}  {'mems':>5}")
    print("  " + "─" * 48)

    for epoch in range(1, 11):
        model.reset_context()
        qr = model.train(DEMO_TEXT, verbose=False)

        classic.reset_context()
        cr = classic.train(DEMO_TEXT, verbose=False)

        avg_pur = float(np.mean(model._pur_history[-100:])) if model._pur_history else 0.0
        print(f"  {epoch:3d}  {qr['accuracy']:6.3f}  {qr['final_qfe']:7.4f}  "
              f"{avg_pur:6.4f}  {cr['accuracy']:6.3f}  {len(model.qmemory):5d}")

    model.summary()

    # ── Born rule vs softmax ───────────────────────────────────────────────────
    print("\n── Born rule vs Softmax (prompt: 'The ') ────────────────────")
    model.reset_context()
    classic.reset_context()
    for tid in tok.encode("The ", add_bos=True, add_eos=False):
        model.step(tid, learn=False)
        classic.step(tid, learn=False)

    last_id = tok.encode("The ", add_bos=False, add_eos=False)[-1]
    q_r = model.step(last_id, learn=False)
    c_r = classic.step(last_id, learn=False)

    print("  Quantum (Born): " +
          ', '.join(f"{tok.id2ch.get(i,'?')!r}:{q_r['probs'][i]:.3f}"
                    for i in np.argsort(q_r['probs'])[-5:][::-1]))
    print("  Classic (soft): " +
          ', '.join(f"{tok.id2ch.get(i,'?')!r}:{c_r['probs'][i]:.3f}"
                    for i in np.argsort(c_r['probs'])[-5:][::-1]))

    # ── Generation ────────────────────────────────────────────────────────────
    print("\n── Quantum generation ───────────────────────────────────────")
    for temp, label in [(0.5, 'focused'), (1.0, 'balanced'), (1.5, 'creative')]:
        model.reset_context()
        out = model.generate("The brain", max_tokens=80, temperature=temp, top_k=8)
        print(f"  T={temp} ({label}): {out!r}")

    # ── Quantum walk ──────────────────────────────────────────────────────────
    print("\n── Quantum walk context spreading ───────────────────────────")
    model.qwalk.reset(0)
    model.qwalk.step(16)
    print(f"  Spreading entropy : {model.qwalk.spreading_entropy():.4f}")
    print(f"  Quantum advantage : {model.qwalk.quantum_advantage(16):.3f}× over classical")
    print(f"  Top activated     : {model.qwalk.top_activated_nodes(5).tolist()}")


if __name__ == "__main__":
    run_demo()
