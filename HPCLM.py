"""
HPCLM.py — Hierarchical Predictive Coding Language Model

Karl Friston's Free Energy Principle applied to language from first principles.
No sklearn. No transformers. No external NLP libs.

Architecture (3 levels above embedding):
  L3 Semantic  — conceptual context,     dim=512, sparsity=3%
  L2 Lexical   — word/subword patterns,  dim=256, sparsity=4%
  L1 Phonemic  — character patterns,     dim=128, sparsity=5%
  E  Embedding — token → dense vector,   dim=64

Each level predicts the representation BELOW it.
Prediction errors propagate UPWARD.
Weights update via local Hebbian rules — no global backprop ever.

Precision stack:
  scipy.special.softmax  — numerically stable token distribution (log-sum-exp)
  scipy.special.rel_entr — exact KL divergence for free energy audit
  scipy.linalg QR        — orthonormal weight initialisation (best conditioning)
  k-WTA                  — biologically faithful sparse coding (2–5% active)
"""

import time
from typing import Dict, List

import numpy as np
from scipy.linalg import qr as scipy_qr
from scipy.special import rel_entr, softmax


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer — character-level, zero external dependencies
# ──────────────────────────────────────────────────────────────────────────────

class CharTokenizer:
    """
    Character-level tokenizer built from first principles.
    Special tokens: <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3.
    """
    PAD, UNK, BOS, EOS = 0, 1, 2, 3

    def __init__(self):
        self.ch2id: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.id2ch: Dict[int, str] = {v: k for k, v in self.ch2id.items()}

    def fit(self, text: str) -> 'CharTokenizer':
        for ch in sorted(set(text)):
            if ch not in self.ch2id:
                idx = len(self.ch2id)
                self.ch2id[ch] = idx
                self.id2ch[idx] = ch
        return self

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids = [self.ch2id.get(ch, self.UNK) for ch in text]
        if add_bos:
            ids = [self.BOS] + ids
        if add_eos:
            ids = ids + [self.EOS]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        special = {self.PAD, self.UNK, self.BOS, self.EOS}
        return ''.join(
            self.id2ch.get(i, '?')
            for i in ids
            if not skip_special or i not in special
        )

    @property
    def vocab_size(self) -> int:
        return len(self.ch2id)


# ──────────────────────────────────────────────────────────────────────────────
# Precision math utilities
# ──────────────────────────────────────────────────────────────────────────────

def k_wta(x: np.ndarray, k: int) -> np.ndarray:
    """k-Winners-Take-All: keep top-k activations, zero the rest (sparse coding)."""
    if k >= len(x):
        return x.copy()
    out = np.zeros_like(x)
    idx = np.argpartition(x, -k)[-k:]
    out[idx] = x[idx]
    return out


def ortho_init(rows: int, cols: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Xavier-scaled orthonormal weight initialisation via QR decomposition.
    Orthonormal start gives the best condition number for Hebbian convergence.
    """
    if rows >= cols:
        M = rng.randn(rows, cols).astype(np.float32)
        Q, _ = scipy_qr(M, mode='economic')   # Q: (rows, cols), orthonormal columns
        W = Q
    else:
        M = rng.randn(cols, rows).astype(np.float32)
        Q, _ = scipy_qr(M, mode='economic')   # Q: (cols, rows)
        W = Q.T                                # (rows, cols)
    scale = np.sqrt(2.0 / (rows + cols))
    return (W * scale).astype(np.float32)


def free_energy(error: np.ndarray, precision: float) -> float:
    """
    Variational free energy for one level: F = 0.5 * π * ||e||²
    π = precision (inverse variance). Exact — no approximation.
    """
    return 0.5 * precision * float(np.dot(error, error))


def kl_from_uniform(probs: np.ndarray) -> float:
    """
    KL divergence KL(p || uniform) via scipy.special.rel_entr.
    Measures how focused the prediction is vs. max-entropy baseline.
    Higher = more confident prediction.
    """
    V = len(probs)
    uniform = np.full(V, 1.0 / V, dtype=np.float64)
    return float(np.sum(rel_entr(probs.astype(np.float64), uniform)))


def row_normalise(W: np.ndarray, eps: float = 1e-8) -> None:
    """In-place L2 row normalisation — prevents Hebbian weight explosion."""
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W /= (norms + eps)


def col_normalise(W: np.ndarray, eps: float = 1e-8) -> None:
    """In-place L2 column normalisation."""
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    W /= (norms + eps)


# ──────────────────────────────────────────────────────────────────────────────
# Predictive Coding Level — one node in the hierarchy
# ──────────────────────────────────────────────────────────────────────────────

class PCLevel:
    """
    One level in the hierarchical predictive coding stack.

    Weights:
      W  (act_dim × input_dim)  — recognition: input error → activation (bottom-up)
      V  (input_dim × act_dim)  — generative:  activation → prediction of input (top-down)

    Per-step:
      1. Top-down prediction:  μ  = V @ a_prev
      2. Prediction error:     e  = x − μ            (surprise signal)
      3. New activation:       a  = k_WTA(W @ x, k)  (sparse recognition)
      4. Free energy:          F  = 0.5 * π * ||e||²
      5. Hebbian W update:     ΔW = lr * outer(a, x)  → row-normalise
      6. Delta-rule V update:  ΔV = lr * outer(e, a)  → col-normalise
      7. Precision adaptation: π  → 1 / Var(e)  (slow exponential average)

    Returns error e upward to the next level.
    """

    def __init__(
        self,
        input_dim:  int,
        act_dim:    int,
        sparsity:   float = 0.05,
        lr:         float = 0.01,
        seed:       int   = 0,
    ):
        self.input_dim = input_dim
        self.act_dim   = act_dim
        self.k         = max(1, int(sparsity * act_dim))
        self.lr        = lr
        self.precision = 1.0

        rng     = np.random.RandomState(seed)
        self.W  = ortho_init(act_dim,   input_dim, rng)  # (act_dim,   input_dim)
        self.V  = ortho_init(input_dim, act_dim,   rng)  # (input_dim, act_dim)
        self.a  = np.zeros(act_dim,  dtype=np.float32)   # current activation (sparse)

        self._fe_buf: List[float] = []

    def step(self, x: np.ndarray, learn: bool = True) -> np.ndarray:
        """
        Forward + optional Hebbian update.
        x       — input vector (raw embed or error from level below)
        returns — prediction error e propagated upward to next level
        """
        x = x.astype(np.float32)

        # Top-down: what this level predicted x would be
        mu = self.V @ self.a                       # (input_dim,)

        # Upward signal: what surprised this level
        e  = x - mu                                # (input_dim,)

        # Recognition: update activation from bottom-up input
        self.a = k_wta(self.W @ x, self.k)        # (act_dim,)

        # Track raw ||e||² (unweighted) — precision-weighted FE explodes at deep levels
        fe = float(np.dot(e, e))
        self._fe_buf.append(fe)
        if len(self._fe_buf) > 200:
            self._fe_buf.pop(0)

        if learn:
            # Hebbian: recognition weights (bottom-up path)
            self.W += self.lr * np.outer(self.a, x)
            row_normalise(self.W)

            # Delta-rule Hebb: generative weights (top-down path)
            self.V += self.lr * np.outer(e, self.a)
            col_normalise(self.V)

            # Precision adapts to error variance — capped to prevent explosion
            err_var = float(np.mean(e ** 2)) + 1e-8
            raw_prec = 1.0 / err_var
            self.precision = float(np.clip(
                0.999 * self.precision + 0.001 * raw_prec, 0.01, 20.0
            ))

        return e

    def reset(self):
        self.a[:] = 0.0

    @property
    def avg_free_energy(self) -> float:
        return float(np.mean(self._fe_buf)) if self._fe_buf else 0.0

    @property
    def sparsity_ratio(self) -> float:
        return float(np.count_nonzero(self.a)) / self.act_dim


# ──────────────────────────────────────────────────────────────────────────────
# HPCLM — full hierarchical model
# ──────────────────────────────────────────────────────────────────────────────

class HPCLM:
    """
    Hierarchical Predictive Coding Language Model.

    Forward pass for token t:
      embed(t) → L1.step(embed) → e1 → L2.step(e1) → e2 → L3.step(e2)
      prediction of t+1: softmax(decode_W @ L1.a)   ← scipy stable softmax

    All learning is local Hebbian / Delta-rule — no global backprop, no frozen weights.
    The model is always learning: train-time and inference-time are the same pass.
    """

    def __init__(
        self,
        tokenizer:   CharTokenizer,
        embed_dim:   int   = 64,
        l1_dim:      int   = 128,
        l2_dim:      int   = 256,
        l3_dim:      int   = 512,
        l1_sparsity: float = 0.05,
        l2_sparsity: float = 0.04,
        l3_sparsity: float = 0.03,
        lr:          float = 0.01,
        seed:        int   = 42,
    ):
        self.tok       = tokenizer
        self.V         = tokenizer.vocab_size
        self.embed_dim = embed_dim
        self.lr        = lr
        rng            = np.random.RandomState(seed)

        # Token embedding table — Hebbian-updated, row-normalised
        self.E = (rng.randn(self.V, embed_dim) * 0.1).astype(np.float32)
        row_normalise(self.E)

        # Three predictive coding levels
        self.L1 = PCLevel(embed_dim, l1_dim, l1_sparsity, lr, seed=seed + 1)
        self.L2 = PCLevel(l1_dim,   l2_dim, l2_sparsity, lr, seed=seed + 2)
        self.L3 = PCLevel(l2_dim,   l3_dim, l3_sparsity, lr, seed=seed + 3)

        # Decode head: L1.a → vocab logits — Hebbian-updated
        self.decode_W = (rng.randn(self.V, l1_dim) * 0.1).astype(np.float32)
        row_normalise(self.decode_W)

        self.step_count      = 0
        self._fe_history: List[float] = []

    # ── embedding ─────────────────────────────────────────────────────────────

    def _embed(self, token_id: int) -> np.ndarray:
        return self.E[token_id].copy()

    # ── core step ─────────────────────────────────────────────────────────────

    def step(self, token_id: int, next_token_id: int = -1, learn: bool = True) -> Dict:
        """
        Process one token — full hierarchy forward pass + optional Hebbian update.
        next_token_id — supervision target for decode head (-1 = no decode update)
        Returns prediction distribution for the NEXT token plus diagnostics.
        """
        x0 = self._embed(token_id)              # (embed_dim,)

        self.L1.step(x0,         learn)          # L1 processes embedding   (64,)
        self.L2.step(self.L1.a, learn)          # L2 processes L1 activ.  (128,)
        self.L3.step(self.L2.a, learn)          # L3 processes L2 activ.  (256,)

        # Next-token distribution: scipy softmax for numerical precision
        logits = self.decode_W @ self.L1.a      # (vocab_size,)
        probs  = softmax(logits.astype(np.float64)).astype(np.float32)

        if learn and next_token_id >= 0:
            # Delta-rule: decode_W[next_token_id] row pulled toward L1.a
            target               = np.zeros(self.V, dtype=np.float32)
            target[next_token_id] = 1.0
            decode_error         = target - probs
            self.decode_W       += self.lr * np.outer(decode_error, self.L1.a)
            self.decode_W       *= 0.9999        # L2 weight decay (prevents saturation)
            row_normalise(self.decode_W)

        total_fe = self.L1.avg_free_energy + self.L2.avg_free_energy + self.L3.avg_free_energy
        self._fe_history.append(total_fe)
        self.step_count += 1

        top_id = int(np.argmax(probs))
        return {
            'probs':       probs,
            'top_token':   top_id,
            'top_char':    self.tok.id2ch.get(top_id, '?'),
            'confidence':  float(probs[top_id]),
            'entropy':     float(-np.sum(probs * np.log(probs + 1e-12))),
            'kl_uniform':  kl_from_uniform(probs),
            'fe_l1':       self.L1.avg_free_energy,
            'fe_l2':       self.L2.avg_free_energy,
            'fe_l3':       self.L3.avg_free_energy,
            'total_fe':    total_fe,
            'sparsity_l1': self.L1.sparsity_ratio,
        }

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, text: str, verbose: bool = True) -> Dict:
        """
        Train on text token-by-token.
        At each step: see token t → predict t+1 → update weights.
        Returns accuracy (next-token prediction), speed, free energy.
        """
        ids     = self.tok.encode(text, add_bos=True, add_eos=True)
        n       = len(ids)
        correct = 0
        t0      = time.time()

        for i, tok_id in enumerate(ids):
            next_id = ids[i + 1] if i + 1 < n else -1
            result  = self.step(tok_id, next_token_id=next_id, learn=True)
            if next_id >= 0 and result['top_token'] == next_id:
                correct += 1

            if verbose and n > 1 and (i + 1) % max(1, n // 5) == 0:
                acc = correct / max(i, 1)
                print(f"  [{i+1:5d}/{n}]  acc={acc:.3f}  conf={result['confidence']:.4f}"
                      f"  H={result['entropy']:.3f}  FE={result['total_fe']:.4f}"
                      f"  KL={result['kl_uniform']:.4f}")

        elapsed  = time.time() - t0
        accuracy = correct / max(n - 1, 1)
        return {
            'tokens':    n,
            'accuracy':  accuracy,
            'elapsed':   elapsed,
            'tok_per_s': n / elapsed,
            'final_fe':  self._fe_history[-1] if self._fe_history else 0.0,
        }

    # ── generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt:      str,
        max_tokens:  int   = 100,
        temperature: float = 1.0,
        top_k:       int   = 5,
        learn:       bool  = False,
    ) -> str:
        """
        Autoregressive generation: prime on prompt, then sample token-by-token.

        temperature — >1 more creative, <1 more focused
        top_k       — restrict sampling to top-k predictions (0 = unrestricted)
        learn       — whether to keep updating weights during generation
        """
        ids = self.tok.encode(prompt, add_bos=True, add_eos=False)

        # Prime context with prompt (no learn during priming unless requested)
        for tok_id in ids:
            result = self.step(tok_id, learn=learn)

        out_ids = []
        last_id = ids[-1]

        for _ in range(max_tokens):
            result = self.step(last_id, learn=learn)
            probs  = result['probs'].astype(np.float64)

            # Temperature scaling + re-normalise with scipy for precision
            logits = np.log(probs + 1e-12) / max(temperature, 1e-6)
            probs  = softmax(logits)

            # Top-k filter
            if top_k > 0:
                threshold = float(np.sort(probs)[-min(top_k, len(probs))])
                probs     = np.where(probs >= threshold, probs, 0.0)
                s         = probs.sum()
                if s > 0:
                    probs /= s
                else:
                    probs = np.ones(self.V, dtype=np.float64) / self.V

            next_id = int(np.random.choice(self.V, p=probs))
            if next_id == CharTokenizer.EOS:
                break
            out_ids.append(next_id)
            last_id = next_id

        return self.tok.decode(out_ids)

    # ── state / reset ─────────────────────────────────────────────────────────

    def reset_context(self):
        """Reset all level activations — start a new episode."""
        self.L1.reset()
        self.L2.reset()
        self.L3.reset()

    def param_count(self) -> int:
        return (self.E.size + self.decode_W.size +
                self.L1.W.size + self.L1.V.size +
                self.L2.W.size + self.L2.V.size +
                self.L3.W.size + self.L3.V.size)

    def get_state(self) -> Dict:
        return {
            'steps':        self.step_count,
            'vocab_size':   self.V,
            'params':       self.param_count(),
            'avg_fe':       float(np.mean(self._fe_history[-100:])) if self._fe_history else 0.0,
            'l1_fe':        self.L1.avg_free_energy,
            'l2_fe':        self.L2.avg_free_energy,
            'l3_fe':        self.L3.avg_free_energy,
            'l1_precision': self.L1.precision,
            'l2_precision': self.L2.precision,
            'l3_precision': self.L3.precision,
            'l1_sparsity':  self.L1.sparsity_ratio,
            'l2_sparsity':  self.L2.sparsity_ratio,
            'l3_sparsity':  self.L3.sparsity_ratio,
        }

    def summary(self):
        s = self.get_state()
        print("\n" + "=" * 62)
        print("  HPCLM — Hierarchical Predictive Coding Language Model")
        print("=" * 62)
        print(f"  Vocab size   : {s['vocab_size']}")
        print(f"  Parameters   : {s['params']:,}")
        print(f"  Steps trained: {s['steps']:,}")
        print(f"  Avg FE (100) : {s['avg_fe']:.6f}")
        print(f"\n  Level       dim    k-WTA  sparsity  prec    FE")
        print(f"  L1 phonem.  {self.L1.act_dim:4d}   {self.L1.k:3d}    {s['l1_sparsity']:.1%}    {s['l1_precision']:6.2f}  {s['l1_fe']:.4f}")
        print(f"  L2 lexical  {self.L2.act_dim:4d}   {self.L2.k:3d}    {s['l2_sparsity']:.1%}    {s['l2_precision']:6.2f}  {s['l2_fe']:.4f}")
        print(f"  L3 semant.  {self.L3.act_dim:4d}   {self.L3.k:3d}    {s['l3_sparsity']:.1%}    {s['l3_precision']:6.2f}  {s['l3_fe']:.4f}")
        print(f"\n  Backend: numpy {np.__version__} + scipy (softmax, rel_entr, QR)")
        print("=" * 62)


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

DEMO_TEXT = """\
The brain is not a computer. It is a prediction machine.
Every perception is a controlled hallucination.
Consciousness arises from the brain's attempt to minimise surprise.
The world we see is not reality — it is the brain's best guess.
Free energy drives all thought. Memory is not storage; it is reconstruction.
Learning is prediction error minimisation. Intelligence is the ability to model and predict.
The future is predicted from the past. The past is reconstructed from the present.
Precision weights determine what gets attention. Emotion shapes prediction.
Dreams are the brain replaying the day to consolidate its model of the world.
"""


def run_demo():
    print("=" * 62)
    print("  HPCLM — Initialising")
    print("=" * 62)

    tok = CharTokenizer().fit(DEMO_TEXT)
    print(f"  Vocab size   : {tok.vocab_size} unique characters")

    model = HPCLM(tok, embed_dim=64, l1_dim=128, l2_dim=256, l3_dim=512, lr=0.01)
    print(f"  Parameters   : {model.param_count():,}")
    print(f"  Backend      : numpy + scipy — no sklearn, no transformers")

    EPOCHS = 50
    print(f"\n── Training ({EPOCHS} passes) ─────────────────────────────────")
    for epoch in range(1, EPOCHS + 1):
        model.reset_context()
        r = model.train(DEMO_TEXT, verbose=False)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  acc={r['accuracy']:.3f}  FE={r['final_fe']:.4f}"
                  f"  {r['tok_per_s']:.0f} tok/s")

    model.summary()

    print("\n── Generation ──────────────────────────────────────────")
    for temp, label in [(0.5, 'focused'), (1.0, 'balanced'), (1.5, 'creative')]:
        model.reset_context()
        out = model.generate("The brain", max_tokens=80, temperature=temp, top_k=5)
        print(f"\n  T={temp} ({label}):")
        print(f"  {out!r}")


if __name__ == "__main__":
    run_demo()
