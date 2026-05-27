"""
Hybrid_LM.py — Hybrid Hebbian + Backprop Language Model

Architecture:
  Token ──► Embedding ──► ContextBuffer ──► L1 (AdamHebbian)
                                                │  L2 (AdamHebbian)
                                                │  L3 (AdamHebbian)
                                                │
                                     concat(L1.a ‖ L2.a ‖ L3.a)
                                                │
                                    NeuralDataProcessor (backprop)
                                      concat_dim → head_dim (ReLU)
                                      head_dim   → vocab_size (Softmax)
                                                │
                                        next-token probs

Learning:
  head  — full backprop + Adam (NeuralDataProcessor)
  L1    — Adam-Hebbian + target propagation via head input gradient
  L2/L3 — Adam-Hebbian local
"""

import json
import time
from typing import Dict, List, Optional, Union

import numpy as np
from scipy.special import softmax

from HPCLM import (
    CharTokenizer, BPETokenizer,
    k_wta, row_normalise, ortho_init, kl_from_uniform,
)
from APEX_LM import AdamHebbianLevel, ContextBuffer
from Neural_Algorithm import (
    NeuralDataProcessor, LayerConfig, ActivationFunction, LossFunction,
)


# ──────────────────────────────────────────────────────────────────────────────
# Head backward helper — computes input gradient NeuralDataProcessor.backward()
# deliberately omits (the `if i > 0` guard stops at the first layer).
# ──────────────────────────────────────────────────────────────────────────────

def _head_backward_and_input_grad(
    head: NeuralDataProcessor,
    y_true: np.ndarray,
):
    """
    Full backprop through head plus gradient w.r.t. its input vector.

    Returns (weight_grads, bias_grads, input_grad) where input_grad has the
    same shape as the concat(L1.a, L2.a, L3.a) vector fed into head.forward().
    """
    # CE + Softmax gradient: δ = ŷ − y  (exact, no Jacobian needed)
    delta = (head.layer_outputs[-1] - y_true).astype(np.float64)

    wg = [None] * len(head.weights)
    bg = [None] * len(head.biases)

    for i in reversed(range(len(head.layer_configs))):
        wg[i] = np.outer(delta, head.layer_outputs[i])
        if head.biases[i] is not None:
            bg[i] = delta.copy()
        # Propagate through weights (always, even at i = 0)
        delta = head.weights[i].T @ delta
        if i > 0:
            delta *= head._activation_derivative(
                head.layer_inputs[i],
                head.layer_configs[i - 1].activation,
            )

    # delta is now ∂L/∂input — shape = (l1_dim + l2_dim + l3_dim,)
    return wg, bg, delta.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid_LM
# ──────────────────────────────────────────────────────────────────────────────

class Hybrid_LM:
    """
    Hybrid Language Model — Hebbian sparse hierarchy + backprop decode head.

    Why this works better than pure-Hebbian APEX_LM:
      - The backprop head gets an exact cross-entropy gradient signal every step,
        not just the noisy Hebbian TP approximation.
      - The head's input gradient is propagated back to L1 as a TP target,
        giving L1 a supervised signal from full backprop without running full
        backprop through the Hebbian weights themselves.
      - L2/L3 remain purely local (Hebbian), preserving biological plausibility
        and avoiding the vanishing-gradient problem in deep sparse networks.
    """

    def __init__(
        self,
        tokenizer:      Union[CharTokenizer, BPETokenizer],
        embed_dim:      int   = 64,
        l1_dim:         int   = 128,
        l2_dim:         int   = 256,
        l3_dim:         int   = 512,
        l1_sparsity:    float = 0.05,
        l2_sparsity:    float = 0.04,
        l3_sparsity:    float = 0.03,
        head_dim:       int   = 256,
        lr:             float = 0.01,
        head_lr:        float = 0.001,
        context_len:    int   = 32,
        context_blend:  float = 0.4,
        beta1:          float = 0.9,
        beta2:          float = 0.999,
        tp_scale:       float = 0.3,
        seed:           int   = 42,
    ):
        self.tok        = tokenizer
        self.V          = tokenizer.vocab_size
        self.embed_dim  = embed_dim
        self.l1_dim     = l1_dim
        self.l2_dim     = l2_dim
        self.l3_dim     = l3_dim
        self.concat_dim = l1_dim + l2_dim + l3_dim
        self.lr         = lr
        self.head_lr    = head_lr
        self.beta1      = beta1
        self.beta2      = beta2
        self.tp_scale   = tp_scale

        rng = np.random.RandomState(seed)

        # Token embeddings (fixed after init, row-normalised)
        self.E = (rng.randn(self.V, embed_dim) * 0.1).astype(np.float32)
        row_normalise(self.E)

        # Adam-Hebbian predictive coding hierarchy
        self.L1 = AdamHebbianLevel(embed_dim, l1_dim, l1_sparsity, lr, beta1, beta2, seed=seed+1)
        self.L2 = AdamHebbianLevel(l1_dim,   l2_dim, l2_sparsity, lr, beta1, beta2, seed=seed+2)
        self.L3 = AdamHebbianLevel(l2_dim,   l3_dim, l3_sparsity, lr, beta1, beta2, seed=seed+3)

        # Kernel-attention context buffer
        self.ctx = ContextBuffer(embed_dim, capacity=context_len, blend=context_blend)

        # Backprop decode head — seed global RNG so init is deterministic
        np.random.seed(seed + 4)
        self.head = NeuralDataProcessor(
            layer_configs=[
                LayerConfig(self.concat_dim, head_dim, ActivationFunction.RELU),
                LayerConfig(head_dim,        self.V,   ActivationFunction.SOFTMAX),
            ],
            learning_rate        = head_lr,
            loss_function        = LossFunction.CROSS_ENTROPY,
            use_precision_calculus = False,
            optimization_method  = 'adam',
        )

        self.step_count       = 0
        self._fe_history: List[float] = []   # capped at 200

    # ── embedding ─────────────────────────────────────────────────────────────

    def _embed(self, token_id: int) -> np.ndarray:
        return self.E[token_id].copy()

    # ── core step ─────────────────────────────────────────────────────────────

    def step(self, token_id: int, next_token_id: int = -1, learn: bool = True) -> Dict:
        # 1. Embed + context augmentation
        x0    = self._embed(token_id)
        x_ctx = self.ctx.augment(x0)
        self.ctx.push(x0)

        # 2. Hebbian forward (no weight update yet)
        self.L1.forward(x_ctx)
        self.L2.forward(self.L1.a)
        self.L3.forward(self.L2.a)

        # 3. Concat activations → head input
        concat = np.concatenate([self.L1.a, self.L2.a, self.L3.a]).astype(np.float32)

        # 4. Backprop head forward
        probs = self.head.forward(concat).astype(np.float32)

        if learn and next_token_id >= 0:
            # 5. Target one-hot
            y_true = np.zeros(self.V, dtype=np.float32)
            y_true[next_token_id] = 1.0

            # 6. Full backprop through head + input gradient
            wg, bg, input_grad = _head_backward_and_input_grad(self.head, y_true)

            # 7. Update head weights with Adam
            self.head._update_parameters_adam(wg, bg)

            # 8. Target propagation: use head's gradient to supervise L1
            #    input_grad[:l1_dim] = ∂L/∂L1.a
            #    Desired L1.a = L1.a − tp_scale · ∂L/∂L1.a   (gradient descent)
            tp_raw    = self.L1.a - self.tp_scale * input_grad[:self.l1_dim]
            tp_target = k_wta(np.clip(tp_raw, -1.0, 1.0), self.L1.k)
            self.L1.update(tp_target)

            # 9. L2, L3: local Adam-Hebbian (no TP signal)
            self.L2.update()
            self.L3.update()

        elif learn:
            # No target → plain Adam-Hebbian everywhere
            self.L1.update()
            self.L2.update()
            self.L3.update()

        total_fe = (self.L1.avg_free_energy
                    + self.L2.avg_free_energy
                    + self.L3.avg_free_energy)
        self._fe_history.append(total_fe)
        if len(self._fe_history) > 200:
            self._fe_history.pop(0)
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
            'total_fe':    total_fe,
            'sparsity_l1': self.L1.sparsity_ratio,
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

            if verbose and n > 1 and (i + 1) % max(1, n // 5) == 0:
                acc = correct / max(i, 1)
                print(f"  [{i+1:5d}/{n}]  acc={acc:.3f}  conf={result['confidence']:.4f}  "
                      f"FE={result['total_fe']:.4f}")

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
        prompt:             str,
        max_tokens:         int   = 100,
        temperature:        float = 1.0,
        top_k:              int   = 5,
        repetition_penalty: float = 0.85,
        learn:              bool  = False,
    ) -> str:
        ids = self.tok.encode(prompt, add_bos=True, add_eos=False)
        for tok_id in ids:
            self.step(tok_id, learn=learn)

        out_ids: List[int] = []
        last_id = ids[-1]
        special = {CharTokenizer.PAD, CharTokenizer.UNK, CharTokenizer.BOS}

        for _ in range(max_tokens):
            result = self.step(last_id, learn=learn)
            probs  = result['probs'].astype(np.float64)

            for sid in special:
                probs[sid] = 0.0
            probs[CharTokenizer.EOS] = 0.0

            if repetition_penalty < 1.0 and out_ids:
                from collections import Counter
                counts = Counter(out_ids[-20:])
                for prev_id, cnt in counts.items():
                    probs[prev_id] *= repetition_penalty ** cnt

            s = probs.sum()
            probs = probs / s if s > 1e-12 else np.ones(self.V) / self.V

            logits = np.log(probs + 1e-12) / max(temperature, 1e-6)
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
        self.ctx.reset()

    def param_count(self) -> int:
        hebb = (self.E.size
                + self.L1.W.size + self.L1.V.size
                + self.L2.W.size + self.L2.V.size
                + self.L3.W.size + self.L3.V.size)
        head = (sum(w.size for w in self.head.weights)
                + sum(b.size for b in self.head.biases if b is not None))
        return hebb + head

    def get_state(self) -> Dict:
        return {
            'steps':        self.step_count,
            'vocab_size':   self.V,
            'params':       self.param_count(),
            'avg_fe':       float(np.mean(self._fe_history[-100:])) if self._fe_history else 0.0,
            'l1_precision': self.L1.precision,
            'l2_precision': self.L2.precision,
            'l3_precision': self.L3.precision,
            'l1_sparsity':  self.L1.sparsity_ratio,
        }

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path: str):
        if not path.endswith('.npz'):
            path += '.npz'

        if isinstance(self.tok, BPETokenizer):
            tok_data = {'type': 'bpe', 'ch2id': self.tok.ch2id, 'merges': self.tok.merges}
        else:
            tok_data = {'type': 'char', 'ch2id': self.tok.ch2id}
        vocab_bytes = json.dumps(tok_data).encode('utf-8')

        # Validate head has exactly 2 layers before saving
        assert len(self.head.weights) == 2, "Expected 2-layer head"

        def _bias_or_zeros(b, ref):
            return b if b is not None else np.zeros(ref.shape[0])

        np.savez(
            path,
            # Embedding
            E           = self.E,
            # Hebbian levels
            L1_W=self.L1.W,   L1_V=self.L1.V,
            L1_mW=self.L1.mW, L1_vW=self.L1.vW,
            L1_mV=self.L1.mV, L1_vV=self.L1.vV,
            L2_W=self.L2.W,   L2_V=self.L2.V,
            L2_mW=self.L2.mW, L2_vW=self.L2.vW,
            L2_mV=self.L2.mV, L2_vV=self.L2.vV,
            L3_W=self.L3.W,   L3_V=self.L3.V,
            L3_mW=self.L3.mW, L3_vW=self.L3.vW,
            L3_mV=self.L3.mV, L3_vV=self.L3.vV,
            # Head weights / biases / Adam moments
            head_w0 = self.head.weights[0],
            head_w1 = self.head.weights[1],
            head_b0 = _bias_or_zeros(self.head.biases[0],  self.head.weights[0]),
            head_b1 = _bias_or_zeros(self.head.biases[1],  self.head.weights[1]),
            head_mw0 = self.head.m_weights[0],
            head_mw1 = self.head.m_weights[1],
            head_vw0 = self.head.v_weights[0],
            head_vw1 = self.head.v_weights[1],
            head_mb0 = _bias_or_zeros(self.head.m_biases[0], self.head.weights[0]),
            head_mb1 = _bias_or_zeros(self.head.m_biases[1], self.head.weights[1]),
            head_vb0 = _bias_or_zeros(self.head.v_biases[0], self.head.weights[0]),
            head_vb1 = _bias_or_zeros(self.head.v_biases[1], self.head.weights[1]),
            # Scalars
            step_count   = np.array(self.step_count),
            head_adam_t  = np.array(self.head.adam_t),
            L1_t         = np.array(self.L1.t),
            L2_t         = np.array(self.L2.t),
            L3_t         = np.array(self.L3.t),
            L1_precision = np.array(self.L1.precision),
            L2_precision = np.array(self.L2.precision),
            L3_precision = np.array(self.L3.precision),
            # Config
            embed_dim  = np.array(self.embed_dim),
            l1_dim     = np.array(self.l1_dim),
            l2_dim     = np.array(self.l2_dim),
            l3_dim     = np.array(self.l3_dim),
            head_dim   = np.array(self.head.weights[0].shape[0]),
            l1_k       = np.array(self.L1.k),
            l2_k       = np.array(self.L2.k),
            l3_k       = np.array(self.L3.k),
            lr         = np.array(self.lr),
            head_lr    = np.array(self.head_lr),
            beta1      = np.array(self.beta1),
            beta2      = np.array(self.beta2),
            tp_scale   = np.array(self.tp_scale),
            ctx_cap    = np.array(self.ctx.capacity),
            ctx_blend  = np.array(self.ctx.blend),
            vocab      = np.frombuffer(vocab_bytes, dtype=np.uint8),
        )

    @classmethod
    def load(cls, path: str) -> 'Hybrid_LM':
        if not path.endswith('.npz'):
            path += '.npz'
        d = np.load(path, allow_pickle=False)

        tok_data = json.loads(bytes(d['vocab']).decode('utf-8'))
        if tok_data.get('type') == 'bpe':
            tok = BPETokenizer()
            tok.ch2id  = tok_data['ch2id']
            tok.id2ch  = {int(v): k for k, v in tok.ch2id.items()}
            tok.merges = [tuple(m) for m in tok_data.get('merges', [])]
        else:
            tok = CharTokenizer()
            tok.ch2id = tok_data['ch2id']
            tok.id2ch = {int(v): k for k, v in tok.ch2id.items()}

        l1_dim = int(d['l1_dim'])
        l2_dim = int(d['l2_dim'])
        l3_dim = int(d['l3_dim'])

        model = cls(
            tok,
            embed_dim     = int(d['embed_dim']),
            l1_dim        = l1_dim,
            l2_dim        = l2_dim,
            l3_dim        = l3_dim,
            l1_sparsity   = float(d['l1_k']) / l1_dim,
            l2_sparsity   = float(d['l2_k']) / l2_dim,
            l3_sparsity   = float(d['l3_k']) / l3_dim,
            head_dim      = int(d['head_dim']),
            lr            = float(d['lr']),
            head_lr       = float(d['head_lr']),
            beta1         = float(d['beta1']),
            beta2         = float(d['beta2']),
            tp_scale      = float(d['tp_scale']),
            context_len   = int(d['ctx_cap']),
            context_blend = float(d['ctx_blend']),
        )
        model.E          = d['E']
        model.step_count = int(d['step_count'])

        for lv, nm in [(model.L1, 'L1'), (model.L2, 'L2'), (model.L3, 'L3')]:
            lv.W         = d[f'{nm}_W'];  lv.V  = d[f'{nm}_V']
            lv.mW        = d[f'{nm}_mW']; lv.vW = d[f'{nm}_vW']
            lv.mV        = d[f'{nm}_mV']; lv.vV = d[f'{nm}_vV']
            lv.t         = int(d[f'{nm}_t'])
            lv.precision = float(d[f'{nm}_precision'])

        model.head.weights[0]   = d['head_w0']
        model.head.weights[1]   = d['head_w1']
        model.head.biases[0]    = d['head_b0']
        model.head.biases[1]    = d['head_b1']
        model.head.m_weights[0] = d['head_mw0']
        model.head.m_weights[1] = d['head_mw1']
        model.head.v_weights[0] = d['head_vw0']
        model.head.v_weights[1] = d['head_vw1']
        model.head.m_biases[0]  = d['head_mb0']
        model.head.m_biases[1]  = d['head_mb1']
        model.head.v_biases[0]  = d['head_vb0']
        model.head.v_biases[1]  = d['head_vb1']
        model.head.adam_t       = int(d['head_adam_t'])

        return model

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self):
        s = self.get_state()
        hebb_p = (self.E.size
                  + self.L1.W.size + self.L1.V.size
                  + self.L2.W.size + self.L2.V.size
                  + self.L3.W.size + self.L3.V.size)
        head_p = (sum(w.size for w in self.head.weights)
                  + sum(b.size for b in self.head.biases if b is not None))

        print("\n" + "=" * 68)
        print("  Hybrid_LM — Hebbian Hierarchy + Backprop Decode Head")
        print("=" * 68)
        print(f"  Vocab size     : {s['vocab_size']}")
        print(f"  Total params   : {s['params']:,}  "
              f"(Hebbian {hebb_p:,}  +  Head {head_p:,})")
        print(f"  Steps trained  : {s['steps']:,}")
        print(f"  Context buffer : {self.ctx.capacity} tokens  blend={self.ctx.blend}")
        print(f"  TP scale       : {self.tp_scale}")
        print(f"  Avg FE (100)   : {s['avg_fe']:.6f}")
        print(f"\n  Hebbian levels:")
        print(f"    L1 {self.l1_dim:4d}d  k={self.L1.k}  "
              f"prec={s['l1_precision']:.2f}  spars={s['l1_sparsity']:.1%}")
        print(f"    L2 {self.l2_dim:4d}d  k={self.L2.k}  prec={s['l2_precision']:.2f}")
        print(f"    L3 {self.l3_dim:4d}d  k={self.L3.k}  prec={s['l3_precision']:.2f}")
        hd = self.head.weights[0].shape[0]
        print(f"\n  Head: {self.concat_dim}→{hd}(ReLU)→{self.V}(Softmax)  "
              f"head_lr={self.head_lr}")
        print(f"\n  Innovations: Adam-Hebbian · ContextBuffer · Backprop-TP")
        print("=" * 68)


# ──────────────────────────────────────────────────────────────────────────────
# Demo — Hybrid_LM vs APEX_LM vs HPCLM on same text / same epochs
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
    from HPCLM import HPCLM
    from APEX_LM import APEX_LM

    print("=" * 68)
    print("  Hybrid_LM vs APEX_LM vs HPCLM — same data, same epochs")
    print("=" * 68)

    tok    = CharTokenizer().fit(DEMO_TEXT)
    hybrid = Hybrid_LM(tok, embed_dim=64, l1_dim=128, l2_dim=256, l3_dim=512,
                       lr=0.01, head_lr=0.001, seed=42)
    apex   = APEX_LM(tok, embed_dim=64, l1_dim=128, l2_dim=256, l3_dim=512,
                     lr=0.01, seed=42)
    hpclm  = HPCLM(tok, embed_dim=64, l1_dim=128, l2_dim=256, l3_dim=512,
                   lr=0.01, seed=42)

    print(f"  Hybrid params  : {hybrid.param_count():,}")
    print(f"  APEX   params  : {apex.param_count():,}")
    print(f"  HPCLM  params  : {hpclm.param_count():,}\n")

    SPLIT      = int(len(DEMO_TEXT) * 0.9)
    train_text = DEMO_TEXT[:SPLIT]
    test_ids   = tok.encode(DEMO_TEXT[SPLIT:], add_bos=True, add_eos=False)

    def test_acc(model):
        model.reset_context()
        return (
            sum(
                model.step(test_ids[i], next_token_id=test_ids[i + 1], learn=False)['top_token']
                == test_ids[i + 1]
                for i in range(len(test_ids) - 1)
            )
            / max(len(test_ids) - 1, 1)
        )

    print(f"  {'Ep':>3}  {'Hyb tr':>8}  {'Hyb te':>8}  {'APEX tr':>8}  "
          f"{'APEX te':>8}  {'HPC tr':>8}  {'HPC te':>8}")
    print("  " + "-" * 68)

    for epoch in range(1, 11):
        lr = 0.01 * (0.7 ** (epoch / 10))

        hybrid.lr = hybrid.L1.lr = hybrid.L2.lr = hybrid.L3.lr = lr
        apex.lr   = apex.L1.lr   = apex.L2.lr   = apex.L3.lr   = lr
        hpclm.lr  = hpclm.L1.lr  = hpclm.L2.lr  = hpclm.L3.lr  = lr

        hybrid.reset_context(); apex.reset_context(); hpclm.reset_context()
        hr = hybrid.train(train_text, verbose=False)
        ar = apex.train(train_text,   verbose=False)
        cr = hpclm.train(train_text,  verbose=False)

        h_te = test_acc(hybrid)
        a_te = test_acc(apex)
        c_te = test_acc(hpclm)

        print(f"  {epoch:3d}  {hr['accuracy']:8.3f}  {h_te:8.3f}  "
              f"{ar['accuracy']:8.3f}  {a_te:8.3f}  "
              f"{cr['accuracy']:8.3f}  {c_te:8.3f}")

    hybrid.summary()

    print("\n── Save / Load round-trip ───────────────────────────────────────")
    import os, tempfile
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        tmp = f.name
    hybrid.save(tmp)
    hybrid2 = Hybrid_LM.load(tmp)
    os.unlink(tmp)
    hybrid2.reset_context()
    out1 = hybrid2.generate("The brain", max_tokens=40, temperature=0.8, top_k=5)
    print(f"  Loaded model sample: {('The brain' + out1)!r}")
    print("  Save/load: OK")

    print("\n── Generation ───────────────────────────────────────────────────")
    for temp in [0.7, 1.0]:
        hybrid.reset_context()
        out = hybrid.generate("The brain", max_tokens=60, temperature=temp, top_k=5)
        print(f"  T={temp}: {('The brain' + out)!r}")


if __name__ == '__main__':
    run_demo()
