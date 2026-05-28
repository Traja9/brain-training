"""
Hybrid_LM_v2.py — Upgraded Hybrid Language Model

Improvements over v1:
  1. Trainable embeddings      — Adam updates per token (was fixed)
  2. TP signal to all 3 layers — L1/L2/L3 all get global backprop signal (was L1 only)
  3. Deeper PyTorch head       — 3 linear layers instead of 2
  4. Larger default dims       — L1=256 L2=512 L3=512 (was 64/128/128)
  5. Longer context buffer     — 128 tokens (was 32)

Architecture:
  Token ──► Embedding (trainable) ──► ContextBuffer(128) ──► L1 (AdamHebbian+TP)
                                                                  L2 (AdamHebbian+TP)
                                                                  L3 (AdamHebbian+TP)
                                                                       │
                                                           concat(L1 ‖ L2 ‖ L3) [1280-d]
                                                                       │
                                                          PyTorch head (GPU):
                                                          1280 → 512 (ReLU)
                                                           512 → 256 (ReLU)
                                                           256 → vocab (Softmax)
"""

import json
import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from HPCLM import CharTokenizer, BPETokenizer, k_wta, row_normalise, kl_from_uniform
from APEX_LM import AdamHebbianLevel, ContextBuffer


class HybridLM_v2:

    def __init__(
        self,
        tokenizer:      Union[CharTokenizer, BPETokenizer],
        embed_dim:      int   = 64,
        l1_dim:         int   = 256,
        l2_dim:         int   = 512,
        l3_dim:         int   = 512,
        l1_sparsity:    float = 0.05,
        l2_sparsity:    float = 0.04,
        l3_sparsity:    float = 0.03,
        head_dim:       int   = 512,
        head_dim2:      int   = 256,
        lr:             float = 0.01,
        head_lr:        float = 0.001,
        embed_lr:       float = 0.001,
        context_len:    int   = 128,
        context_blend:  float = 0.4,
        beta1:          float = 0.9,
        beta2:          float = 0.999,
        tp_scale:       float = 0.3,
        seed:           int   = 42,
        device:         Optional[str] = None,
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
        self.embed_lr   = embed_lr
        self.beta1      = beta1
        self.beta2      = beta2
        self.tp_scale   = tp_scale
        self.step_count = 0

        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        rng = np.random.RandomState(seed)

        # Trainable token embeddings + Adam moments
        self.E  = (rng.randn(self.V, embed_dim) * 0.1).astype(np.float32)
        row_normalise(self.E)
        self.mE = np.zeros_like(self.E)
        self.vE = np.zeros_like(self.E)
        self.t_embed = 0

        # Adam-Hebbian predictive coding hierarchy
        self.L1 = AdamHebbianLevel(embed_dim, l1_dim, l1_sparsity, lr, beta1, beta2, seed=seed+1)
        self.L2 = AdamHebbianLevel(l1_dim,   l2_dim, l2_sparsity, lr, beta1, beta2, seed=seed+2)
        self.L3 = AdamHebbianLevel(l2_dim,   l3_dim, l3_sparsity, lr, beta1, beta2, seed=seed+3)

        # Context buffer (longer window)
        self.ctx = ContextBuffer(embed_dim, capacity=context_len, blend=context_blend)

        # Deeper PyTorch decode head on GPU
        torch.manual_seed(seed + 4)
        self.head = nn.Sequential(
            nn.Linear(self.concat_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim2),
            nn.ReLU(),
            nn.Linear(head_dim2, self.V),
        ).to(self.device)
        self.head_opt = torch.optim.Adam(self.head.parameters(), lr=head_lr)
        self.ce_loss  = nn.CrossEntropyLoss()

        self._fe_history: List[float] = []

    # ── embedding update (Adam) ───────────────────────────────────────────────

    def _embed_update(self, tok_id: int, grad: np.ndarray):
        self.t_embed += 1
        bc1 = 1.0 - self.beta1 ** self.t_embed
        bc2 = 1.0 - self.beta2 ** self.t_embed
        g = grad.astype(np.float32)
        self.mE[tok_id] = self.beta1 * self.mE[tok_id] + (1.0 - self.beta1) * g
        self.vE[tok_id] = self.beta2 * self.vE[tok_id] + (1.0 - self.beta2) * g ** 2
        self.E[tok_id] += self.embed_lr * (self.mE[tok_id] / bc1) / (
            np.sqrt(self.vE[tok_id] / bc2) + 1e-8
        )
        n = np.linalg.norm(self.E[tok_id]) + 1e-8
        self.E[tok_id] /= n

    # ── core step ─────────────────────────────────────────────────────────────

    def step(self, token_id: int, next_token_id: int = -1, learn: bool = True) -> Dict:
        # 1. Embed + context
        x0    = self.E[token_id].copy()
        x_ctx = self.ctx.augment(x0)
        self.ctx.push(x0)

        # 2. Hebbian forward
        self.L1.forward(x_ctx)
        self.L2.forward(self.L1.a)
        self.L3.forward(self.L2.a)

        # 3. Concat activations
        concat = np.concatenate([self.L1.a, self.L2.a, self.L3.a]).astype(np.float32)

        # 4. Head forward + backward
        if learn and next_token_id >= 0:
            x_t = torch.tensor(concat[None], device=self.device, requires_grad=True)
            logits = self.head(x_t)
            y_t = torch.tensor([next_token_id], device=self.device)
            loss = self.ce_loss(logits, y_t)
            self.head_opt.zero_grad()
            loss.backward()
            self.head_opt.step()
            inp_grad = x_t.grad[0].cpu().numpy()

            # TP to all three Hebbian layers
            l1e = self.l1_dim
            l2e = l1e + self.l2_dim

            tp1 = k_wta(np.clip(self.L1.a - self.tp_scale * inp_grad[:l1e], -1.0, 1.0), self.L1.k)
            tp2 = k_wta(np.clip(self.L2.a - self.tp_scale * inp_grad[l1e:l2e], -1.0, 1.0), self.L2.k)
            tp3 = k_wta(np.clip(self.L3.a - self.tp_scale * inp_grad[l2e:], -1.0, 1.0), self.L3.k)
            self.L1.update(tp1)
            self.L2.update(tp2)
            self.L3.update(tp3)

            # Embedding gradient: propagate TP signal back through L1
            embed_grad = (1.0 - self.ctx.blend) * (self.L1.W.T @ inp_grad[:l1e])
            self._embed_update(token_id, embed_grad)

            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

        elif learn:
            x_t = torch.tensor(concat[None], device=self.device)
            with torch.no_grad():
                logits = self.head(x_t)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            self.L1.update(); self.L2.update(); self.L3.update()

        else:
            x_t = torch.tensor(concat[None], device=self.device)
            with torch.no_grad():
                logits = self.head(x_t)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        total_fe = self.L1.avg_free_energy + self.L2.avg_free_energy + self.L3.avg_free_energy
        self._fe_history.append(total_fe)
        if len(self._fe_history) > 200:
            self._fe_history.pop(0)
        self.step_count += 1

        top_id = int(np.argmax(probs))
        return {
            'probs':      probs,
            'top_token':  top_id,
            'top_char':   self.tok.id2ch.get(top_id, '?'),
            'confidence': float(probs[top_id]),
            'entropy':    float(-np.sum(probs * np.log(probs + 1e-12))),
            'total_fe':   total_fe,
        }

    # ── generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt:             str,
        max_tokens:         int   = 150,
        temperature:        float = 0.8,
        top_k:              int   = 8,
        repetition_penalty: float = 0.85,
    ) -> str:
        self.head.eval()
        ids = self.tok.encode(prompt, add_bos=True, add_eos=False)
        for tid in ids:
            self.step(tid, learn=False)

        out_ids: List[int] = []
        last_id = ids[-1]
        special = {CharTokenizer.PAD, CharTokenizer.UNK, CharTokenizer.BOS}

        for _ in range(max_tokens):
            result = self.step(last_id, learn=False)
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

        self.head.train()
        return self.tok.decode(out_ids)

    # ── helpers ───────────────────────────────────────────────────────────────

    def reset_context(self):
        self.L1.reset(); self.L2.reset(); self.L3.reset(); self.ctx.reset()

    def param_count(self) -> int:
        hebb = (self.E.size
                + self.L1.W.size + self.L1.V.size
                + self.L2.W.size + self.L2.V.size
                + self.L3.W.size + self.L3.V.size)
        head = sum(p.numel() for p in self.head.parameters())
        return hebb + head

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path: str):
        os_path = path if path.endswith('.pt') else path + '.pt'
        torch.save(self.head.state_dict(), os_path)

        np_path = path.replace('.pt', '') + '_hebb.npz'
        if isinstance(self.tok, BPETokenizer):
            tok_data = {'type': 'bpe', 'ch2id': self.tok.ch2id, 'merges': self.tok.merges}
        else:
            tok_data = {'type': 'char', 'ch2id': self.tok.ch2id}
        vocab_bytes = json.dumps(tok_data).encode('utf-8')

        np.savez(
            np_path,
            E=self.E, mE=self.mE, vE=self.vE,
            L1_W=self.L1.W, L1_V=self.L1.V,
            L2_W=self.L2.W, L2_V=self.L2.V,
            L3_W=self.L3.W, L3_V=self.L3.V,
            embed_dim  = np.array(self.embed_dim),
            l1_dim     = np.array(self.l1_dim),
            l2_dim     = np.array(self.l2_dim),
            l3_dim     = np.array(self.l3_dim),
            l1_k       = np.array(self.L1.k),
            l2_k       = np.array(self.L2.k),
            l3_k       = np.array(self.L3.k),
            lr         = np.array(self.lr),
            head_lr    = np.array(self.head_lr),
            embed_lr   = np.array(self.embed_lr),
            beta1      = np.array(self.beta1),
            beta2      = np.array(self.beta2),
            tp_scale   = np.array(self.tp_scale),
            ctx_cap    = np.array(self.ctx.capacity),
            ctx_blend  = np.array(self.ctx.blend),
            step_count = np.array(self.step_count),
            t_embed    = np.array(self.t_embed),
            vocab      = np.frombuffer(vocab_bytes, dtype=np.uint8),
        )

    @classmethod
    def load(cls, path: str, device=None) -> 'HybridLM_v2':
        pt_path  = path if path.endswith('.pt') else path + '.pt'
        np_path  = path.replace('.pt', '') + '_hebb.npz'
        d = np.load(np_path, allow_pickle=False)

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

        l1_dim = int(d['l1_dim']); l2_dim = int(d['l2_dim']); l3_dim = int(d['l3_dim'])

        model = cls(
            tok,
            embed_dim    = int(d['embed_dim']),
            l1_dim       = l1_dim,
            l2_dim       = l2_dim,
            l3_dim       = l3_dim,
            l1_sparsity  = float(d['l1_k']) / l1_dim,
            l2_sparsity  = float(d['l2_k']) / l2_dim,
            l3_sparsity  = float(d['l3_k']) / l3_dim,
            lr           = float(d['lr']),
            head_lr      = float(d['head_lr']),
            embed_lr     = float(d['embed_lr']),
            beta1        = float(d['beta1']),
            beta2        = float(d['beta2']),
            tp_scale     = float(d['tp_scale']),
            context_len  = int(d['ctx_cap']),
            context_blend= float(d['ctx_blend']),
            device       = device,
        )
        model.E          = d['E'].copy()
        model.mE         = d['mE'].copy()
        model.vE         = d['vE'].copy()
        model.step_count = int(d['step_count'])
        model.t_embed    = int(d['t_embed'])

        for lv, nm in [(model.L1, 'L1'), (model.L2, 'L2'), (model.L3, 'L3')]:
            lv.W = d[f'{nm}_W'].copy()
            lv.V = d[f'{nm}_V'].copy()

        model.head.load_state_dict(
            torch.load(pt_path, map_location=model.device)
        )
        return model

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self):
        hebb_p = (self.E.size
                  + self.L1.W.size + self.L1.V.size
                  + self.L2.W.size + self.L2.V.size
                  + self.L3.W.size + self.L3.V.size)
        head_p = sum(p.numel() for p in self.head.parameters())
        print("\n" + "=" * 68)
        print("  Hybrid_LM v2 — Trainable Embeddings + TP-All + Deeper Head")
        print("=" * 68)
        print(f"  Device         : {self.device}")
        print(f"  Vocab size     : {self.V}")
        print(f"  Total params   : {self.param_count():,}  (Hebbian {hebb_p:,} + Head {head_p:,})")
        print(f"  Steps trained  : {self.step_count:,}")
        print(f"  Context buffer : {self.ctx.capacity} tokens  blend={self.ctx.blend}")
        print(f"  TP scale       : {self.tp_scale}")
        avg_fe = float(np.mean(self._fe_history[-100:])) if self._fe_history else 0.0
        print(f"  Avg FE (100)   : {avg_fe:.6f}")
        print(f"\n  Hebbian levels (all get TP signal):")
        print(f"    L1 {self.l1_dim:4d}d  k={self.L1.k}  prec={self.L1.precision:.2f}")
        print(f"    L2 {self.l2_dim:4d}d  k={self.L2.k}  prec={self.L2.precision:.2f}")
        print(f"    L3 {self.l3_dim:4d}d  k={self.L3.k}  prec={self.L3.precision:.2f}")
        print(f"\n  Head: {self.concat_dim}→512(ReLU)→256(ReLU)→{self.V}(Softmax)")
        print(f"  Embed LR: {self.embed_lr}  (trainable embeddings ON)")
        print("=" * 68)


import os as _os
