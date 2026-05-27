"""
Torch_Brain.py — PyTorch × Precious Brain Hybrid

Upgrades the Precious Brain with six deep-learning modules:

  PerceptualEncoder        — residual MLP: raw input → rich latent
  PlasticityController     — neural net: dynamic LTP/LTD gating per context
  AttentiveMemoryBank      — multi-head attention over consciousness memories
  PredictiveCodingModule   — GRU-based free-energy minimisation
  WorkingMemoryTransformer — transformer over working-memory items
  EmotionRegressor         — valence/arousal prediction from latent

PyTorch handles: perception, plasticity gating, memory attention, prediction, emotion.
Brain    handles: Hebbian plasticity, consciousness, consolidation, dream replay, evolution.
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("PyTorch not found.  Install with:  pip install torch")

sys.path.insert(0, os.path.dirname(__file__))
from A_Brain import PreciousBrain, PreciousBrainConfig
from Neural_Consciousness import EmotionType


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Perceptual Encoder
# ──────────────────────────────────────────────────────────────────────────────
class PerceptualEncoder(nn.Module):
    """Residual MLP — compresses raw sensory input into a stable latent vector."""

    def __init__(self, input_size: int, latent_size: int):
        super().__init__()
        h = max(latent_size * 2, 128)
        self.proj  = nn.Linear(input_size, h)
        self.res1  = self._block(h)
        self.res2  = self._block(h)
        self.out   = nn.Linear(h, latent_size)
        self.norm  = nn.LayerNorm(latent_size)
        self.act   = nn.GELU()

    @staticmethod
    def _block(h: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h), nn.LayerNorm(h),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.proj(x))
        h = h + self.act(self.res1(h))
        h = h + self.act(self.res2(h))
        return self.norm(self.out(h))


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Plasticity Controller
# ──────────────────────────────────────────────────────────────────────────────
class PlasticityController(nn.Module):
    """
    Outputs dynamic LTP/LTD thresholds from the current latent context.
    High arousal / novel input  → lower LTP threshold  (easier potentiation)
    Familiar / low-energy input → raise LTP threshold  (harder to overwrite)
    """

    def __init__(self, latent_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, 64), nn.Tanh(),
            nn.Linear(64, 32),          nn.Tanh(),
            nn.Linear(32, 2),           nn.Tanh(),   # output ∈ (−1, 1)
        )

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(latent)
        ltp = (0.7 + 0.2 * out[:, 0]).clamp(0.50, 0.90)   # [0.50, 0.90]
        ltd = (0.3 + 0.15 * out[:, 1]).clamp(0.15, 0.45)  # [0.15, 0.45]
        return ltp, ltd


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Attentive Memory Bank
# ──────────────────────────────────────────────────────────────────────────────
class AttentiveMemoryBank(nn.Module):
    """
    Multi-head attention retrieval over a set of consciousness-memory vectors.
    Returns a context vector that blends the most relevant past experiences
    into the current latent before it reaches form_thought().
    """

    def __init__(self, query_size: int, mem_size: int, n_heads: int = 4):
        super().__init__()
        self.q_proj  = nn.Linear(query_size, mem_size)
        self.k_proj  = nn.Linear(mem_size, mem_size)
        self.v_proj  = nn.Linear(mem_size, mem_size)
        self.attn    = nn.MultiheadAttention(mem_size, n_heads, batch_first=True)
        self.out     = nn.Linear(mem_size, query_size)

    def forward(
        self,
        query: torch.Tensor,           # (B, query_size)
        memories: torch.Tensor,        # (B, N, mem_size)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(query).unsqueeze(1)
        k = self.k_proj(memories)
        v = self.v_proj(memories)
        ctx, weights = self.attn(q, k, v)
        return self.out(ctx.squeeze(1)), weights.squeeze(1)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Predictive Coding Module
# ──────────────────────────────────────────────────────────────────────────────
class PredictiveCodingModule(nn.Module):
    """
    GRU maintains a context across steps and predicts the NEXT latent.
    Prediction error = proxy for surprise / free energy.
    High free energy → signal for heightened plasticity and arousal.
    """

    def __init__(self, latent_size: int, context_size: int = 64):
        super().__init__()
        self.rnn     = nn.GRUCell(latent_size, context_size)
        self.predict = nn.Sequential(
            nn.Linear(context_size, latent_size * 2), nn.GELU(),
            nn.Linear(latent_size * 2, latent_size),
        )
        self.context_size = context_size
        self._h: Optional[torch.Tensor] = None

    def reset(self):
        self._h = None

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = latent.size(0)
        if self._h is None or self._h.size(0) != B:
            self._h = torch.zeros(B, self.context_size, device=latent.device)
        self._h  = self.rnn(latent, self._h.detach())   # detach to truncate BPTT
        pred     = self.predict(self._h)
        free_eng = F.mse_loss(pred, latent.detach())
        return pred, free_eng


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Working Memory Transformer
# ──────────────────────────────────────────────────────────────────────────────
class WorkingMemoryTransformer(nn.Module):
    """
    Transformer encoder over working-memory item embeddings.
    Mean-pools the attended representations into a single context vector
    that can be injected back into the latent path.
    """

    def __init__(self, item_size: int, n_heads: int = 2, n_layers: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=item_size, nhead=n_heads,
            dim_feedforward=item_size * 4,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool        = nn.Linear(item_size, item_size)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        # items: (B, N_items, item_size)
        out = self.transformer(items)
        return self.pool(out.mean(dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Emotion Regressor
# ──────────────────────────────────────────────────────────────────────────────
class EmotionRegressor(nn.Module):
    """
    Predicts continuous valence, arousal, and discrete emotion class
    from the perceptual latent.  These predictions feed back into the
    biological brain's emotion modulation pathway.
    """

    LABELS = [e.value for e in EmotionType]

    def __init__(self, latent_size: int, n_emotions: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, 64), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),          nn.ReLU(),
            nn.Linear(32, n_emotions + 2),
        )
        self.n_emotions = n_emotions

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.net(latent)
        return {
            'emotion_logits': out[:, :self.n_emotions],
            'emotion_probs':  F.softmax(out[:, :self.n_emotions], dim=-1),
            'valence':        torch.tanh(out[:, self.n_emotions]),
            'arousal':        torch.sigmoid(out[:, self.n_emotions + 1]),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TorchBrainConfig:
    brain_config:   PreciousBrainConfig = field(default_factory=PreciousBrainConfig)
    input_size:     int   = 64     # raw sensory input dimension
    latent_size:    int   = 128    # encoder output / thought seed dimension
    mem_embed_size: int   = 64     # memory vector dimension for attention
    wm_item_size:   int   = 32     # working-memory item embedding size
    lr:             float = 1e-3
    plasticity_lr:  float = 5e-4
    device:         str   = 'cuda' if torch.cuda.is_available() else 'cpu'


# ──────────────────────────────────────────────────────────────────────────────
# Main Hybrid Class
# ──────────────────────────────────────────────────────────────────────────────
class TorchPreciousBrain(nn.Module):
    """
    PyTorch × Precious Brain hybrid.

    Forward path:
      raw input
        → PerceptualEncoder            (rich latent)
        → PlasticityController         (sets LTP/LTD thresholds)
        → PredictiveCodingModule       (free-energy surprise signal)
        → AttentiveMemoryBank          (blends relevant past memories)
        → EmotionRegressor             (valence / arousal estimate)
        → Brain.consciousness.form_thought()   (Hebbian thought formation)

    Learning path (learn_step):
      PyTorch: predictive-coding loss + emotion-arousal consistency loss
      Brain:   standard Hebbian LTP/LTD via brain.learn()
    """

    def __init__(self, config: TorchBrainConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # ── Biological brain ──────────────────────────────────────────────────
        self.brain = PreciousBrain(config.brain_config)

        # ── PyTorch modules ───────────────────────────────────────────────────
        self.encoder          = PerceptualEncoder(config.input_size, config.latent_size)
        self.plasticity_ctrl  = PlasticityController(config.latent_size)
        self.memory_attn      = AttentiveMemoryBank(
                                    config.latent_size, config.mem_embed_size)
        self.predictive_coder = PredictiveCodingModule(config.latent_size)
        self.wm_transformer   = WorkingMemoryTransformer(config.wm_item_size)
        self.emotion_reg      = EmotionRegressor(config.latent_size)

        self.to(self.device)

        # ── Optimiser (per-module learning rates) ─────────────────────────────
        self.optimizer = torch.optim.AdamW([
            {'params': self.encoder.parameters(),          'lr': config.lr},
            {'params': self.memory_attn.parameters(),      'lr': config.lr},
            {'params': self.predictive_coder.parameters(), 'lr': config.lr},
            {'params': self.wm_transformer.parameters(),   'lr': config.lr},
            {'params': self.emotion_reg.parameters(),      'lr': config.lr},
            {'params': self.plasticity_ctrl.parameters(),  'lr': config.plasticity_lr},
        ], weight_decay=1e-4)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-5
        )

        self.step_count:         int        = 0
        self.loss_history:       List[float] = []
        self.free_energy_history: List[float] = []

    # ── helpers ───────────────────────────────────────────────────────────────
    def _t(self, x: np.ndarray) -> torch.Tensor:
        t = torch.FloatTensor(x).to(self.device)
        return t.unsqueeze(0) if t.dim() == 1 else t

    def _pad(self, t: torch.Tensor, size: int) -> torch.Tensor:
        if t.size(-1) < size:
            return F.pad(t, (0, size - t.size(-1)))
        return t[..., :size]

    def _memory_tensors(self) -> Optional[torch.Tensor]:
        """Pack up to 50 consciousness memories into (1, N, mem_embed_size)."""
        if not (self.brain.consciousness and self.brain.consciousness.memories):
            return None
        vecs = []
        for m in list(self.brain.consciousness.memories.values())[:50]:
            c = np.array(m.content, dtype=np.float32)
            if c.size < self.config.mem_embed_size:
                c = np.pad(c, (0, self.config.mem_embed_size - c.size))
            else:
                c = c[:self.config.mem_embed_size]
            vecs.append(c)
        return torch.FloatTensor(np.stack(vecs)).unsqueeze(0).to(self.device)

    # ── public API ────────────────────────────────────────────────────────────
    def think(self, raw_input: np.ndarray, context: str = "perception") -> Dict:
        """
        One cognitive step: encode → gate → predict → attend → form_thought.
        Returns a dict with latent, thresholds, free energy, emotion estimates.
        """
        with torch.no_grad():
            x      = self._pad(self._t(raw_input), self.config.input_size)
            latent = self.encoder(x)

            # Dynamic plasticity
            ltp, ltd = self.plasticity_ctrl(latent)
            if self.brain.consciousness:
                self.brain.consciousness.ltp_threshold = float(ltp[0])
                self.brain.consciousness.ltd_threshold = float(ltd[0])

            # Surprise / free energy
            _, free_eng = self.predictive_coder(latent)
            self.free_energy_history.append(float(free_eng))

            # Emotion estimate
            emo = self.emotion_reg(latent)

            # Memory-augmented latent
            mem_t = self._memory_tensors()
            if mem_t is not None:
                ctx, _ = self.memory_attn(latent, mem_t)
                seed   = (latent + 0.3 * ctx).squeeze(0).cpu().numpy()
            else:
                seed   = latent.squeeze(0).cpu().numpy()

        # Biological thought formation
        thought = None
        if self.brain.consciousness:
            thought = self.brain.consciousness.form_thought(seed, context)

        return {
            'thought':       thought,
            'ltp_threshold': float(ltp[0]),
            'ltd_threshold': float(ltd[0]),
            'free_energy':   float(free_eng),
            'valence':       float(emo['valence'][0]),
            'arousal':       float(emo['arousal'][0]),
            'top_emotion':   EmotionRegressor.LABELS[
                                 int(emo['emotion_probs'][0].argmax())],
        }

    def learn_step(self, inputs: np.ndarray) -> Dict[str, float]:
        """
        Fast per-step learning:
          PyTorch  — predictive-coding loss + emotion-arousal consistency (backprop)
          Hebbian  — form_thought() on the encoded latent (cheap, no evolution)
        For full evolutionary biological learning, call bio_train() between episodes.
        """
        self.optimizer.zero_grad()

        x      = self._pad(self._t(inputs), self.config.input_size)
        latent = self.encoder(x)

        # Predictive coding loss
        pred, free_eng = self.predictive_coder(latent)
        pred_loss = F.mse_loss(pred, latent.detach())

        # Emotion consistency: high free energy → high arousal
        emo           = self.emotion_reg(latent)
        target_arousal = torch.sigmoid(free_eng.detach() * 5).expand(x.size(0))
        emo_loss      = F.mse_loss(emo['arousal'], target_arousal)

        loss = pred_loss + 0.1 * emo_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Lightweight Hebbian: seed consciousness with encoded latent
        seed_np = latent.squeeze(0).detach().cpu().numpy()
        if self.brain.consciousness:
            self.brain.consciousness.form_thought(seed_np, "learn")

        # Store in hippocampus so dream consolidation has content.
        # Truncate latent to brain input_size so shapes match during replay.
        if self.brain.hippocampus:
            bsz = self.config.brain_config.input_size
            mem_vec = seed_np[:bsz] if len(seed_np) >= bsz else np.pad(seed_np, (0, bsz - len(seed_np)))
            self.brain.memorize(
                mem_vec,
                label=np.array([float(free_eng)]),
                importance=float(np.clip(float(free_eng) * 3.0, 0.1, 3.0)),
            )

        self.step_count += 1
        self.loss_history.append(float(loss))

        return {
            'total_loss':    float(loss),
            'pred_loss':     float(pred_loss),
            'emo_loss':      float(emo_loss),
            'free_energy':   float(free_eng),
            'ltp_threshold': self.brain.consciousness.ltp_threshold
                             if self.brain.consciousness else 0.7,
        }

    def bio_train(self, X: np.ndarray, y: np.ndarray):
        """
        Full evolutionary biological training (brain.learn).
        Heavy — runs neuroevolution + curriculum.  Call once per episode,
        not inside a per-step loop.
        """
        bc = self.config.brain_config
        bx = np.stack([self._pad_np(r, bc.input_size)  for r in X])
        by = np.stack([self._pad_np(r, bc.output_size) for r in y])
        self.brain.learn(bx, by, verbose=False)

    @staticmethod
    def _pad_np(arr: np.ndarray, size: int) -> np.ndarray:
        if len(arr) >= size:
            return arr[:size]
        return np.pad(arr, (0, size - len(arr)))

    def dream(self, n_cycles: int = 3):
        """Biological dream replay + reset predictive coder context."""
        self.brain.dream(n_cycles=n_cycles, verbose=False)
        self.predictive_coder.reset()

    def get_state(self) -> Dict:
        snap = self.brain.get_consciousness_snapshot() if self.brain.consciousness else {}
        state = {
            **snap,
            'step_count':      self.step_count,
            'torch_loss':      self.loss_history[-1]          if self.loss_history else 0.0,
            'avg_loss_100':    float(np.mean(self.loss_history[-100:]))
                               if self.loss_history else 0.0,
            'avg_free_energy': float(np.mean(self.free_energy_history[-100:]))
                               if self.free_energy_history else 0.0,
            'torch_params':    sum(p.numel() for p in self.parameters()),
            'device':          str(self.device),
        }
        # Quantum preciousness
        if self.brain.quantum is not None:
            state['quantum_preciousness'] = self.brain.quantum_preciousness()
        # Meta-cognition
        meta = self.brain.metacognitive_update()
        state['meta_lr_scale'] = self.brain._meta_lr_scale
        state['meta_regime']   = meta.get('regime', meta.get('meta_cognition', 'n/a'))
        return state

    def save(self, path: str):
        """Save torch weights + biological brain state together."""
        torch.save({
            'model':          self.state_dict(),
            'optimizer':      self.optimizer.state_dict(),
            'loss_history':   self.loss_history,
            'fe_history':     self.free_energy_history,
            'step_count':     self.step_count,
        }, path)
        print(f"  Torch state → {path}")
        brain_path = os.path.splitext(path)[0] + '_bio.pkl'
        self.brain.save(brain_path)

    def load(self, path: str):
        """Load torch weights + biological brain state."""
        ck = torch.load(path, map_location=self.device)
        self.load_state_dict(ck['model'])
        self.optimizer.load_state_dict(ck['optimizer'])
        self.loss_history         = ck['loss_history']
        self.free_energy_history  = ck['fe_history']
        self.step_count           = ck['step_count']
        print(f"  Torch state ← {path}")
        brain_path = os.path.splitext(path)[0] + '_bio.pkl'
        if os.path.exists(brain_path):
            from A_Brain import PreciousBrain
            self.brain = PreciousBrain.load(brain_path)

    # ── continual learning / forgetting pass-throughs ─────────────────────────

    def consolidate_knowledge(self, X: Optional[np.ndarray] = None,
                               y: Optional[np.ndarray] = None,
                               n_anchors: int = 64) -> int:
        """Anchor current knowledge for EWC-style catastrophic forgetting prevention."""
        return self.brain.consolidate_knowledge(X, y, n_anchors)

    def forget(self, query: np.ndarray, threshold: float = 0.80) -> Dict:
        """Remove memories similar to query from all memory stores."""
        return self.brain.forget(query, threshold)

    def summary(self):
        s = self.get_state()
        print("\n" + "=" * 62)
        print("🔥  TORCH PRECIOUS BRAIN")
        print("=" * 62)
        print(f"  Device          : {s['device']}")
        print(f"  Torch params    : {s['torch_params']:,}")
        print(f"  Steps trained   : {s['step_count']}")
        print(f"  Avg loss (100)  : {s['avg_loss_100']:.6f}")
        print(f"  Avg free-energy : {s['avg_free_energy']:.6f}")
        if self.brain.consciousness:
            print(f"\n  Consciousness")
            print(f"    Thoughts      : {s.get('total_thoughts', 0)}")
            print(f"    Memories      : {s.get('memories_stored', 0)}")
            print(f"    LTP threshold : {s.get('ltp_threshold', 0.7):.3f}  ← dynamic")
            print(f"    LTD threshold : {s.get('ltd_threshold', 0.3):.3f}  ← dynamic")
            print(f"    Emotion       : {s.get('current_emotion', 'n/a')}")
        qp = s.get('quantum_preciousness', {})
        if qp:
            print(f"\n  Quantum Cognition")
            print(f"    Preciousness  : {qp.get('total', 0):.1f} / {qp.get('max', 100):.0f}")
            print(f"    Coherence     : {qp.get('coherence', 0):.2f}")
            print(f"    Entanglement  : {qp.get('entanglement', 0):.2f}")
            print(f"    Decohere res. : {qp.get('decohere_resist', 0):.2f}")
        print(f"\n  Meta-Cognition")
        print(f"    LR scale      : {s.get('meta_lr_scale', 1.0):.3f}")
        print(f"    Regime        : {s.get('meta_regime', 'n/a')}")
        print(f"    EWC anchors   : {len(self.brain._ewc_anchors)}")
        print("=" * 62)


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────
def _banner(title: str):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def demo_think(brain: TorchPreciousBrain):
    _banner("DEMO 1 — THINK")
    raw = np.random.randn(brain.config.input_size)
    result = brain.think(raw, context="demo_perception")
    print(f"  Free energy  : {result['free_energy']:.4f}")
    print(f"  LTP threshold: {result['ltp_threshold']:.3f}  (was 0.700 static)")
    print(f"  LTD threshold: {result['ltd_threshold']:.3f}  (was 0.300 static)")
    print(f"  Valence      : {result['valence']:+.3f}")
    print(f"  Arousal      : {result['arousal']:.3f}")
    print(f"  Top emotion  : {result['top_emotion']}")
    if result['thought']:
        t = result['thought']
        print(f"  Active neurons: {len(t.active_neurons)}")


def demo_learn(brain: TorchPreciousBrain, steps: int = 100):
    _banner(f"DEMO 2 — LEARNING ({steps} steps)")
    t0     = time.time()
    losses = []
    for i in range(steps):
        inp = np.random.randn(brain.config.input_size)
        m = brain.learn_step(inp)
        losses.append(m['total_loss'])
        if (i + 1) % 25 == 0:
            print(f"  step {i+1:4d}  loss={m['total_loss']:.5f}"
                  f"  FE={m['free_energy']:.4f}"
                  f"  LTP>{m['ltp_threshold']:.3f}")
    elapsed = time.time() - t0
    print(f"\n  {steps} steps in {elapsed:.2f}s  ({steps/elapsed:.1f} steps/sec)")
    print(f"  Loss drop   : {losses[0]:.5f} → {losses[-1]:.5f}")


def demo_dream(brain: TorchPreciousBrain):
    _banner("DEMO 3 — DREAM STATE")
    brain.dream(n_cycles=3)
    print("  Predictive coder context reset — fresh context for next episode.")


def demo_benchmark(brain: TorchPreciousBrain, n: int = 200):
    _banner(f"BENCHMARK — think() × {n}")
    inp = np.random.randn(brain.config.input_size)
    # warmup
    for _ in range(5):
        brain.think(inp)
    t0 = time.time()
    for _ in range(n):
        brain.think(np.random.randn(brain.config.input_size))
    elapsed = time.time() - t0
    print(f"  {n} thoughts in {elapsed:.3f}s  ({n/elapsed:.1f} thoughts/sec)")


def run_demo():
    print("=" * 62)
    print("  TORCH PRECIOUS BRAIN — INITIALISING")
    print("=" * 62)

    torch_cfg = TorchBrainConfig(
        brain_config=PreciousBrainConfig(
            input_size=64,
            output_size=1,
            n_consciousness_neurons=5000,
        ),
        input_size=64,
        latent_size=128,
    )

    brain = TorchPreciousBrain(torch_cfg)
    print(f"  Torch params : {sum(p.numel() for p in brain.parameters()):,}")
    print(f"  Device       : {brain.device}")
    print(f"  Neurons      : {torch_cfg.brain_config.n_consciousness_neurons:,}")

    demo_think(brain)
    demo_learn(brain, steps=100)
    demo_dream(brain)
    demo_benchmark(brain, n=200)
    brain.summary()


if __name__ == "__main__":
    run_demo()
