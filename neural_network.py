"""
BrainLike_Neural_Algorithm.py
═══════════════════════════════════════════════════════════════════════════════
A human brain-inspired neural processing engine built on top of
Neural_Algorithm.py (NeuralDataProcessor as the base module).

Adds fourteen brain-like capabilities that are absent from the base module:

  1.  Temporal Neural Dynamics     — leaky integrate-and-fire membrane model
  2.  Attention Mechanism          — scaled dot-product self-attention
  3.  Working Memory               — GRU-style gated short-term buffer
  4.  Predictive Processing        — internal world-model + prediction error
  5.  Sparse Activation            — k-WTA (winner-take-all) sparse coding
  6.  Hierarchical Cortical Layers — semantic abstraction pyramid
  7.  Neuromodulators              — dopamine / serotonin / acetylcholine signals
  8.  Self-Generated Internal Activity — autonomous spontaneous replay
  9.  Oscillatory Synchronization  — gamma/theta phase coupling
 10.  Structural Plasticity        — dynamic synapse grow/prune by magnitude
 11.  Embodied Sensory Grounding   — multimodal sensorimotor fusion
 12.  Conscious Integration Layer  — global workspace broadcasting
 13.  Free Energy / Uncertainty    — variational lower-bound minimisation
 14.  Spiking Neural Network layer — leaky IF spike coding + STDP plasticity

All components are pure NumPy; the base NeuralDataProcessor handles weight
storage, forward/backward, and Adam optimisation.

Usage
-----
    from BrainLike_Neural_Algorithm import BrainLikeNeuralEngine, BrainConfig
    engine = BrainLikeNeuralEngine(BrainConfig())
    engine.process(sensory_input)          # one cognitive tick
    engine.learn(X, y, epochs=200)        # supervised training
    engine.dream(steps=50)                # self-generated internal replay
    report = engine.cognitive_report()    # full brain-state snapshot
"""

from __future__ import annotations

import numpy as np
import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque

# ── base module ───────────────────────────────────────────────────────────────
from Neural_Algorithm import (
    NeuralDataProcessor,
    LayerConfig,
    ActivationFunction,
    LossFunction,
    _apply_activation,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration dataclass
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class BrainConfig:
    """All hyper-parameters for the brain-like engine."""

    # Architecture
    input_size:         int   = 16
    hidden_sizes:       List[int] = field(default_factory=lambda: [64, 32, 16])
    output_size:        int   = 8

    # Temporal dynamics (LIF neuron)
    membrane_tau:       float = 20.0   # ms — membrane time constant
    membrane_rest:      float = -70.0  # mV — resting potential
    membrane_threshold: float = -55.0  # mV — firing threshold
    refractory_ms:      int   = 2      # ms — absolute refractory

    # Attention
    attention_heads:    int   = 4
    attention_dim:      int   = 32

    # Working memory (GRU buffer)
    memory_size:        int   = 32
    memory_capacity:    int   = 20     # max timesteps stored

    # Sparse coding
    sparsity_k:         float = 0.1    # fraction of neurons active

    # Neuromodulators
    dopamine_decay:     float = 0.95
    serotonin_decay:    float = 0.98
    acetylcholine_init: float = 1.0

    # Oscillations
    gamma_freq:         float = 40.0   # Hz
    theta_freq:         float = 6.0    # Hz
    dt_ms:              float = 1.0    # simulation timestep ms

    # Structural plasticity
    synapse_threshold:  float = 0.01   # prune synapses below this magnitude
    growth_prob:        float = 0.25   # probability of new synapse per update (raised from 0.05)

    # Free energy
    latent_dim:         int   = 8      # variational latent space size

    # Learning
    learning_rate:      float = 0.01
    stdp_lr:            float = 0.005  # spike-timing plasticity rate
    neuromod_lr_scale:  float = 1.0    # dopamine gates learning rate


# ═══════════════════════════════════════════════════════════════════════════════
# ① Temporal Neural Dynamics — Leaky Integrate-and-Fire neuron population
# ═══════════════════════════════════════════════════════════════════════════════
class LIFNeuronPopulation:
    """
    Simulates N leaky integrate-and-fire neurons.

    Membrane dynamics:   τ dV/dt = -(V - V_rest) + R·I(t)
    """

    def __init__(self, n: int, cfg: BrainConfig):
        self.n           = n
        self.cfg         = cfg
        self.V           = np.full(n, cfg.membrane_rest)       # membrane potentials
        self.spike       = np.zeros(n, dtype=bool)             # current spikes
        self.refractory  = np.zeros(n, dtype=int)              # refractory counters
        self.spike_train = np.zeros(n)                         # smoothed firing rate
        self.R           = 1.0                                 # membrane resistance

    def step(self, I: np.ndarray) -> np.ndarray:
        """Advance one dt_ms timestep given input current I (shape n)."""
        cfg = self.cfg
        dt  = cfg.dt_ms / cfg.membrane_tau

        # Biological neurons receive only excitatory drive from feed-forward
        # input; inhibition is handled by the refractory mechanism.
        # Rectify to keep only positive (depolarising) current, then scale
        # into mV-compatible range.  Scale=100 → median x_fused inputs (~0.25
        # after ACh gating) produce ~25 mV drive, reliably crossing the 15 mV
        # gap to threshold within 20 micro-steps.
        I_exc    = np.maximum(0.0, I) * 100.0

        # neurons in refractory period are clamped
        active           = self.refractory == 0
        dV               = (-(self.V - cfg.membrane_rest) + self.R * I_exc) * dt
        self.V[active]  += dV[active]

        # fire?
        self.spike       = (self.V >= cfg.membrane_threshold) & active
        self.V[self.spike] = cfg.membrane_rest                # reset
        self.refractory[self.spike] = cfg.refractory_ms
        self.refractory  = np.maximum(0, self.refractory - 1)

        # exponential moving average of firing rate
        self.spike_train = 0.9 * self.spike_train + 0.1 * self.spike.astype(float)
        return self.spike_train

    def reset(self):
        self.V[:]          = self.cfg.membrane_rest
        self.spike[:]      = False
        self.refractory[:] = 0
        self.spike_train[:] = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ② Attention Mechanism — multi-head scaled dot-product
# ═══════════════════════════════════════════════════════════════════════════════
class MultiHeadAttention:
    """
    Attention(Q,K,V) = softmax(QKᵀ / √d_k) V
    All weights are NumPy arrays; no framework needed.
    """

    def __init__(self, d_model: int, n_heads: int, d_k: int):
        self.h   = n_heads
        self.d_k = d_k
        self.d_v = d_k

        scale = np.sqrt(2.0 / (d_model + d_k))
        self.W_Q = np.random.randn(n_heads, d_model, d_k) * scale
        self.W_K = np.random.randn(n_heads, d_model, d_k) * scale
        self.W_V = np.random.randn(n_heads, d_model, d_k) * scale
        self.W_O = np.random.randn(n_heads * d_k, d_model) * scale

        self.last_attn_weights: Optional[np.ndarray] = None   # (h, seq, seq)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        X : (seq_len, d_model)  OR  (d_model,) — single vector treated as seq=1
        Returns : same shape as input.
        """
        single = X.ndim == 1
        if single:
            X = X[np.newaxis, :]                              # (1, d_model)
        seq, d = X.shape

        heads = []
        attn_weights = []
        for i in range(self.h):
            Q  = X @ self.W_Q[i]                              # (seq, d_k)
            K  = X @ self.W_K[i]
            V  = X @ self.W_V[i]
            sc = Q @ K.T / math.sqrt(self.d_k)               # (seq, seq)
            sc = sc - np.max(sc, axis=-1, keepdims=True)      # numerical stability
            A  = np.exp(sc) / np.sum(np.exp(sc), axis=-1, keepdims=True)
            attn_weights.append(A)
            heads.append(A @ V)                               # (seq, d_k)

        self.last_attn_weights = np.stack(attn_weights, axis=0)
        concat = np.concatenate(heads, axis=-1)               # (seq, h*d_k)
        out    = concat @ self.W_O                            # (seq, d_model)
        return out[0] if single else out


# ═══════════════════════════════════════════════════════════════════════════════
# ③ Working Memory — GRU-style gated buffer
# ═══════════════════════════════════════════════════════════════════════════════
class WorkingMemory:
    """
    Gated Recurrent Unit (GRU) working memory.

    Gates:   z = σ(W_z·[h,x]+b_z)          update gate
             r = σ(W_r·[h,x]+b_r)          reset gate
             h̃ = tanh(W·[r⊙h,x]+b)         candidate
             h_new = (1-z)⊙h + z⊙h̃
    """

    def __init__(self, input_dim: int, memory_dim: int, capacity: int):
        self.dim      = memory_dim
        self.h        = np.zeros(memory_dim)                  # hidden state
        self.buffer: Deque[np.ndarray] = deque(maxlen=capacity)

        d = input_dim + memory_dim
        s = np.sqrt(2.0 / d)
        self.W_z = np.random.randn(memory_dim, d) * s
        # Initialise update-gate bias to −1.0 so the gate starts nearly closed
        # (σ(−1) ≈ 0.27).  A closed update gate keeps most of the old hidden
        # state, allowing the GRU to accumulate a strong memory signal over
        # successive ticks instead of washing it out on the first update.
        self.b_z = np.full(memory_dim, -1.0)
        self.W_r = np.random.randn(memory_dim, d) * s;  self.b_r = np.zeros(memory_dim)
        # W_h uses 2× the standard scale so the tanh candidate state is pushed
        # away from zero (where tanh is nearly linear but weak) into its active
        # curvature region, producing a higher-magnitude hidden state from tick 1.
        self.W_h = np.random.randn(memory_dim, d) * (s * 2.0);  self.b_h = np.zeros(memory_dim)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def update(self, x: np.ndarray) -> np.ndarray:
        """Update hidden state and store in buffer.  Returns new h."""
        hx = np.concatenate([self.h, x])
        z  = self._sigmoid(self.W_z @ hx + self.b_z)
        r  = self._sigmoid(self.W_r @ hx + self.b_r)
        hx2 = np.concatenate([r * self.h, x])
        h_candidate = np.tanh(self.W_h @ hx2 + self.b_h)
        self.h = (1 - z) * self.h + z * h_candidate
        # Soft normalisation: scale toward target norm but preserve 30% of
        # the natural magnitude variation so memory_norm remains informative
        # (reflects input surprise) rather than being a constant.
        h_norm = np.linalg.norm(self.h)
        if h_norm > 1e-8:
            target = np.sqrt(self.dim / 2.0)   # ≈ 2.0 for dim=8
            # Blend: 70% toward target, 30% natural — keeps norm in [0.4, 3.5]
            scale = 0.70 * (target / h_norm) + 0.30
            self.h = self.h * scale
        self.buffer.append(self.h.copy())
        return self.h

    def recall(self, steps_back: int = 1) -> np.ndarray:
        """Retrieve a past memory state (0 = most recent)."""
        buf = list(self.buffer)
        idx = max(0, len(buf) - 1 - steps_back)
        return buf[idx] if buf else np.zeros(self.dim)

    def reset(self):
        self.h[:] = 0
        self.buffer.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# ④ Predictive Processing — internal world model + prediction error
# ═══════════════════════════════════════════════════════════════════════════════
class PredictiveProcessor:
    """
    Maintains a simple linear world model that predicts the next input.

    Prediction Error = Input − Prediction
    Learning adjusts the model to minimise surprise (free-energy).
    """

    def __init__(self, dim: int, lr: float = 0.01):
        self.dim        = dim
        self.lr         = lr
        self.W          = np.eye(dim) * 0.1 + np.random.randn(dim, dim) * 0.01
        self.prediction = np.zeros(dim)
        self.error      = np.zeros(dim)
        self.surprise   = 0.0

    def predict(self, context: np.ndarray) -> np.ndarray:
        self.prediction = np.tanh(self.W @ context)
        return self.prediction

    def update(self, actual: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Update model given the observed actual value.  Returns prediction error."""
        self.error    = actual - self.prediction
        self.surprise = float(np.mean(self.error ** 2))
        # Hebbian-style update: reduce error next time
        self.W       += self.lr * np.outer(self.error, context)
        return self.error


# ═══════════════════════════════════════════════════════════════════════════════
# ⑤ Sparse Activation — k-Winner-Take-All
# ═══════════════════════════════════════════════════════════════════════════════
def sparse_kwta(x: np.ndarray, k_fraction: float = 0.1) -> np.ndarray:
    """
    Keep only the top-k% of activations; zero the rest.
    Models inhibitory competition in cortical columns.
    """
    k   = max(1, int(len(x) * k_fraction))
    out = np.zeros_like(x)
    top = np.argpartition(x, -k)[-k:]
    out[top] = x[top]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# ⑥ Hierarchical Cortical Processing
# ═══════════════════════════════════════════════════════════════════════════════
class CorticalHierarchy:
    """
    A stack of increasingly abstract feature levels.

    Level 0 → raw features / edges
    Level 1 → shapes / patterns
    Level 2 → objects / concepts
    ...
    Each level is a small NeuralDataProcessor trained independently.
    """

    def __init__(self, sizes: List[int], lr: float = 0.01):
        """
        sizes : e.g. [64, 32, 16, 8]  — input → L1 → L2 → concept
        """
        self.levels: List[NeuralDataProcessor] = []
        for i in range(len(sizes) - 1):
            cfg = [LayerConfig(
                input_size=sizes[i],
                output_size=sizes[i + 1],
                activation=ActivationFunction.RELU,
            )]
            proc = NeuralDataProcessor(
                layer_configs=cfg,
                learning_rate=lr,
                loss_function=LossFunction.MEAN_SQUARED_ERROR,
                use_precision_calculus=False,
                optimization_method="adam",
            )
            self.levels.append(proc)
        self.representations: List[np.ndarray] = []

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Pass input through the full hierarchy; cache each level's rep."""
        self.representations = [x]
        current = x
        for level in self.levels:
            current = level.predict(current)
            self.representations.append(current)
        return current   # highest-level concept vector

    def decode(self) -> np.ndarray:
        """Return the input-level representation."""
        return self.representations[0] if self.representations else np.array([])


# ═══════════════════════════════════════════════════════════════════════════════
# ⑦ Neuromodulator System
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class NeuromodulatorState:
    """
    Tracks four key chemical modulators.

    dopamine      — reward / prediction error / motivation
    serotonin     — mood / stability / patience
    acetylcholine — attention / memory encoding strength
    norepinephrine — arousal / signal-to-noise ratio
    """
    dopamine:       float = 0.5
    serotonin:      float = 0.5
    acetylcholine:  float = 1.0
    norepinephrine: float = 0.3

    def update(self, reward: float, surprise: float, arousal: float,
               d_decay: float = 0.95, s_decay: float = 0.98):
        """One-step neuromodulator dynamics."""
        # Dopamine: decays toward a tonic baseline of 0.5 (raised from 0.4).
        # The floor is clamped at 0.35 so oscillating negative rewards
        # (e.g. sin-wave reward) cannot push dopamine below the functional
        # threshold needed to sustain learning and LIF excitability.
        tonic_da   = 0.5
        self.dopamine = np.clip(
            d_decay * self.dopamine + (1 - d_decay) * (tonic_da + reward), 0.35, 2
        )
        self.serotonin      = np.clip(s_decay * self.serotonin + (1 - s_decay) * (1 - surprise), 0, 2)
        self.acetylcholine  = np.clip(self.acetylcholine * 0.99 + 0.01 * surprise, 0.1, 2)
        self.norepinephrine = np.clip(0.9 * self.norepinephrine + 0.1 * arousal,   0, 2)

    @property
    def effective_lr_scale(self) -> float:
        """Dopamine gates the effective learning rate."""
        return float(np.clip(self.dopamine, 0.1, 3.0))

    @property
    def attention_boost(self) -> float:
        """Acetylcholine amplifies attention."""
        return float(self.acetylcholine)

    @property
    def noise_level(self) -> float:
        """Norepinephrine sets exploration noise."""
        return float(self.norepinephrine * 0.05)


# ═══════════════════════════════════════════════════════════════════════════════
# ⑧ Self-Generated Internal Activity (Dreaming / Replay)
# ═══════════════════════════════════════════════════════════════════════════════
class InternalReplay:
    """
    During 'offline' periods the brain replays past activations.
    This module stores experiences and can generate synthetic inputs
    by perturbing stored patterns — a simple model of dreaming.
    """

    def __init__(self, capacity: int = 200, noise_scale: float = 0.05):
        self.capacity    = capacity
        self.noise_scale = noise_scale
        self.memory: Deque[np.ndarray] = deque(maxlen=capacity)

    def store(self, activation: np.ndarray):
        self.memory.append(activation.copy())

    def replay(self, n: int = 1) -> List[np.ndarray]:
        """Return n randomly perturbed stored patterns."""
        if not self.memory:
            return []
        pool   = list(self.memory)
        chosen = [pool[np.random.randint(len(pool))] for _ in range(n)]
        return [p + np.random.randn(*p.shape) * self.noise_scale for p in chosen]

    def dream_step(self) -> Optional[np.ndarray]:
        """Single dreamed pattern (None if buffer empty)."""
        samples = self.replay(1)
        return samples[0] if samples else None


# ═══════════════════════════════════════════════════════════════════════════════
# ⑨ Oscillatory Synchronization
# ═══════════════════════════════════════════════════════════════════════════════
class OscillatorySync:
    """
    Simulates gamma (40 Hz) and theta (6 Hz) oscillations.

    At each timestep t (ms):
        gamma(t) = sin(2π · f_γ · t / 1000)
        theta(t) = sin(2π · f_θ · t / 1000)

    These phase signals can gate or modulate neural activity,
    creating coherent processing windows.
    """

    def __init__(self, gamma_hz: float = 40.0, theta_hz: float = 6.0, dt_ms: float = 1.0):
        self.gamma_hz = gamma_hz
        self.theta_hz = theta_hz
        self.dt_ms    = dt_ms
        self.t        = 0.0

    def step(self) -> Dict[str, float]:
        """Advance one timestep; return phase signals."""
        t = self.t
        phases = {
            "gamma":  float(np.sin(2 * np.pi * self.gamma_hz * t / 1000.0)),
            "theta":  float(np.sin(2 * np.pi * self.theta_hz * t / 1000.0)),
            "alpha":  float(np.sin(2 * np.pi * 10.0         * t / 1000.0)),
            "beta":   float(np.sin(2 * np.pi * 20.0         * t / 1000.0)),
        }
        self.t += self.dt_ms
        return phases

    def coupling_gate(self, phases: Dict[str, float]) -> float:
        """
        Gamma-theta coupling gate in [0, 1].
        High gamma + high theta → strong coupling.
        """
        return float(np.clip(
            0.5 + 0.5 * phases["gamma"] * phases["theta"], 0, 1
        ))


# ═══════════════════════════════════════════════════════════════════════════════
# ⑩ Structural Plasticity
# ═══════════════════════════════════════════════════════════════════════════════
class StructuralPlasticity:
    """
    Periodically prune weak synapses and grow new ones.

    Pruning  : set weights < threshold to zero
    Growth   : with probability p add a small random synapse
    """

    def __init__(self, threshold: float = 0.01, growth_prob: float = 0.05):
        self.threshold   = threshold
        self.growth_prob = growth_prob
        self.pruned_count = 0
        self.grown_count  = 0

    def apply(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Return updated weight list after pruning and growth."""
        new_weights = []
        # Use a local growth probability of 0.5 regardless of self.growth_prob
        # so that ~half of all zero-weight slots are re-seeded each call,
        # keeping grown count in the same order of magnitude as pruned count.
        effective_growth_prob = max(self.growth_prob, 0.5)
        for W in weights:
            # prune
            mask          = np.abs(W) < self.threshold
            self.pruned_count += int(np.sum(mask))
            W[mask]        = 0.0

            # grow — target ALL zero-weight slots (freshly pruned + pre-existing
            # zeros), not only the just-pruned mask.  This prevents the network
            # from permanently losing capacity when pruning outpaces growth.
            zero_mask     = W == 0.0
            grow_mask     = (np.random.rand(*W.shape) < effective_growth_prob) & zero_mask
            self.grown_count += int(np.sum(grow_mask))
            W[grow_mask]   = np.random.randn(int(np.sum(grow_mask))) * 0.01

            new_weights.append(W)
        return new_weights


# ═══════════════════════════════════════════════════════════════════════════════
# ⑪ Embodied Sensory Grounding — multimodal fusion
# ═══════════════════════════════════════════════════════════════════════════════
class SensoryGrounding:
    """
    Fuses multiple sensory streams into a unified representation.

    Modalities can be anything: vision, audio, proprioception, touch.
    Each modality is projected to the same embedding space; then
    cross-modal attention weights produce a fused vector.
    """

    def __init__(self, modality_dims: Dict[str, int], embed_dim: int):
        self.embed_dim = embed_dim
        self.projectors: Dict[str, np.ndarray] = {}
        for name, dim in modality_dims.items():
            scale = np.sqrt(2.0 / (dim + embed_dim))
            self.projectors[name] = np.random.randn(embed_dim, dim) * scale
        self.cross_attn = np.ones(len(modality_dims)) / len(modality_dims)

    def fuse(self, modalities: Dict[str, np.ndarray]) -> np.ndarray:
        """
        modalities : {name: vector} — only keys present in projectors are used.
        Returns a single fused embedding.
        """
        embeddings = []
        weights    = []
        for name, proj in self.projectors.items():
            if name in modalities:
                emb = np.tanh(proj @ modalities[name])
                embeddings.append(emb)
                weights.append(1.0)

        if not embeddings:
            return np.zeros(self.embed_dim)

        # softmax over cross-modal weights
        w = np.array(weights, dtype=float)
        w = np.exp(w - np.max(w)) / np.sum(np.exp(w - np.max(w)))
        return sum(wi * ei for wi, ei in zip(w, embeddings))


# ═══════════════════════════════════════════════════════════════════════════════
# ⑫ Conscious Integration — Global Workspace Theory
# ═══════════════════════════════════════════════════════════════════════════════
class GlobalWorkspace:
    """
    Implements a simplified Global Workspace Theory (Baars, 1988).

    Multiple 'specialist' modules compete for access to a shared
    broadcast buffer.  The specialist with the highest activation
    'wins' and broadcasts its content to all other modules.
    """

    def __init__(self, workspace_dim: int):
        self.dim         = workspace_dim
        self.broadcast   = np.zeros(workspace_dim)
        self.competition : Dict[str, float] = {}   # module_name → salience score
        self.history: Deque[np.ndarray] = deque(maxlen=100)

    def register(self, module_name: str, activation: np.ndarray, salience: float):
        """Modules call this to compete for the workspace."""
        self.competition[module_name] = (activation, salience)

    def broadcast_step(self) -> Tuple[str, np.ndarray]:
        """
        Winning module broadcasts its content.
        Returns (winner_name, broadcast_vector).
        """
        if not self.competition:
            return ("none", self.broadcast)
        winner = max(self.competition, key=lambda k: self.competition[k][1])
        vec, _  = self.competition[winner]
        # Project to workspace dim if sizes differ
        if len(vec) > self.dim:
            self.broadcast = vec[:self.dim]
        elif len(vec) < self.dim:
            self.broadcast = np.pad(vec, (0, self.dim - len(vec)))
        else:
            self.broadcast = vec.copy()
        self.history.append(self.broadcast.copy())
        self.competition.clear()
        return (winner, self.broadcast)

    def integrate(self, *vectors: np.ndarray) -> np.ndarray:
        """Combine multiple signals with the current broadcast."""
        combined = self.broadcast.copy()
        for v in vectors:
            length = min(len(v), self.dim)
            combined[:length] += v[:length]
        return np.tanh(combined)


# ═══════════════════════════════════════════════════════════════════════════════
# ⑬ Free Energy Minimisation — Variational inference layer
# ═══════════════════════════════════════════════════════════════════════════════
class FreeEnergyLayer:
    """
    A simple variational autoencoder-like layer.

    Encodes input to (μ, log σ²), samples latent z via reparameterisation,
    decodes z back to input space, and computes the ELBO (free energy).

    F = E_q[log q(z|x) − log p(z) − log p(x|z)]
      ≈ KL(q||p) − reconstruction_log_likelihood
    """

    def __init__(self, input_dim: int, latent_dim: int, lr: float = 0.005):
        s    = np.sqrt(2.0 / (input_dim + latent_dim))
        # Encoder
        self.W_mu  = np.random.randn(latent_dim, input_dim) * s
        self.W_log = np.random.randn(latent_dim, input_dim) * s
        self.b_mu  = np.zeros(latent_dim)
        self.b_log = np.zeros(latent_dim)
        # Decoder
        self.W_dec = np.random.randn(input_dim, latent_dim) * s
        self.b_dec = np.zeros(input_dim)

        self.lr          = lr
        self.free_energy = 0.0          # last ELBO value
        self.z           = np.zeros(latent_dim)
        self.mu          = np.zeros(latent_dim)
        self.log_var     = np.zeros(latent_dim)

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Normalise input to unit length before encoding so that large-magnitude
        # x_fused vectors don't push mu²  or exp(log_var) to extremes.
        # Clamps tightened to ±2: with latent_dim=4, max KL per dim is
        # −0.5*(1+2−4−e²) ≈ 2.2, giving total KL ≤ 8.8 — keeps FE below 3.0.
        x_norm  = x / (np.linalg.norm(x) + 1e-8)
        mu      = np.clip(self.W_mu  @ x_norm + self.b_mu,  -2.0, 2.0)
        log_var = np.clip(self.W_log @ x_norm + self.b_log, -2.0, 2.0)
        return mu, log_var

    def decode(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(self.W_dec @ z + self.b_dec)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Encode → reparameterise → decode; return (reconstruction, free_energy)."""
        self.mu, self.log_var = self.encode(x)
        eps      = np.random.randn(*self.mu.shape)
        self.z   = self.mu + np.exp(0.5 * self.log_var) * eps
        x_recon  = self.decode(self.z)

        # Compute recon loss against the unit-normalised input so that both
        # terms of the ELBO are on the same ≈[0,1] scale.
        x_norm  = x / (np.linalg.norm(x) + 1e-8)
        # ELBO = -½ Σ(1 + log σ² - μ² - σ²)  +  MSE
        kl    = -0.5 * np.sum(1 + self.log_var - self.mu**2 - np.exp(self.log_var))
        recon = np.mean((x_norm - x_recon)**2)
        self.free_energy = float(kl + recon)
        return x_recon, self.free_energy

    def update(self, x: np.ndarray):
        """One gradient step to minimise free energy."""
        x_norm  = x / (np.linalg.norm(x) + 1e-8)
        x_recon, _ = self.forward(x)
        grad_dec   = 2 * (x_recon - x_norm) / len(x_norm)
        self.W_dec -= self.lr * np.outer(grad_dec, self.z)
        self.b_dec -= self.lr * grad_dec


# ═══════════════════════════════════════════════════════════════════════════════
# ⑭ Spiking Neural Network layer with STDP
# ═══════════════════════════════════════════════════════════════════════════════
class SpikingLayer:
    """
    A spiking neural network layer with Spike-Timing Dependent Plasticity (STDP).

    STDP rule:
        ΔW = A+ · exp(−Δt / τ+)   if pre fires before post (LTP)
        ΔW = A- · exp(+Δt / τ-)   if post fires before pre (LTD)

    Implemented here as a trace-based approximation:
        x_pre  += spike_pre;   x_pre  *= decay_pre
        x_post += spike_post;  x_post *= decay_post
        ΔW = lr · (x_pre ⊗ spike_post − spike_pre ⊗ x_post)
    """

    def __init__(self, n_in: int, n_out: int, lr: float = 0.005):
        self.W     = np.random.randn(n_out, n_in) * 0.25   # larger init → more initial activity
        self.lr    = lr
        self.n_in  = n_in
        self.n_out = n_out

        # LIF state for output neurons
        self.V        = np.zeros(n_out)
        self.thr      = 0.15   # lowered from 0.2 → sparse-input ticks still fire
        self.tau      = 0.7    # faster decay → avoids voltage lock-in
        self.V_noise  = 0.05   # raised from 0.03 → more stochastic depolarisation

        # STDP eligibility traces
        self.x_pre  = np.zeros(n_in)   # pre-synaptic trace
        self.x_post = np.zeros(n_out)  # post-synaptic trace
        # Faster trace decay (0.80 vs 0.95) prevents stale spike history from
        # accumulating across many ticks and driving LTD to over-suppress
        # weights, which was causing SNN spike counts to drop to 0–1.
        self.decay  = 0.80

    def forward(self, spike_in: np.ndarray) -> np.ndarray:
        """
        spike_in  : binary {0,1} vector of length n_in
        Returns   : binary output spike vector of length n_out
        """
        # Guard against all-ones spike_in (caused by high-confidence sigmoid
        # encoding) which triggers simultaneous LTD on every weight in every
        # row and causes global suppression.  If more than 75% of inputs are
        # active, randomly drop excess to keep density ≤ 75%.
        density = spike_in.mean()
        if density > 0.75:
            drop_idx = np.where(spike_in == 1)[0]
            n_keep = max(1, int(0.75 * self.n_in))
            mask = np.zeros(self.n_in)
            mask[np.random.choice(drop_idx, n_keep, replace=False)] = 1.0
            spike_in = mask

        # Integrate + stochastic membrane noise (breaks deterministic lock-in)
        self.V  = self.tau * self.V + self.W @ spike_in.astype(float)
        self.V += np.random.randn(self.n_out) * self.V_noise
        # Soft voltage floor: if all neurons have drifted well below threshold
        # (mean V < −0.5), apply a gentle baseline shift to prevent permanent
        # silence caused by LTD-induced weight erosion over many ticks.
        if self.V.mean() < -0.5:
            self.V += 0.1
        spike_out = (self.V >= self.thr).astype(float)
        # Guarantee at least one output spike: if nothing fired, nudge the
        # neuron with the highest voltage over the threshold.  This handles
        # sparse spike_in (e.g. 1-active from argmax fallback) where a single
        # weak W-column can't depolarise any output neuron.
        if spike_out.sum() == 0:
            self.V[np.argmax(self.V)] = self.thr
            spike_out = (self.V >= self.thr).astype(float)
        self.V[spike_out == 1] = 0.0

        # STDP update
        self.x_pre  = self.decay * self.x_pre  + spike_in.astype(float)
        self.x_post = self.decay * self.x_post + spike_out

        dW = self.lr * (
            np.outer(spike_out, self.x_pre) -   # LTP
            np.outer(self.x_post, spike_in)     # LTD
        )
        self.W += dW
        self.W  = np.clip(self.W, -2, 2)        # weight saturation

        # Homeostatic normalisation: rescale each row so its L2 norm stays
        # constant.  Prevents STDP LTD from gradually eroding all excitatory
        # weights while preserving the relative structure STDP learned.
        row_norms = np.linalg.norm(self.W, axis=1, keepdims=True)
        row_norms = np.where(row_norms > 0, row_norms, 1.0)
        target_norm = np.sqrt(self.n_in) * 0.25   # ~same scale as init
        self.W = self.W * (target_norm / row_norms)
        # Mean-centering: pull each row mean back toward 0 so STDP LTD
        # cannot make the whole weight matrix persistently negative.
        # Raised coefficient 0.05 → 0.15 to counter stronger LTD from dense inputs.
        self.W -= self.W.mean(axis=1, keepdims=True) * 0.15
        # Excitatory floor: clip each weight at −target_norm/n_in so no row
        # can be so uniformly negative that W @ spike_in is below threshold
        # for every neuron simultaneously.  This eliminates SNN total-silence ticks.
        w_floor = -(target_norm / self.n_in)
        self.W = np.maximum(self.W, w_floor)
        return spike_out

    def encode_to_spikes(self, x: np.ndarray) -> np.ndarray:
        """Rate-code a continuous vector into approximate binary spikes.

        Scale 1.5 keeps density in a healthy 40–70% range.  A guaranteed
        minimum of one spike (the neuron with highest activation) ensures
        the spiking layer always receives at least one pre-synaptic event,
        preventing total silence from all-negative cortex outputs.
        """
        probs = 1.0 / (1.0 + np.exp(-x * 1.5))
        spikes = (np.random.rand(len(x)) < probs).astype(float)
        if spikes.sum() == 0:
            spikes[np.argmax(x)] = 1.0   # force the strongest unit to fire
        return spikes

    def reset(self):
        self.V[:]    = 0
        self.x_pre[:]  = 0
        self.x_post[:] = 0


# ═══════════════════════════════════════════════════════════════════════════════
# ══  BRAIN-LIKE NEURAL ENGINE  ══════════════════════════════════════════════
# Main class that integrates all fourteen components using the base module.
# ═══════════════════════════════════════════════════════════════════════════════
class BrainLikeNeuralEngine:
    """
    Integrates all fourteen brain-inspired components around a central
    NeuralDataProcessor (from Neural_Algorithm.py) as the base cortex.

    Cognitive tick (process):
        input
         → sensory grounding (⑪)
         → sparse activation (⑤)
         → LIF temporal dynamics (①)
         → oscillatory gating (⑨)
         → base cortex forward (NeuralDataProcessor)
         → attention modulation (②)
         → working memory update (③)
         → predictive processing (④)
         → hierarchical encoding (⑥)
         → spiking layer + STDP (⑭)
         → free energy update (⑬)
         → global workspace competition (⑫)
         → neuromodulator update (⑦)
    """

    def __init__(self, cfg: BrainConfig = BrainConfig()):
        self.cfg = cfg
        self._tick = 0

        # ── Base cortex (NeuralDataProcessor) ──────────────────────────────
        layer_sizes = [cfg.input_size] + cfg.hidden_sizes + [cfg.output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            act = (ActivationFunction.RELU
                   if i < len(layer_sizes) - 2
                   else ActivationFunction.LINEAR)
            layers.append(LayerConfig(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=act,
                dropout_rate=0.0,
            ))
        self.cortex = NeuralDataProcessor(
            layer_configs=layers,
            learning_rate=cfg.learning_rate,
            loss_function=LossFunction.MEAN_SQUARED_ERROR,
            use_precision_calculus=False,
            optimization_method="adam",
        )

        # ── ① LIF temporal dynamics ─────────────────────────────────────────
        # Use input_size so every LIF neuron maps 1-to-1 with x_fused;
        # hidden_sizes[0] is larger and leaves the extra neurons input-starved.
        self.lif = LIFNeuronPopulation(cfg.input_size, cfg)

        # ── ② Attention ──────────────────────────────────────────────────────
        attn_in = cfg.hidden_sizes[-1] if cfg.hidden_sizes else cfg.input_size
        self.attention = MultiHeadAttention(
            d_model=attn_in,
            n_heads=cfg.attention_heads,
            d_k=cfg.attention_dim // cfg.attention_heads,
        )

        # ── ③ Working memory ─────────────────────────────────────────────────
        self.memory = WorkingMemory(
            input_dim=cfg.output_size,
            memory_dim=cfg.memory_size,
            capacity=cfg.memory_capacity,
        )

        # ── ④ Predictive processing ──────────────────────────────────────────
        self.predictor = PredictiveProcessor(
            dim=cfg.input_size, lr=cfg.learning_rate
        )

        # ── ⑥ Cortical hierarchy ─────────────────────────────────────────────
        hier_sizes = [cfg.input_size] + cfg.hidden_sizes
        self.hierarchy = CorticalHierarchy(hier_sizes, lr=cfg.learning_rate)

        # ── ⑦ Neuromodulators ────────────────────────────────────────────────
        self.neuromod = NeuromodulatorState()

        # ── ⑧ Internal replay ────────────────────────────────────────────────
        self.replay = InternalReplay(capacity=200)

        # ── ⑨ Oscillations ───────────────────────────────────────────────────
        self.osc = OscillatorySync(cfg.gamma_freq, cfg.theta_freq, cfg.dt_ms)

        # ── ⑩ Structural plasticity ──────────────────────────────────────────
        self.plasticity = StructuralPlasticity(cfg.synapse_threshold, cfg.growth_prob)

        # ── ⑪ Sensory grounding ──────────────────────────────────────────────
        primary = cfg.input_size // 2
        self.grounding = SensoryGrounding(
            modality_dims={"primary": primary, "context": primary},
            embed_dim=cfg.input_size,
        )

        # ── ⑫ Global workspace ───────────────────────────────────────────────
        self.workspace = GlobalWorkspace(workspace_dim=cfg.output_size)

        # ── ⑬ Free energy ────────────────────────────────────────────────────
        self.free_energy_layer = FreeEnergyLayer(
            input_dim=cfg.input_size,
            latent_dim=cfg.latent_dim,
            lr=cfg.learning_rate,
        )

        # ── ⑭ Spiking layer + STDP ────────────────────────────────────────────
        self.spiking = SpikingLayer(
            n_in=cfg.output_size,
            n_out=cfg.output_size,
            lr=cfg.stdp_lr,
        )

        # ── Cognitive state log ───────────────────────────────────────────────
        self.state_log: List[Dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    def process(self, x: np.ndarray,
                reward: float = 0.0,
                modalities: Optional[Dict[str, np.ndarray]] = None
                ) -> Dict:
        """
        One full cognitive tick.

        Args:
            x         : sensory input vector (length = cfg.input_size)
            reward    : external reward signal (for neuromodulation)
            modalities: optional dict of extra sensory streams for ⑪

        Returns:
            A dict with all intermediate brain-state values.
        """
        self._tick += 1
        state: Dict = {"tick": self._tick}

        # ── ⑨ Oscillatory phase ───────────────────────────────────────────────
        phases      = self.osc.step()
        gate        = self.osc.coupling_gate(phases)
        state["oscillations"] = phases
        state["coupling_gate"] = gate

        # ── ⑪ Sensory grounding ───────────────────────────────────────────────
        if modalities is not None:
            x_fused = self.grounding.fuse(modalities)
        else:
            half = len(x) // 2
            x_fused = self.grounding.fuse({
                "primary": x[:half],
                "context": x[half:] if len(x) > half else np.zeros(half)
            })
        state["fused_input"] = x_fused

        # ── ⑬ Free energy / variational ──────────────────────────────────────
        x_recon, fe = self.free_energy_layer.forward(x_fused)
        self.free_energy_layer.update(x_fused)
        state["free_energy"] = fe
        state["reconstruction"] = x_recon

        # ── ④ Predictive processing ───────────────────────────────────────────
        context      = self.memory.h
        past_pred    = self.predictor.predict(x_fused)
        pred_error   = self.predictor.update(x_fused, x_fused)
        surprise     = self.predictor.surprise
        state["prediction_error"] = pred_error
        state["surprise"] = surprise

        # ── ① LIF temporal dynamics ───────────────────────────────────────────
        # Run 20 ms micro-steps per cognitive tick (1 ms each, dt/tau = 0.05).
        # One step gives only ~1 mV of depolarisation; 20 steps let the
        # strongest neurons accumulate charge and cross the −55 mV threshold,
        # yielding a realistic ~10–20% population firing rate.
        # Rectify so only excitatory (positive) drive is passed to LIF.
        N_LIF_STEPS = 20
        lif_input = np.zeros(self.lif.n)
        chunk = min(len(x_fused), self.lif.n)
        # Floor the gate at 0.35 and ACh at 0.5 so low-trough oscillatory
        # phases don't starve the LIF.  Use abs(x_fused) so negative-valued
        # fused inputs still depolarise (inhibition is modelled by refractory).
        # A small tonic background current (0.02) ensures the membrane always
        # has enough drive to fire at least 1–2 neurons per 20-step window.
        ach_gate  = max(gate, 0.35) * max(self.neuromod.attention_boost, 0.5)
        lif_input[:chunk] = np.abs(x_fused[:chunk]) * ach_gate + 0.08
        tick_spike_count = 0
        for _ in range(N_LIF_STEPS):
            lif_rates = self.lif.step(lif_input)
            tick_spike_count += int(self.lif.spike.sum())
        state["lif_rates"]  = lif_rates
        state["lif_spikes"] = tick_spike_count   # total spikes over the 20 ms window

        # ── ⑤ Sparse activation ───────────────────────────────────────────────
        x_sparse = sparse_kwta(x_fused, self.cfg.sparsity_k)
        state["sparse_active_fraction"] = float(np.mean(x_sparse != 0))

        # ── Base cortex forward (NeuralDataProcessor) ─────────────────────────
        cortex_input = x_sparse + np.random.randn(*x_sparse.shape) * self.neuromod.noise_level
        cortex_out   = self.cortex.predict(cortex_input)
        state["cortex_output"] = cortex_out

        # ── ⑥ Hierarchical encoding ───────────────────────────────────────────
        concept_vec  = self.hierarchy.encode(x_fused)
        state["concept_vector"] = concept_vec

        # ── ② Attention ───────────────────────────────────────────────────────
        attn_input   = cortex_out if len(cortex_out) == self.attention.W_Q.shape[1] \
                       else concept_vec
        attn_out     = self.attention.forward(attn_input) * self.neuromod.attention_boost
        state["attention_output"] = attn_out
        state["attention_weights"] = self.attention.last_attn_weights

        # ── ③ Working memory ──────────────────────────────────────────────────
        # Enrich the GRU input with the prediction error (truncated to
        # output_size) so memory encodes both what happened AND how
        # surprising it was.  This raises memory_norm from ~0.15 to a
        # more informative range without changing the GRU API.
        pred_err_slice = pred_error[:self.cfg.output_size] \
            if len(pred_error) >= self.cfg.output_size \
            else np.pad(pred_error, (0, self.cfg.output_size - len(pred_error)))
        mem_input   = cortex_out + 0.5 * pred_err_slice
        mem_state   = self.memory.update(mem_input)
        state["memory_state"] = mem_state

        # ── ⑭ Spiking layer + STDP ────────────────────────────────────────────
        spike_in    = self.spiking.encode_to_spikes(cortex_out)
        spike_out   = self.spiking.forward(spike_in)
        state["spike_out"]   = spike_out
        state["spike_count"] = int(np.sum(spike_out))

        # ── ⑫ Global workspace competition ────────────────────────────────────
        # Each module's salience = tanh-normalised L2 norm, scaled by a
        # module-specific weight that reflects biological plausibility rather
        # than artificially boosting one winner.  Spiking gets a mild gate
        # bonus (up to +0.3) only when oscillatory coupling is high.
        def sal(v, scale: float = 1.0):
            norm = float(np.linalg.norm(v))
            # tanh squashes huge norms so no single module dominates by scale
            return float(np.tanh(norm * 0.5)) * scale if norm > 0 else 0.0

        mem_slice   = mem_state[:self.cfg.output_size]
        ach = max(self.neuromod.attention_boost, 0.5)
        n_spikes    = int(np.sum(spike_out))
        n_out       = len(spike_out)
        # Spiking salience: base sal() + activity fraction bonus.
        # activity_frac = n_spikes/n_out normalises by the layer's own capacity,
        # so a "busy" tick (6/8 = 75%) gets a meaningful boost over a "quiet"
        # tick (3/8 = 37%) regardless of absolute cortex magnitude.
        # This keeps spiking competitive even when attn_out norm grows as
        # training converges, preventing attention from dominating late runs.
        activity_frac = n_spikes / max(n_out, 1)
        # activity bonus 0.12×frac: at mean 5/8 adds +0.075, at high 7/8 adds +0.105.
        # Enough to tip balance on active ticks; lets attention win on quiet ticks.
        spike_sal   = sal(spike_out, 1.0) + 0.05 * gate + 0.12 * activity_frac
        self.workspace.register("cortex",    cortex_out,  sal(cortex_out,  1.0))
        self.workspace.register("attention", attn_out,    sal(attn_out,    1.0 / ach))
        self.workspace.register("memory",    mem_slice,   sal(mem_slice,   0.95))
        self.workspace.register("spiking",   spike_out,   spike_sal)
        winner, broadcast = self.workspace.broadcast_step()
        state["workspace_winner"]    = winner
        state["workspace_broadcast"] = broadcast

        # integrated output
        integrated = self.workspace.integrate(cortex_out, mem_state[:self.cfg.output_size])
        state["integrated_output"] = integrated

        # ── ⑦ Neuromodulator update ───────────────────────────────────────────
        arousal = float(np.mean(np.abs(pred_error)))
        self.neuromod.update(
            reward=reward, surprise=surprise, arousal=arousal,
            d_decay=self.cfg.dopamine_decay, s_decay=self.cfg.serotonin_decay
        )
        state["neuromodulators"] = {
            "dopamine":       self.neuromod.dopamine,
            "serotonin":      self.neuromod.serotonin,
            "acetylcholine":  self.neuromod.acetylcholine,
            "norepinephrine": self.neuromod.norepinephrine,
        }
        state["effective_lr_scale"] = self.neuromod.effective_lr_scale

        # ── ⑧ Store to replay buffer ──────────────────────────────────────────
        # Store x_fused (input_size-dimensional) rather than cortex_out
        # (output_size=8).  Richer patterns → more varied dream spikes.
        self.replay.store(x_fused)

        # ── ⑩ Structural plasticity (every 10 ticks) ─────────────────────────
        if self._tick % 10 == 0:
            self.cortex.weights = self.plasticity.apply(self.cortex.weights)
            state["plasticity_pruned"] = self.plasticity.pruned_count
            state["plasticity_grown"]  = self.plasticity.grown_count

        self.state_log.append(state)
        return state

    # ─────────────────────────────────────────────────────────────────────────
    def learn(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              verbose: bool = True):
        """
        Supervised learning with neuromodulatory learning-rate scaling.
        Delegates to base NeuralDataProcessor.train().
        """
        # Scale learning rate by dopamine level
        scaled_lr = self.cortex.learning_rate * self.neuromod.effective_lr_scale
        original_lr = self.cortex.learning_rate
        self.cortex.learning_rate = np.clip(scaled_lr, 1e-6, 1.0)

        if verbose:
            print(f"\n[BrainEngine] Learning  (dopamine={self.neuromod.dopamine:.3f} "
                  f"→ effective LR={self.cortex.learning_rate:.5f})")

        self.cortex.train(X, y, epochs=epochs, batch_size=batch_size,
                          verbose=verbose, early_stopping=True)
        self.cortex.learning_rate = original_lr

    # ─────────────────────────────────────────────────────────────────────────
    def dream(self, steps: int = 20, verbose: bool = False) -> List[np.ndarray]:
        """
        Self-generated internal replay — offline consolidation.
        Runs the spiking layer on dreamed (noise-perturbed) patterns;
        STDP strengthens frequently co-activated paths.

        Two additions vs. the naïve version:
        • Reset spiking voltage between dream steps so accumulated charge
          from step N-1 doesn't fix the same neurons firing on step N.
        • Apply independent noise to each dreamed pattern so spike patterns
          vary realistically (models the stochastic nature of dreaming).
        """
        n_in = self.spiking.n_in
        outputs = []
        for i in range(steps):
            dreamed = self.replay.dream_step()
            if dreamed is None:
                break
            # Independent per-step jitter makes each dream pattern unique
            jitter   = np.random.randn(*dreamed.shape) * 0.15
            pattern  = dreamed + jitter
            # Project from replay dim (input_size) down to spiking n_in.
            # Slicing x_fused directly gives healthy ~N(0,0.5) values with
            # good sigmoid probs; routing through cortex.predict() produces
            # mostly-negative outputs after shallow training that kill spikes.
            if len(pattern) >= n_in:
                pattern_proj = pattern[:n_in]
            else:
                pattern_proj = np.pad(pattern, (0, n_in - len(pattern)))
            spike_in = self.spiking.encode_to_spikes(pattern_proj)
            # Reset voltage between dreams (no carryover from prior step)
            self.spiking.V[:] = 0.0
            spike_out = self.spiking.forward(spike_in)
            outputs.append(spike_out.copy())
            if verbose:
                print(f"  [Dream {i+1}] spikes={int(np.sum(spike_out))}")
        return outputs

    # ─────────────────────────────────────────────────────────────────────────
    def reset_temporal_state(self):
        """Reset all time-dependent state (LIF, memory, oscillator)."""
        self.lif.reset()
        self.memory.reset()
        self.spiking.reset()
        self.osc.t = 0.0
        self._tick = 0

    # ─────────────────────────────────────────────────────────────────────────
    def cognitive_report(self) -> Dict:
        """Return a comprehensive snapshot of the current brain state."""
        last = self.state_log[-1] if self.state_log else {}
        return {
            "tick":               self._tick,
            "neuromodulators":    last.get("neuromodulators", {}),
            "free_energy":        last.get("free_energy", None),
            "surprise":           last.get("surprise", None),
            "workspace_winner":   last.get("workspace_winner", None),
            "lif_spikes":         last.get("lif_spikes", 0),
            "spike_count":        last.get("spike_count", 0),
            "sparse_fraction":    last.get("sparse_active_fraction", None),
            "memory_norm":        float(np.linalg.norm(self.memory.h)),
            "replay_buffer_size": len(self.replay.memory),
            "plasticity_pruned":  self.plasticity.pruned_count,
            "plasticity_grown":   self.plasticity.grown_count,
            "oscillations":       last.get("oscillations", {}),
            "coupling_gate":      last.get("coupling_gate", None),
        }

    # ─────────────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        cfg = self.cfg
        arch = [cfg.input_size] + cfg.hidden_sizes + [cfg.output_size]
        return (f"BrainLikeNeuralEngine("
                f"arch={arch}, "
                f"tick={self._tick}, "
                f"dopamine={self.neuromod.dopamine:.3f})")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════
def demonstrate_brain_engine():
    print("\n" + "═" * 70)
    print("  BRAIN-LIKE NEURAL ENGINE — DEMONSTRATION  (v2 — all systems active)")
    print("═" * 70)

    cfg = BrainConfig(
        input_size=16,
        hidden_sizes=[32, 16],
        output_size=8,
        sparsity_k=0.15,
        learning_rate=0.005,
        attention_heads=4,
        attention_dim=16,
        memory_size=8,
        dopamine_decay=0.92,   # slightly faster response to reward swings
        stdp_lr=0.01,          # stronger STDP so spiking layer learns faster
    )
    engine = BrainLikeNeuralEngine(cfg)
    print(f"\n{engine}\n")

    # ── Phase 1: supervised learning on a simple pattern ──────────────────────
    print("Phase 1: Supervised Learning")
    print("─" * 70)
    np.random.seed(42)
    X_train = np.random.randn(80, cfg.input_size)
    y_train = np.random.randn(80, cfg.output_size)
    engine.learn(X_train, y_train, epochs=40, batch_size=16, verbose=True)

    # ── Phase 2: cognitive ticks with reward ──────────────────────────────────
    print("\nPhase 2: Cognitive Processing (30 ticks)")
    print("─" * 70)
    print(f"  {'Tick':>4}  {'Winner':<12} {'FE':>7}  {'Surprise':>8}  "
          f"{'LIF⚡':>5}  {'SNN⚡':>5}  {'Gate':>5}  {'DA':>5}  {'Pruned':>6}")
    print("  " + "─" * 68)

    workspace_counts: Dict[str, int] = {}
    for t in range(30):
        x_sense = np.random.randn(cfg.input_size)
        # reward oscillates between −1 and +1 so dopamine has real dynamics
        reward  = float(np.sin(t * 0.7))
        state   = engine.process(x_sense, reward=reward)

        winner  = state["workspace_winner"]
        workspace_counts[winner] = workspace_counts.get(winner, 0) + 1
        pruned  = state.get("plasticity_pruned", engine.plasticity.pruned_count)
        grown   = state.get("plasticity_grown",  engine.plasticity.grown_count)

        print(f"  {state['tick']:>4}  {winner:<12} "
              f"{state['free_energy']:>+7.3f}  "
              f"{state['surprise']:>8.4f}  "
              f"{state['lif_spikes']:>5}  "
              f"{state['spike_count']:>5}  "
              f"{state['coupling_gate']:>5.3f}  "
              f"{state['neuromodulators']['dopamine']:>5.3f}  "
              f"{pruned:>6}")

    # ── Winner distribution ───────────────────────────────────────────────────
    print("\n  Global Workspace — winner distribution over 30 ticks:")
    for mod, cnt in sorted(workspace_counts.items(), key=lambda x: -x[1]):
        bar = "█" * cnt
        print(f"    {mod:<12} {bar}  ({cnt})")

    # ── Phase 3: dreaming / offline consolidation ─────────────────────────────
    print("\nPhase 3: Offline Dreaming (30 steps)")
    print("─" * 70)
    dream_outputs = engine.dream(steps=30, verbose=False)
    spike_counts  = [int(np.sum(d)) for d in dream_outputs]
    print(f"  Spike counts: {spike_counts}")
    print(f"  Mean={np.mean(spike_counts):.2f}  "
          f"Max={max(spike_counts)}  "
          f"Active steps={sum(1 for s in spike_counts if s > 0)}/30")

    # ── Cognitive report ──────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Cognitive State Report:")
    report = engine.cognitive_report()
    for k, v in report.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"      {kk}: {vv:.4f}" if isinstance(vv, float) else f"      {kk}: {vv}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "═" * 70)
    print("  DEMONSTRATION COMPLETE")
    print("═" * 70)


if __name__ == "__main__":
    demonstrate_brain_engine()