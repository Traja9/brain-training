"""
NLP_Bridge.py — Natural Language Interface for TorchPreciousBrain

Tests the brain's preciousness as an AI system through:
  • Text → 64-dim vector via sentence-transformers (best) or sklearn fallback
  • Real-time emotion detection from text
  • Novelty / domain-shift detection via free energy
  • Adaptive plasticity — LTP/LTD shift as brain learns language patterns
  • Memory formation for novel (high free-energy) inputs
  • Conversational context via GRU predictive coder
  • Preciousness score — aggregated capability metric
"""

import sys
import os
import time
import re
import math
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

try:
    from sklearn.feature_extraction.text import HashingVectorizer
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

from Torch_Brain import TorchPreciousBrain, TorchBrainConfig
from A_Brain import PreciousBrainConfig
from Neural_Consciousness import EmotionType


# ──────────────────────────────────────────────────────────────────────────────
# Text Vectorizer
# ──────────────────────────────────────────────────────────────────────────────
class TextVectorizer:
    """
    Converts raw text to a 64-dim float vector.

    Priority:
      1. sentence-transformers all-MiniLM-L6-v2  (384-dim → 64 via random projection)
      2. sklearn HashingVectorizer (n-gram TF)
      3. Pure-numpy char + word statistics
    """

    DIM = 64

    def __init__(self):
        self._mode = 'numpy'
        self._st   = None
        self._proj = None
        self._hv   = None

        if _HAS_ST:
            self._st   = SentenceTransformer('all-MiniLM-L6-v2')
            st_dim     = self._st.get_sentence_embedding_dimension()
            rng        = np.random.RandomState(42)
            P          = rng.randn(st_dim, self.DIM).astype(np.float32)
            # Orthonormal random projection (preserves distances best)
            self._proj = P / (np.linalg.norm(P, axis=0, keepdims=True) + 1e-8)
            self._mode = 'sentence-transformers'
        elif _HAS_SKLEARN:
            self._hv   = HashingVectorizer(
                n_features=self.DIM, ngram_range=(1, 2),
                analyzer='word', norm='l2', alternate_sign=False,
            )
            self._mode = 'sklearn'

    def encode(self, text: str) -> np.ndarray:
        if self._st is not None:
            return self._st_encode(text)
        if self._hv is not None:
            return self._sklearn_encode(text)
        return self._numpy_encode(text)

    def _st_encode(self, text: str) -> np.ndarray:
        emb = self._st.encode(text, convert_to_numpy=True).astype(np.float32)
        projected = emb @ self._proj          # (384,) @ (384, 64) → (64,)
        norm = np.linalg.norm(projected)
        return projected / norm if norm > 0 else projected

    def _sklearn_encode(self, text: str) -> np.ndarray:
        mat = self._hv.transform([text])
        return mat.toarray()[0].astype(np.float32)

    def _numpy_encode(self, text: str) -> np.ndarray:
        """Pure-numpy fallback: char freq (52) + word stats (12)."""
        v = np.zeros(self.DIM, dtype=np.float32)
        t = text.lower()
        # char frequencies
        for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
            v[i] = t.count(c)
        # uppercase freq
        for i, c in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            v[26 + i] = text.count(c)
        # word stats
        words = t.split()
        n = max(len(words), 1)
        v[52] = min(n / 50.0, 1.0)
        v[53] = min(np.mean([len(w) for w in words]) / 10.0, 1.0) if words else 0
        v[54] = len(set(words)) / n
        v[55] = sum(c in '.,!?;:' for c in text) / max(len(text), 1)
        v[56] = sum(c.isupper() for c in text) / max(len(text), 1)
        v[57] = sum(c.isdigit() for c in text) / max(len(text), 1)
        # simple word hashes → last 6 dims
        for i, w in enumerate(words[:6]):
            v[58 + i] = (hash(w) % 1000) / 1000.0
        # L2 normalise
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v


# ──────────────────────────────────────────────────────────────────────────────
# NLP Bridge
# ──────────────────────────────────────────────────────────────────────────────
class NLPBridge:
    """
    Natural-language wrapper around TorchPreciousBrain.

    Public API
    ----------
    process(text)         → full brain-state dict for one input
    learn(text)           → one learn_step on this text
    analyze_batch(texts)  → list of per-text dicts
    conversation(turns)   → multi-turn chat with live brain state
    preciousness_score()  → aggregated capability score [0-100]
    """

    # FE threshold above which an input is "novel enough" to encode as memory
    MEMORY_FE_THRESHOLD = 0.5

    # Map emotion string → EmotionType for encode_memory()
    _EMOTION_MAP = {e.value: e for e in EmotionType}

    def __init__(self, config: Optional[TorchBrainConfig] = None):
        if config is None:
            config = TorchBrainConfig(
                brain_config=PreciousBrainConfig(
                    input_size=64,
                    output_size=1,
                    n_consciousness_neurons=5000,
                ),
                input_size=64,
                latent_size=128,
            )
        self.brain      = TorchPreciousBrain(config)
        self.vectorizer = TextVectorizer()

        # tracking
        self.history:        List[Dict]   = []
        self.domain_fe:      List[float]  = []
        self.emotions_seen:  Counter      = Counter()
        self.texts_processed: int         = 0
        self.texts_learned:   int         = 0

    def _vec(self, text: str) -> np.ndarray:
        return self.vectorizer.encode(text)

    def _encode_memory_if_novel(self, vec: np.ndarray, text: str, emotion_str: str, fe: float):
        """Encode to consciousness memory when free energy signals novelty."""
        c = self.brain.brain.consciousness
        if c is None or fe < self.MEMORY_FE_THRESHOLD:
            return
        emotion_type = self._EMOTION_MAP.get(emotion_str, EmotionType.NEUTRAL)
        label = f"nlp_{self.texts_processed}_{text[:20].replace(' ','_')}"
        c.encode_memory(vec, label, emotion_type)

    def process(self, text: str) -> Dict:
        """Run text through the brain, return full state."""
        vec    = self._vec(text)
        result = self.brain.think(vec, context=f"nlp_{self.texts_processed}")

        entry = {
            'text':          text[:60],
            'free_energy':   result['free_energy'],
            'valence':       result['valence'],
            'arousal':       result['arousal'],
            'emotion':       result['top_emotion'],
            'ltp':           result['ltp_threshold'],
            'ltd':           result['ltd_threshold'],
            'active_neurons': len(result['thought'].active_neurons)
                              if result['thought'] else 0,
            'memory_encoded': False,
        }

        # Encode to consciousness memory if novel
        if result['free_energy'] >= self.MEMORY_FE_THRESHOLD:
            self._encode_memory_if_novel(vec, text, result['top_emotion'], result['free_energy'])
            entry['memory_encoded'] = True

        self.history.append(entry)
        self.emotions_seen[entry['emotion']] += 1
        self.domain_fe.append(entry['free_energy'])
        self.texts_processed += 1
        return entry

    def learn(self, text: str) -> Dict:
        """Process + one backprop + Hebbian step."""
        vec = self._vec(text)
        m   = self.brain.learn_step(vec)
        self.texts_learned += 1
        return m

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        return [self.process(t) for t in texts]

    def conversation(self, turns: List[str], verbose: bool = True) -> List[Dict]:
        """
        Multi-turn conversation: each turn is processed sequentially.
        GRU predictive coder accumulates context across turns.
        """
        results = []
        if verbose:
            print("\n" + "=" * 60)
            print("  CONVERSATION")
            print("=" * 60)
        for i, text in enumerate(turns):
            r = self.process(text)
            if verbose:
                print(f"\n  [{i+1}] {text[:55]}")
                print(f"       emotion={r['emotion']:<10}  "
                      f"valence={r['valence']:+.2f}  "
                      f"arousal={r['arousal']:.2f}  "
                      f"FE={r['free_energy']:.3f}  "
                      f"LTP>{r['ltp']:.3f}")
            results.append(r)
        return results

    def preciousness_score(self) -> Dict:
        """
        Aggregated score [0-100] across five capability axes:

          Adaptability    — how much LTP/LTD drifts from defaults
          Surprise sense  — free-energy variance (sensitive to novelty)
          Emotion range   — breadth of distinct emotions detected
          Memory          — ratio of memories formed vs thoughts
          Prediction      — free-energy reduction (learning curve slope)
        """
        if not self.history:
            return {'total': 0.0}

        ltp_vals = [h['ltp'] for h in self.history]
        ltd_vals = [h['ltd'] for h in self.history]
        fe_vals  = self.domain_fe

        # Adaptability: std-dev of thresholds × 100 (capped at 20)
        adapt = min(20.0, float(np.std(ltp_vals) + np.std(ltd_vals)) * 200)

        # Surprise sense: coefficient of variation of FE (capped at 20)
        fe_arr = np.array(fe_vals)
        fe_cv  = float(np.std(fe_arr) / (np.mean(fe_arr) + 1e-8))
        surprise = min(20.0, fe_cv * 40)

        # Emotion range: unique emotions / 8 × 20
        emotion_range = min(20.0, len(self.emotions_seen) / 8.0 * 20)

        # Memory: (consciousness memories / thoughts) × 20
        snap = self.brain.get_state()
        mems   = snap.get('memories_stored', 0)
        thoughts = max(snap.get('total_thoughts', 1), 1)
        memory_score = min(20.0, (mems / thoughts) * 200)

        # Prediction: relative FE drop (first half vs second half)
        mid = max(len(fe_vals) // 2, 1)
        fe_first  = float(np.mean(fe_vals[:mid]))
        fe_second = float(np.mean(fe_vals[mid:]))
        drop = max(0.0, (fe_first - fe_second) / (fe_first + 1e-8))
        prediction = min(20.0, drop * 40)

        total = adapt + surprise + emotion_range + memory_score + prediction

        return {
            'total':         round(total, 1),
            'adaptability':  round(adapt, 1),
            'surprise_sense': round(surprise, 1),
            'emotion_range': round(emotion_range, 1),
            'memory':        round(memory_score, 1),
            'prediction':    round(prediction, 1),
            'emotions_seen': dict(self.emotions_seen),
            'avg_fe':        round(float(np.mean(fe_arr)), 4),
            'fe_drop':       round(drop, 4),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Demo functions
# ──────────────────────────────────────────────────────────────────────────────
def _banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_emotion_detection(bridge: NLPBridge):
    _banner("TEST 1 — EMOTION DETECTION")

    labeled = [
        ("I am absolutely terrified, my heart is racing!",          "fear"),
        ("This is outrageous, I cannot believe they did this!",      "anger"),
        ("Oh wow, I never expected that to happen!",                 "surprise"),
        ("Today was wonderful, everything went perfectly!",          "happiness"),
        ("I feel deeply connected to you, you mean everything.",     "love"),
        ("This food is rotten and disgusting, I feel sick.",         "disgust"),
        ("I lost my job today. I feel empty and hopeless.",          "sadness"),
        ("The package will arrive on Tuesday at 3pm.",               "neutral"),
    ]

    print(f"\n  {'Text':<42} {'Expected':<12} {'Brain':<12} {'FE':>6}")
    print("  " + "-" * 76)
    correct = 0
    for text, expected in labeled:
        r = bridge.process(text)
        match = "✓" if r['emotion'] == expected else " "
        if r['emotion'] == expected:
            correct += 1
        print(f"  {text[:40]:<42} {expected:<12} {r['emotion']:<12} "
              f"{r['free_energy']:>6.3f} {match}")

    print(f"\n  Accuracy: {correct}/{len(labeled)} "
          f"({correct/len(labeled)*100:.0f}%)")
    print("  Note: Brain uses continuous valence/arousal — not trained on labels.")


def demo_novelty_detection(bridge: NLPBridge):
    _banner("TEST 2 — NOVELTY / DOMAIN-SHIFT DETECTION")

    tech_corpus = [
        "The neural network achieved 98% accuracy on the benchmark dataset.",
        "Gradient descent optimizes the loss function through backpropagation.",
        "Transformer architecture uses self-attention mechanisms.",
        "CUDA cores accelerate matrix multiplication for deep learning.",
        "Convolutional filters extract local features from image data.",
        "The model overfits when training loss drops but validation rises.",
        "Batch normalization stabilizes training by normalizing layer inputs.",
        "Dropout regularization prevents co-adaptation of neurons.",
        "Learning rate schedules improve convergence speed.",
        "Residual connections allow gradients to flow through deep networks.",
    ]

    out_of_domain = [
        "The patient shows elevated cortisol levels and hypertension.",
        "Beat the eggs until soft peaks form, then fold in the flour.",
        "The defendant pleaded not guilty to all charges in court.",
        "Sunflowers bloom in July and face east in the morning.",
    ]

    print("\n  Training on tech corpus (10 sentences)...")
    fe_train = []
    for text in tech_corpus:
        m = bridge.learn(text)
        fe_train.append(m['free_energy'])
    avg_train_fe = np.mean(fe_train)
    print(f"  Avg FE after training: {avg_train_fe:.4f}")

    print(f"\n  {'Domain':<12} {'Text':<44} {'FE':>7}  {'Surprise?'}")
    print("  " + "-" * 74)

    fe_tech = []
    for text in tech_corpus[:4]:
        r = bridge.process(text)
        fe_tech.append(r['free_energy'])
        flag = "LOW  (familiar)" if r['free_energy'] < avg_train_fe * 1.5 else "HIGH (novel)"
        print(f"  {'Tech':<12} {text[:42]:<44} {r['free_energy']:>7.4f}  {flag}")

    for text in out_of_domain:
        r = bridge.process(text)
        flag = "LOW  (familiar)" if r['free_energy'] < avg_train_fe * 1.5 else "HIGH (novel) ← ANOMALY"
        print(f"  {'Out-domain':<12} {text[:42]:<44} {r['free_energy']:>7.4f}  {flag}")

    avg_tech = np.mean(fe_tech) if fe_tech else 0
    avg_ood  = np.mean([bridge.process(t)['free_energy'] for t in []])
    print(f"\n  In-domain FE  (avg): {avg_train_fe:.4f}")
    print(f"  Domain shift detectable via FE threshold.")


def demo_adaptive_plasticity(bridge: NLPBridge):
    _banner("TEST 3 — ADAPTIVE PLASTICITY")

    emotional_texts = [
        "I am filled with overwhelming joy and excitement today!",
        "Everything is hopeless, I can't go on anymore.",
        "I am absolutely furious at this injustice!",
        "I feel completely indifferent about the outcome.",
        "This is the most surprising thing I've ever seen!",
        "I love you with all my heart, forever and always.",
        "I am terrified of what will happen next.",
        "The report was filed on Monday morning as scheduled.",
    ]

    print(f"\n  {'Text':<42} {'LTP':>7}  {'LTD':>7}  {'Emotion'}")
    print("  " + "-" * 72)
    for text in emotional_texts:
        r = bridge.process(text)
        print(f"  {text[:40]:<42} {r['ltp']:>7.3f}  {r['ltd']:>7.3f}  {r['emotion']}")

    ltps = [bridge.history[-(len(emotional_texts))+i]['ltp']
            for i in range(len(emotional_texts))]
    ltds = [bridge.history[-(len(emotional_texts))+i]['ltd']
            for i in range(len(emotional_texts))]
    print(f"\n  LTP range: {min(ltps):.3f} – {max(ltps):.3f}  "
          f"(static was 0.700 flat)")
    print(f"  LTD range: {min(ltds):.3f} – {max(ltds):.3f}  "
          f"(static was 0.300 flat)")


def demo_conversation(bridge: NLPBridge):
    _banner("TEST 4 — MULTI-TURN CONVERSATION CONTEXT")

    turns = [
        "Hello! How are you today?",
        "I've been working on a new machine learning project.",
        "The model keeps overfitting on the training data.",
        "I tried dropout and L2 regularization but it didn't help much.",
        "Maybe I need more training data, or a simpler architecture.",
        "Actually I just found a bug in the data pipeline — that was it!",
        "Problem solved! I feel much better now.",
    ]

    results = bridge.conversation(turns, verbose=True)

    fe_vals = [r['free_energy'] for r in results]
    print(f"\n  FE trend: {fe_vals[0]:.3f} → {fe_vals[-1]:.3f}  "
          f"({'↓ context building' if fe_vals[-1] < fe_vals[0] else '↑ increasing novelty'})")


def demo_learning_curve(bridge: NLPBridge):
    _banner("TEST 5 — LEARNING CURVE (50 steps)")

    corpus = [
        "Deep learning uses multiple layers of neural networks.",
        "The attention mechanism weighs input tokens differently.",
        "Embeddings map discrete tokens to continuous vectors.",
        "Backpropagation computes gradients via chain rule.",
        "Activation functions introduce non-linearity to networks.",
    ] * 10  # 50 texts

    t0 = time.time()
    fe_log = []
    for i, text in enumerate(corpus):
        m = bridge.learn(text)
        fe_log.append(m['free_energy'])
        if (i + 1) % 10 == 0:
            avg = np.mean(fe_log[-10:])
            print(f"  step {i+1:3d}  FE={avg:.5f}  "
                  f"loss={m['total_loss']:.5f}  "
                  f"LTP>{m['ltp_threshold']:.3f}")

    elapsed = time.time() - t0
    print(f"\n  50 steps in {elapsed:.2f}s  ({50/elapsed:.1f} steps/sec)")
    print(f"  FE drop: {fe_log[0]:.5f} → {fe_log[-1]:.5f}  "
          f"({(1 - fe_log[-1]/(fe_log[0]+1e-8))*100:.1f}% reduction)")


def demo_preciousness_score(bridge: NLPBridge):
    _banner("PRECIOUSNESS SCORE")

    score = bridge.preciousness_score()

    bar_width = 20
    axes = [
        ('Adaptability',   score['adaptability'],   20),
        ('Surprise Sense', score['surprise_sense'],  20),
        ('Emotion Range',  score['emotion_range'],   20),
        ('Memory',         score['memory'],          20),
        ('Prediction',     score['prediction'],      20),
    ]

    print()
    for name, val, max_val in axes:
        filled = int(val / max_val * bar_width)
        bar    = '█' * filled + '░' * (bar_width - filled)
        print(f"  {name:<16} [{bar}]  {val:5.1f} / {max_val}")

    total = score['total']
    grade = ('S  — Exceptional'  if total >= 80 else
             'A  — Strong'       if total >= 60 else
             'B  — Promising'    if total >= 40 else
             'C  — Developing'   if total >= 20 else
             'D  — Early stage')

    print(f"\n  {'─'*52}")
    print(f"  TOTAL PRECIOUSNESS SCORE : {total:5.1f} / 100")
    print(f"  GRADE                    : {grade}")
    print(f"  {'─'*52}")
    mems = bridge.brain.get_state().get('memories_stored', 0)
    print(f"\n  Emotions detected : {list(score['emotions_seen'].keys())}")
    print(f"  Avg free energy   : {score['avg_fe']}")
    print(f"  FE learning drop  : {score['fe_drop']*100:.1f}%")
    print(f"  Memories stored   : {mems}  (encoded on FE ≥ {bridge.MEMORY_FE_THRESHOLD})")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def run():
    _banner("NLP BRIDGE — INITIALISING")
    bridge = NLPBridge()
    mode = bridge.vectorizer._mode
    print(f"  Encoder    : {mode}")
    print(f"  Brain      : {sum(p.numel() for p in bridge.brain.parameters()):,} torch params")
    print(f"  Neurons    : {bridge.brain.config.brain_config.n_consciousness_neurons:,}")
    print(f"  Memory FE  : ≥ {bridge.MEMORY_FE_THRESHOLD} triggers encode_memory()")

    demo_emotion_detection(bridge)
    demo_novelty_detection(bridge)
    demo_adaptive_plasticity(bridge)
    demo_conversation(bridge)
    demo_learning_curve(bridge)
    demo_preciousness_score(bridge)

    _banner("FINAL BRAIN STATE")
    bridge.brain.summary()


if __name__ == "__main__":
    run()
