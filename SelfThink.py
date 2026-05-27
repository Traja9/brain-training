"""
SelfThink.py — Autonomous Thought Loop

Wires A_Brain.py (PreciousBrain) + Hybrid_LM.py into a closed loop:

  Brain emotional state  ──►  emotion_to_prompt()
                                      │
                               Hybrid_LM.generate()
                                      │
                              text_to_vector()
                                      │
                         brain.think() + brain.feel()
                                      │
                         Updated emotional state  ──► (repeat)

The brain is no longer static — it drives its own input, experiences
its own output, and evolves continuously without external prompting.
"""

import os
import sys
import time
import json
import threading
from typing import Optional, List, Dict

import numpy as np

from A_Brain import PreciousBrain, PreciousBrainConfig
from Hybrid_LM import Hybrid_LM
from HPCLM import CharTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Emotion → language prompt mapping
# ──────────────────────────────────────────────────────────────────────────────

_EMOTION_PROMPTS = {
    'fear':      ['I am afraid of ',    'The danger is ',       'I must escape '],
    'happiness': ['I feel joyful about ','The best thing is ',  'I love the feeling of '],
    'sadness':   ['I feel sad because ', 'I miss ',             'It hurts to think about '],
    'anger':     ['I am frustrated by ', 'It angers me that ',  'Why does '],
    'surprise':  ['I am amazed that ',  'Unexpectedly ',        'I never knew that '],
    'disgust':   ['Something feels wrong with ', 'I dislike ',  'It bothers me that '],
    'neutral':   ['I wonder about ',    'I think about ',       'The world is '],
    'curiosity': ['I am curious about ','What if ',             'I want to understand '],
}

_DEFAULT_PROMPTS = ['I think ', 'The ', 'It is ']


def emotion_to_prompt(emotion_name: str, arousal: float, rng: np.random.RandomState) -> str:
    """Pick a prompt based on current emotion. High arousal = more intense prompts."""
    key = emotion_name.lower()
    options = _EMOTION_PROMPTS.get(key, _DEFAULT_PROMPTS)
    # High arousal picks from the more intense (later) options
    idx = min(len(options) - 1, int(arousal * len(options)))
    return options[idx]


def text_to_vector(text: str, size: int) -> np.ndarray:
    """
    Encode text as a fixed-size float vector for feeding to PreciousBrain.
    Uses normalised character frequency — deterministic, no learned embeddings.
    """
    vec = np.zeros(size, dtype=np.float32)
    if not text:
        return vec
    for i, ch in enumerate(text):
        vec[i % size] += ord(ch) / 127.0
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# SelfThink — the autonomous loop
# ──────────────────────────────────────────────────────────────────────────────

class SelfThink:
    """
    Autonomous thought loop connecting PreciousBrain and Hybrid_LM.

    Call run() to start thinking. It loops until stopped or max_thoughts
    is reached. Each iteration:
      1. Reads brain's emotional state
      2. Picks a prompt matching the emotion
      3. Generates a thought with Hybrid_LM
      4. Feeds the thought back to the brain
      5. Sleeps (shorter when aroused, longer when calm)
      6. Dreams automatically when fatigued > 0.7
    """

    def __init__(
        self,
        brain:          PreciousBrain,
        lm:             Hybrid_LM,
        base_interval:  float = 1.5,   # seconds between thoughts at arousal=0.5
        max_tokens:     int   = 40,
        temperature:    float = 0.85,
        top_k:          int   = 8,
        log_file:       Optional[str] = 'results/selfthink_log.jsonl',
        seed:           int   = 0,
    ):
        self.brain         = brain
        self.lm            = lm
        self.base_interval = base_interval
        self.max_tokens    = max_tokens
        self.temperature   = temperature
        self.top_k         = top_k
        self.log_file      = log_file
        self.rng           = np.random.RandomState(seed)

        self.thought_log:  List[Dict] = []
        self.running       = False
        self._count        = 0
        self._log_fh       = None

        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            self._log_fh = open(log_file, 'w')

    # ── single thought cycle ──────────────────────────────────────────────────

    def think_once(self) -> Dict:
        """Run one thought cycle. Returns the thought record."""
        # 1. Read brain state
        snap    = self.brain.get_consciousness_snapshot()
        emotion = snap.get('current_emotion', 'neutral')
        arousal = float(self.brain.state.arousal_level)
        fatigue = float(self.brain.state.fatigue_level)

        # 2. Pick prompt based on emotion
        prompt  = emotion_to_prompt(emotion, arousal, self.rng)

        # 3. Generate thought
        self.lm.reset_context()
        generated = self.lm.generate(
            prompt,
            max_tokens  = self.max_tokens,
            temperature = self.temperature + arousal * 0.3,
            top_k       = self.top_k,
            learn       = False,
        )
        full_thought = prompt + generated

        # 4. Feed thought back to brain
        input_size = self.brain.config.input_size
        vec = text_to_vector(full_thought, input_size)
        self.brain.think(vec, context={'name': f'self_thought_{self._count}'})
        self.brain.feel(vec, {
            'heart_rate': 70 + arousal * 30,
            'arousal':    arousal,
        })

        # 5. Build record
        record = {
            'n':       self._count,
            'time':    round(time.time(), 2),
            'emotion': emotion,
            'arousal': round(arousal, 3),
            'fatigue': round(fatigue, 3),
            'prompt':  prompt,
            'thought': full_thought,
        }
        self.thought_log.append(record)
        self._count += 1

        # 6. Persist to log file
        if self._log_fh:
            self._log_fh.write(json.dumps(record) + '\n')
            self._log_fh.flush()

        return record

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self, max_thoughts: Optional[int] = None, verbose: bool = True):
        """
        Run the autonomous thought loop.

        Args:
            max_thoughts: Stop after this many thoughts (None = run forever).
            verbose:      Print each thought to stdout.
        """
        self.running = True

        if verbose:
            print("\n" + "=" * 68)
            print("  SelfThink — Autonomous Thought Loop")
            print("  Press Ctrl-C to stop")
            print("=" * 68)
            print(f"  {'#':>4}  {'Emotion':12}  {'Ar':>4}  {'Ft':>4}  Thought")
            print("  " + "-" * 68)

        try:
            while self.running:
                if max_thoughts is not None and self._count >= max_thoughts:
                    break

                record = self.think_once()

                if verbose:
                    thought_preview = record['thought'].replace('\n', '↵')[:45]
                    print(f"  {record['n']:4d}  {record['emotion']:12}  "
                          f"{record['arousal']:4.2f}  {record['fatigue']:4.2f}  "
                          f"{thought_preview!r}")

                # Dream when fatigued
                if self.brain.state.fatigue_level > 0.7:
                    if verbose:
                        print(f"\n  [DREAMING — fatigue={self.brain.state.fatigue_level:.2f}]")
                    self.brain.dream(n_cycles=2, verbose=False)
                    if verbose:
                        print(f"  [AWAKE — fatigue={self.brain.state.fatigue_level:.2f}]\n")

                # Sleep: high arousal = faster thinking
                interval = self.base_interval * (1.5 - record['arousal'])
                time.sleep(max(0.05, interval))

        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            if verbose:
                print(f"\n  Stopped after {self._count} thoughts.")
                self._print_summary()

    def stop(self):
        self.running = False

    def run_in_background(self, max_thoughts: Optional[int] = None):
        """Start the thought loop in a background thread."""
        t = threading.Thread(target=self.run, kwargs={'max_thoughts': max_thoughts,
                                                       'verbose': False}, daemon=True)
        t.start()
        return t

    # ── introspection ─────────────────────────────────────────────────────────

    def _print_summary(self):
        if not self.thought_log:
            return
        from collections import Counter
        emotions = Counter(r['emotion'] for r in self.thought_log)
        avg_ar   = np.mean([r['arousal'] for r in self.thought_log])
        print("\n  Summary")
        print(f"  Total thoughts : {self._count}")
        print(f"  Avg arousal    : {avg_ar:.3f}")
        print(f"  Emotions       : {dict(emotions.most_common(5))}")

    def last_thoughts(self, n: int = 5) -> List[str]:
        return [r['thought'] for r in self.thought_log[-n:]]

    def close(self):
        if self._log_fh:
            self._log_fh.close()


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def run_demo():
    print("=" * 68)
    print("  SelfThink Demo — Brain + Language Model Closed Loop")
    print("=" * 68)

    # ── 1. Build or load Hybrid_LM ────────────────────────────────────────────
    LM_PATH = 'results/hybrid_model.npz'
    if os.path.exists(LM_PATH):
        print(f"\n  Loading Hybrid_LM from {LM_PATH} ...")
        lm = Hybrid_LM.load(LM_PATH)
        print(f"  Loaded. Vocab={lm.V}  Params={lm.param_count():,}")
    else:
        print("\n  No saved model found — training quickly on demo text ...")
        import string

        _CORPUS = """\
The brain is not a computer. It is a prediction machine.
Every perception is a controlled hallucination shaped by past experience.
Consciousness arises from the brain's attempt to minimise surprise.
The world we see is not reality — it is the brain's best guess.
Free energy drives all thought and all action.
Memory is not storage; it is reconstruction from fragments.
Learning is prediction error minimisation applied recursively.
Intelligence models the world and predicts what comes next.
Emotions are the body's signal to the brain about its own state.
Curiosity is the drive to reduce uncertainty about the world.
Fear is the prediction that something bad is about to happen.
Happiness is the signal that predictions are being confirmed.
"""
        tok = CharTokenizer().fit(_CORPUS)
        lm  = Hybrid_LM(tok, embed_dim=32, l1_dim=64, l2_dim=64, l3_dim=64,
                        head_dim=32, lr=0.02, head_lr=0.002, seed=42)
        ids = tok.encode(_CORPUS, add_bos=True, add_eos=False)
        n   = len(ids)
        for epoch in range(20):
            lm.reset_context()
            for i, tid in enumerate(ids):
                nid = ids[i+1] if i+1 < n else -1
                lm.step(tid, nid, learn=True)
        lm.reset_context()
        print(f"  Trained. Vocab={lm.V}  Params={lm.param_count():,}")

    # ── 2. Initialize PreciousBrain (lightweight — consciousness only) ─────────
    print("\n  Initializing PreciousBrain ...")
    config = PreciousBrainConfig(
        name                   = 'SelfThinker',
        input_size             = 8,
        output_size            = 1,
        n_consciousness_neurons = 2000,
        enable_consciousness   = True,
        enable_neurogenesis    = False,
        enable_hippocampus     = False,
        enable_motor_cortex    = False,
        enable_evolution       = False,
        enable_dreaming        = True,
        enable_quantum         = False,
        decision_latency_ms    = 100.0,
    )
    brain = PreciousBrain(config=config)

    # Warm up — give the brain a few random perceptions so it has an emotion
    print("  Warming up brain state ...")
    for _ in range(5):
        brain.feel(np.random.randn(8), {'heart_rate': 75, 'arousal': 0.4})

    # ── 3. Run the autonomous loop ────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    thinker = SelfThink(
        brain          = brain,
        lm             = lm,
        base_interval  = 0.0,   # no sleep in demo, go fast
        max_tokens     = 40,
        temperature    = 0.85,
        top_k          = 8,
        log_file       = 'results/selfthink_log.jsonl',
    )

    thinker.run(max_thoughts=30, verbose=True)

    # ── 4. Show final brain state ─────────────────────────────────────────────
    print("\n  Final brain snapshot:")
    snap = brain.get_consciousness_snapshot()
    print(f"    Thoughts formed : {snap['total_thoughts']}")
    print(f"    Emotions        : {snap['total_emotions']}")
    print(f"    Current emotion : {snap['current_emotion']}")
    print(f"    Arousal         : {brain.state.arousal_level:.3f}")
    print(f"    Fatigue         : {brain.state.fatigue_level:.3f}")

    print("\n  Last 3 thoughts:")
    for t in thinker.last_thoughts(3):
        print(f"    {t!r}")

    thinker.close()
    print(f"\n  Full log saved to results/selfthink_log.jsonl")


if __name__ == '__main__':
    run_demo()
