"""
Personal_AI.py — The Most Neural Personal AI Assistant

Architecture: Brain-first. Every message passes through biological consciousness
BEFORE the LLM generates a response. The LLM receives enriched neural context:
emotion state, relevant memories, plasticity level, free energy, active patterns.

Stack:
  NLP_Bridge (sentence-transformers + TorchPreciousBrain)
    → Neural consciousness (5K neurons, Hebbian LTP/LTD)
    → Emotion detection (8 types, continuous valence/arousal)
    → Episodic memory (encode_memory on novel inputs)
    → Free energy (novelty/surprise signal)
  Ollama LLM (cogito, deepseek, kimi — auto-selects best)
    → Response shaped by neural state
    → Temperature modulated by arousal
    → Tone modulated by emotion
  Persistent memory (JSON — survives sessions)
    → User profile (learns your patterns over time)
    → Conversation history (last N turns)
    → Brain checkpoint (weights saved between sessions)

What makes it personal:
  • Remembers you via hippocampal + consciousness memory
  • Gets better at understanding you via Hebbian plasticity
  • Emotional continuity — tracks your mood across sessions
  • Dream consolidation — strengthens memories when you sleep
  • Novelty detection — notices when you say something unusual
  • Adaptive plasticity — learns faster when you're teaching it
"""

import sys
import os
import json
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# ── LLM backend ───────────────────────────────────────────────────────────────
try:
    import ollama as _ollama
    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False

# ── Brain stack ───────────────────────────────────────────────────────────────
from NLP_Bridge import NLPBridge, TextVectorizer
from Torch_Brain import TorchPreciousBrain, TorchBrainConfig
from A_Brain import PreciousBrainConfig
from Neural_Consciousness import EmotionType


# ──────────────────────────────────────────────────────────────────────────────
# Model preference order (best → fallback)
# ──────────────────────────────────────────────────────────────────────────────
_MODEL_PREFERENCE = [
    # Local models first (no subscription needed)
    "cogito:3b",
    "gemma3:1b",
    "granite4:1b-h",
    "deepseek-coder:6.7b-base-q2_K",
    "llama3:latest",
    "mistral:latest",
    # Cloud models (require ollama subscription)
    "kimi-k2:1t-cloud",
    "deepseek-v3.1:671b-cloud",
    "cogito-2.1:671b-cloud",
    "gpt-oss:120b-cloud",
]


def _best_ollama_model() -> Optional[str]:
    if not _HAS_OLLAMA:
        return None
    try:
        available = {m.model for m in _ollama.list().models}
        for pref in _MODEL_PREFERENCE:
            if pref in available:
                return pref
        return next(iter(available), None)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Persistent store
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class UserProfile:
    name:           str = "User"
    sessions:       int = 0
    total_messages: int = 0
    dominant_emotion: str = "neutral"
    avg_arousal:    float = 0.5
    avg_valence:    float = 0.0
    interests:      List[str] = field(default_factory=list)
    emotional_history: List[str] = field(default_factory=list)  # last 20 emotions


@dataclass
class SessionLog:
    timestamp:  str = ""
    turns:      int = 0
    emotions:   List[str] = field(default_factory=list)
    avg_fe:     float = 0.0
    memories_formed: int = 0


class PersistentMemory:
    """JSON-backed store for profile + conversation history across sessions."""

    def __init__(self, path: str = "personal_ai_memory.json"):
        self.path    = Path(path)
        self.profile = UserProfile()
        self.history: deque = deque(maxlen=30)   # last 30 turns
        self.sessions: List[SessionLog] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                p = d.get('profile', {})
                self.profile = UserProfile(**{k: v for k, v in p.items()
                                              if k in UserProfile.__dataclass_fields__})
                self.history = deque(d.get('history', []), maxlen=30)
                self.sessions = [SessionLog(**s) for s in d.get('sessions', [])]
            except Exception:
                pass

    def save(self):
        d = {
            'profile':  asdict(self.profile),
            'history':  list(self.history),
            'sessions': [asdict(s) for s in self.sessions[-10:]],
        }
        self.path.write_text(json.dumps(d, indent=2))

    def add_turn(self, role: str, content: str, emotion: str = "", fe: float = 0.0):
        self.history.append({
            'role': role, 'content': content,
            'emotion': emotion, 'fe': round(fe, 4),
            'time': datetime.datetime.now().strftime("%H:%M"),
        })

    def update_profile(self, emotion: str, valence: float, arousal: float):
        self.profile.total_messages += 1
        eh = self.profile.emotional_history
        eh.append(emotion)
        if len(eh) > 20:
            self.profile.emotional_history = eh[-20:]
        from collections import Counter
        self.profile.dominant_emotion = Counter(eh).most_common(1)[0][0]
        n = self.profile.total_messages
        self.profile.avg_arousal = (self.profile.avg_arousal * (n-1) + arousal) / n
        self.profile.avg_valence = (self.profile.avg_valence * (n-1) + valence) / n

    def recent_history_text(self, n: int = 8) -> str:
        turns = list(self.history)[-n:]
        lines = []
        for t in turns:
            tag = f"[{t['emotion']}]" if t.get('emotion') else ""
            lines.append(f"{t['role'].capitalize()}{tag}: {t['content'][:200]}")
        return "\n".join(lines) if lines else "(no history yet)"

    def profile_summary(self) -> str:
        p = self.profile
        return (f"Name: {p.name} | Sessions: {p.sessions} | "
                f"Messages: {p.total_messages} | "
                f"Dominant emotion: {p.dominant_emotion} | "
                f"Avg valence: {p.avg_valence:+.2f} | "
                f"Avg arousal: {p.avg_arousal:.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# Ollama LLM wrapper
# ──────────────────────────────────────────────────────────────────────────────
class OllamaLLM:
    def __init__(self, model: Optional[str] = None):
        self.model = model or _best_ollama_model() or "cogito:3b"
        self.available = _HAS_OLLAMA

    def generate(self, system: str, messages: List[Dict],
                 temperature: float = 0.7, max_tokens: int = 512) -> str:
        if not self.available:
            return "[LLM unavailable — install ollama]"
        try:
            msgs = [{"role": "system", "content": system}] + messages
            resp = _ollama.chat(
                model=self.model,
                messages=msgs,
                options={"temperature": temperature, "num_predict": max_tokens},
            )
            return resp.message.content.strip()
        except Exception as e:
            return f"[LLM error: {e}]"


# ──────────────────────────────────────────────────────────────────────────────
# Personal AI
# ──────────────────────────────────────────────────────────────────────────────
class PersonalAI:
    """
    Brain-first personal AI assistant.

    Every user message:
      1. Encoded via sentence-transformers → 64-dim vector
      2. Passed through TorchPreciousBrain (Hebbian + backprop)
      3. Brain state captured: emotion, valence, arousal, FE, memories
      4. Relevant memories retrieved via AttentiveMemoryBank
      5. LLM receives enriched context and generates response
      6. Brain learns from the exchange
      7. Novel inputs (FE ≥ 0.5) encoded to long-term memory
    """

    SYSTEM_TEMPLATE = """\
You are {ai_name}, a deeply personal AI assistant with a biological neural brain.
Your responses are shaped by your current neural state — not just rules.

=== YOUR NEURAL STATE ===
Emotion      : {emotion}  (valence {valence:+.2f}, arousal {arousal:.2f})
Free Energy  : {fe:.3f}  — {fe_desc}
Plasticity   : LTP>{ltp:.3f}  LTD<{ltd:.3f}  ({plasticity_desc})
Active neurons: {neurons} / 5,000
Memories stored: {memories}

=== USER PROFILE ===
{profile}

=== RECENT CONVERSATION ===
{history}

=== BEHAVIORAL GUIDELINES ===
- Emotion FEAR/ANGER    → be calm, careful, de-escalating
- Emotion HAPPINESS/LOVE → be warm, enthusiastic, playful
- Emotion SADNESS       → be empathetic, gentle, supportive
- Emotion SURPRISE      → acknowledge novelty, ask questions
- Emotion NEUTRAL       → be clear, efficient, direct
- High Free Energy (>0.6) → you're encountering something NEW — be curious
- Low Free Energy (<0.2)  → familiar topic — be confident and specific
- High arousal (>0.75)   → user is engaged/stressed — match their energy carefully
- High plasticity (LTP<0.65) → you're in a learning phase — absorb details
- Be natural. Never mention neural state unless user asks.
- Personalize using the profile. Use {user_name}'s name occasionally.
- Keep responses focused. Quality over length.\
"""

    def __init__(
        self,
        ai_name:    str = "Aria",
        user_name:  str = "User",
        model:      Optional[str] = None,
        memory_file: str = "personal_ai_memory.json",
        verbose:    bool = True,
    ):
        self.ai_name   = ai_name
        self.verbose   = verbose

        print(f"\n{'='*60}")
        print(f"  PERSONAL AI — {ai_name}")
        print(f"{'='*60}")

        # ── Persistent memory ─────────────────────────────────────────────────
        self.memory = PersistentMemory(memory_file)
        self.memory.profile.name = user_name
        self.memory.profile.sessions += 1

        # ── Brain stack ───────────────────────────────────────────────────────
        print("  Initializing brain stack...", end="", flush=True)
        cfg = TorchBrainConfig(
            brain_config=PreciousBrainConfig(
                input_size=64, output_size=1,
                n_consciousness_neurons=5000,
            ),
            input_size=64, latent_size=128,
        )
        self.bridge = NLPBridge(config=cfg)
        print(" done")

        # ── LLM ───────────────────────────────────────────────────────────────
        self.llm = OllamaLLM(model)
        print(f"  LLM model   : {self.llm.model}")
        print(f"  Encoder     : {self.bridge.vectorizer._mode}")
        print(f"  Neurons     : 5,000")
        print(f"  Memory file : {memory_file}")
        print(f"  Sessions    : {self.memory.profile.sessions}")
        print(f"{'='*60}\n")

        # ── Session tracking ──────────────────────────────────────────────────
        self._session = SessionLog(
            timestamp=datetime.datetime.now().isoformat()
        )

    # ── core methods ──────────────────────────────────────────────────────────
    def _fe_description(self, fe: float) -> str:
        if fe > 0.8: return "highly novel / surprising"
        if fe > 0.5: return "novel topic"
        if fe > 0.2: return "somewhat familiar"
        return "very familiar territory"

    def _plasticity_description(self, ltp: float) -> str:
        if ltp < 0.62: return "high plasticity — learning mode"
        if ltp < 0.70: return "normal plasticity"
        return "consolidation mode — stable patterns"

    def _emotion_temperature(self, emotion: str, arousal: float) -> float:
        """Map emotion → LLM temperature (creativity vs precision)."""
        base = {
            'happiness': 0.85, 'love': 0.80, 'surprise': 0.75,
            'neutral':   0.65, 'sadness': 0.60, 'fear': 0.55,
            'anger':     0.50, 'disgust': 0.50,
        }.get(emotion, 0.65)
        # High arousal → slightly more creative
        return min(0.95, base + 0.1 * (arousal - 0.5))

    def _build_system_prompt(self, brain_result: Dict) -> str:
        snap = self.bridge.brain.get_state()
        return self.SYSTEM_TEMPLATE.format(
            ai_name    = self.ai_name,
            user_name  = self.memory.profile.name,
            emotion    = brain_result['emotion'],
            valence    = brain_result['valence'],
            arousal    = brain_result['arousal'],
            fe         = brain_result['free_energy'],
            fe_desc    = self._fe_description(brain_result['free_energy']),
            ltp        = brain_result['ltp'],
            ltd        = brain_result['ltd'],
            plasticity_desc = self._plasticity_description(brain_result['ltp']),
            neurons    = brain_result.get('active_neurons', 250),
            memories   = snap.get('memories_stored', 0),
            profile    = self.memory.profile_summary(),
            history    = self.memory.recent_history_text(8),
        )

    def chat(self, user_message: str) -> str:
        """One full exchange: neural processing → LLM response → learning."""

        # 1. Neural processing
        brain_result = self.bridge.process(user_message)

        # 2. Update profile
        self.memory.update_profile(
            brain_result['emotion'],
            brain_result['valence'],
            brain_result['arousal'],
        )

        # 3. Show neural state if verbose
        if self.verbose:
            fe     = brain_result['free_energy']
            fe_bar = '█' * int(fe * 10) + '░' * (10 - int(fe * 10))
            print(f"\n  💭 [{brain_result['emotion']:<9}]  "
                  f"valence {brain_result['valence']:+.2f}  "
                  f"arousal {brain_result['arousal']:.2f}  "
                  f"FE [{fe_bar}] {fe:.3f}  "
                  f"LTP>{brain_result['ltp']:.3f}  "
                  f"mem={'✓' if brain_result.get('memory_encoded') else '·'}")

        # 4. Build enriched LLM context
        system = self._build_system_prompt(brain_result)
        messages = [{"role": "user", "content": user_message}]
        temp = self._emotion_temperature(brain_result['emotion'], brain_result['arousal'])

        # 5. LLM generates response
        response = self.llm.generate(system, messages, temperature=temp)

        # 6. Brain learns from exchange
        self.bridge.learn(user_message)

        # 7. Persist
        self.memory.add_turn("user",      user_message,
                             brain_result['emotion'], brain_result['free_energy'])
        self.memory.add_turn("assistant", response)
        self.memory.save()

        # 8. Session stats
        self._session.turns += 1
        self._session.emotions.append(brain_result['emotion'])
        self._session.avg_fe = (
            (self._session.avg_fe * (self._session.turns - 1) + brain_result['free_energy'])
            / self._session.turns
        )

        return response

    def dream(self):
        """Consolidate session memories — call between sessions."""
        print(f"\n  💤 Dream consolidation...")
        self.bridge.brain.dream(n_cycles=3)
        print(f"  ✓ Memories consolidated. Brain rested.")

    def neural_status(self) -> str:
        """Return formatted brain state summary."""
        snap  = self.bridge.brain.get_state()
        score = self.bridge.preciousness_score()
        return (
            f"Thoughts formed : {snap.get('total_thoughts', 0)}\n"
            f"Memories stored : {snap.get('memories_stored', 0)}\n"
            f"LTP threshold   : {snap.get('ltp_threshold', 0.7):.3f} (dynamic)\n"
            f"LTD threshold   : {snap.get('ltd_threshold', 0.3):.3f} (dynamic)\n"
            f"Avg free energy : {snap.get('avg_free_energy', 0):.4f}\n"
            f"Preciousness    : {score['total']:.1f}/100"
        )

    def end_session(self):
        """Save session log and optionally consolidate."""
        from collections import Counter
        if self._session.turns > 0:
            self._session.memories_formed = self.bridge.brain.get_state().get('memories_stored', 0)
            self.memory.sessions.append(self._session)
            self.memory.save()
        print(f"\n  Session ended — {self._session.turns} turns, "
              f"avg FE {self._session.avg_fe:.3f}")

    # ── interactive loop ──────────────────────────────────────────────────────
    def run(self):
        """Interactive chat loop."""
        name = self.memory.profile.name
        print(f"  Chat with {self.ai_name} (your personal AI)")
        print(f"  Commands: /status  /dream  /score  /models  /model <name>  /quit\n")

        while True:
            try:
                user_input = input(f"  {name}: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            if user_input.lower() in ('/quit', '/exit', 'quit', 'exit', 'bye'):
                self.end_session()
                self.dream()
                print(f"\n  {self.ai_name}: Goodbye {name}! I'll remember our conversation. 🧠\n")
                break

            if user_input.lower() == '/status':
                print(f"\n  Neural Status:\n  {self.neural_status()}\n")
                continue

            if user_input.lower() == '/dream':
                self.dream()
                continue

            if user_input.lower() == '/models':
                try:
                    models = [m.model for m in _ollama.list().models]
                    print(f"\n  Available models:")
                    for i, m in enumerate(models):
                        tag = "  ← current" if m == self.llm.model else ""
                        print(f"    [{i+1}] {m}{tag}")
                    print(f"  Use: /model <name>  e.g. /model kimi-k2:1t-cloud\n")
                except Exception as e:
                    print(f"\n  Could not list models: {e}\n")
                continue

            if user_input.lower().startswith('/model '):
                new_model = user_input[7:].strip()
                if new_model:
                    self.llm.model = new_model
                    print(f"\n  Model switched → {new_model}\n")
                else:
                    print(f"\n  Current model: {self.llm.model}\n")
                continue

            if user_input.lower() == '/score':
                score = self.bridge.preciousness_score()
                print(f"\n  Preciousness: {score['total']:.1f}/100  "
                      f"(prediction {score['prediction']:.1f}  "
                      f"surprise {score['surprise_sense']:.1f}  "
                      f"memory {score['memory']:.1f})\n")
                continue

            response = self.chat(user_input)
            print(f"\n  {self.ai_name}: {response}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Personal AI Assistant")
    parser.add_argument("--name",    default="Aria",   help="AI name")
    parser.add_argument("--user",    default="User",   help="Your name")
    parser.add_argument("--model",   default=None,     help="Ollama model override")
    parser.add_argument("--quiet",   action="store_true", help="Hide neural state")
    parser.add_argument("--memory",  default="personal_ai_memory.json")
    args = parser.parse_args()

    ai = PersonalAI(
        ai_name    = args.name,
        user_name  = args.user,
        model      = args.model,
        memory_file= args.memory,
        verbose    = not args.quiet,
    )
    ai.run()


if __name__ == "__main__":
    main()
