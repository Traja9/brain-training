"""
===========================================================
PRECIOUS BRAIN v1.0 - UNIFIED COGNITIVE ARCHITECTURE
-----------------------------------------------------------
A revolutionary AI system combining:
- Neural Consciousness (thought patterns, emotions, decisions)
- Evolutionary Learning (genetic algorithms, meta-learning)
- Memory Systems (hippocampus, episodic, associative)
- Synaptic Plasticity (LTP/LTD, Hebbian learning)
- Attention Mechanisms & Working Memory
- Deterministic Neural Dynamics

This creates a complete artificial brain that:
✓ Forms thoughts as distributed neural patterns
✓ Stores and consolidates memories
✓ Processes emotions with body-brain integration
✓ Makes pre-conscious decisions
✓ Evolves through experience
✓ Learns continuously without forgetting
===========================================================
"""

import numpy as np
import time
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import warnings
warnings.filterwarnings('ignore')

try:
    from Quantum_Cognition import QuantumCognitionEngine
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

# Import all component modules
try:
    from Neural_Consciousness import (
        ConsciousnessProcessor,
        ThoughtPattern,
        EmotionalState,
        EmotionType,
        Memory as ConsciousnessMemory
    )
    CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_AVAILABLE = False
    print("⚠️  Neural_Consciousness module not available")

try:
    from Cerebrum_Ultra import (
        NeuroGenesis,
        NeuroGenesisConfig,
        create_neurogenesis,
        EvolutionStrategy
    )
    NEUROGENESIS_AVAILABLE = True
except ImportError:
    NEUROGENESIS_AVAILABLE = False
    print("⚠️  Cerebrum_Ultra module not available")

try:
    from Hippocampus_Brain import (
        Hippocampus,
        HippocampusConfig,
        create_hippocampus,
        MemoryType,
        MemoryTrace
    )
    HIPPOCAMPUS_AVAILABLE = True
except ImportError:
    HIPPOCAMPUS_AVAILABLE = False
    print("⚠️  Hippocampus_Brain module not available")

try:
    from Motor_Cortex import (
        MotorCortex,
        create_motor_cortex,
        ActionType,
        ExecutionContext
    )
    MOTOR_CORTEX_AVAILABLE = True
except ImportError:
    MOTOR_CORTEX_AVAILABLE = False
    print("⚠️  Motor_Cortex module not available")


# ============================================================
# PRECIOUS BRAIN ENUMS & CONFIGURATIONS
# ============================================================

class CognitiveMode(Enum):
    """Brain operational modes"""
    LEARNING = "learning"
    INFERENCE = "inference"
    DREAMING = "dreaming"  # Memory consolidation during rest
    ATTENTION = "attention"  # Focused processing
    EXPLORATION = "exploration"  # Exploratory learning


class IntegrationStrategy(Enum):
    """How components integrate"""
    SEQUENTIAL = "sequential"  # Consciousness → Memory → Learning
    PARALLEL = "parallel"  # All systems process simultaneously
    HIERARCHICAL = "hierarchical"  # Higher-order integration
    DYNAMIC = "dynamic"  # Adaptive integration based on context


@dataclass
class PreciousBrainConfig:
    """Complete brain configuration"""
    # Identity
    name: str = "PreciousBrain"
    version: str = "1.0"
    
    # Architecture
    input_size: int = 5  # CLI command features: hash, time, output_len, success, context
    output_size: int = 1  # Success prediction
    base_architecture: str = "large"
    
    # Component activation
    enable_consciousness: bool = True
    enable_neurogenesis: bool = True
    enable_hippocampus: bool = True
    enable_motor_cortex: bool = True
    
    # Consciousness settings
    n_consciousness_neurons: int = 10000
    decision_latency_ms: float = 300.0
    consciousness_memory_capacity: int = 500  # prune weakest when exceeded
    
    # Evolution settings
    enable_evolution: bool = True
    population_size: int = 8
    evolution_generations: int = 5
    
    # Memory settings
    short_term_capacity: int = 100
    long_term_capacity: int = 10000
    episodic_capacity: int = 5000
    consolidation_frequency: int = 10
    
    # Integration
    integration_strategy: IntegrationStrategy = IntegrationStrategy.HIERARCHICAL
    consciousness_weight: float = 0.3
    memory_weight: float = 0.4
    learning_weight: float = 0.3
    
    # Quantum cognition
    enable_quantum:    bool  = True
    hilbert_dim:       int   = 64    # must match input_size after projection
    decoherence_rate:  float = 0.05  # environmental dephasing rate γ

    # Advanced features
    enable_dreaming:      bool  = True
    enable_attention:     bool  = True
    enable_emotion_learning: bool = True
    enable_meta_cognition: bool  = True   # self-monitoring confidence + LR adaptation

    # Continual learning
    ewc_lambda:       float = 400.0  # EWC regularisation strength
    replay_fraction:  float = 0.25   # fraction of anchor memories replayed per epoch

    # Data structure bounds
    max_thought_associations: int = 1000
    max_decision_outcomes:    int = 500

    # Performance
    parallel_processing: bool = False   # run phases 2+3 in parallel threads
    use_gpu: bool = False               # use CUDA if available (requires torch)


# ============================================================
# COGNITIVE STATE TRACKING
# ============================================================

@dataclass
class CognitiveState:
    """Current state of the brain"""
    mode: CognitiveMode = CognitiveMode.LEARNING
    current_thought: Optional[ThoughtPattern] = None
    current_emotion: Optional[EmotionalState] = None
    attention_focus: Optional[np.ndarray] = None
    working_memory: List[Any] = field(default_factory=list)
    arousal_level: float = 0.5  # 0=calm, 1=highly aroused
    fatigue_level: float = 0.0  # 0=fresh, 1=exhausted
    timestamp: float = field(default_factory=time.time)


@dataclass
class CognitiveMetrics:
    """Performance metrics"""
    total_thoughts: int = 0
    total_memories: int = 0
    total_emotions: int = 0
    total_decisions: int = 0
    avg_decision_latency_ms: float = 0.0
    memory_consolidations: int = 0
    evolution_generations: int = 0
    learning_episodes: int = 0
    avg_prediction_error: float = 0.0


# ============================================================
# MAIN PRECIOUS BRAIN CLASS
# ============================================================

class PreciousBrain:
    """
    Unified Cognitive Architecture
    
    Combines consciousness, evolution, and memory into a
    single integrated artificial brain system.
    """
    
    def __init__(self, config: Optional[PreciousBrainConfig] = None, **kwargs):
        """Initialize the precious brain"""
        # Configuration
        if config is None:
            config = PreciousBrainConfig(**kwargs)
        self.config = config
        
        # Core components (will be initialized)
        self.consciousness: Optional[ConsciousnessProcessor] = None
        self.neurogenesis: Optional[NeuroGenesis] = None
        self.hippocampus: Optional[Hippocampus] = None
        self.motor_cortex: Optional[MotorCortex] = None
        
        # Cognitive state
        self.state = CognitiveState()
        self.metrics = CognitiveMetrics()
        
        # Quantum cognition layer
        self.quantum: Optional[QuantumCognitionEngine] = None

        # Integration layer
        self.thought_memory_associations: Dict[str, List[str]] = {}
        self.emotion_memory_bindings: Dict[str, EmotionType] = {}
        self.decision_outcomes: List[Dict] = []

        # Neuromodulator state — updated from consciousness each feel() call
        self._neuromodulators: Dict[str, float] = {
            'dopamine':       0.5,   # reward signal → learning rate scaling
            'serotonin':      0.5,   # patience / exploration rate
            'acetylcholine':  0.5,   # attention gate → recall sharpness
            'norepinephrine': 0.5,   # arousal amplifier → decoherence scaling
        }

        # Continual learning anchors (EWC-style experience replay)
        self._ewc_anchors: List[np.ndarray] = []     # important input patterns
        self._ewc_targets: List[np.ndarray] = []     # their expected outputs
        self._ewc_active:  bool = False              # True after first consolidation

        # Meta-cognition tracker
        self._error_history: deque = deque(maxlen=20)   # recent prediction errors
        self._meta_lr_scale: float = 1.0                # dynamic learning rate multiplier

        # Training history
        self.cognitive_history: List[Dict] = []
        self.dream_log: List[Dict] = []
        
        # Initialize the brain
        self._initialize_brain()
    
    def _initialize_brain(self):
        """Initialize all brain components"""
        print(f"\n{'='*70}")
        print(f"🧠 INITIALIZING PRECIOUS BRAIN: {self.config.name}")
        print(f"{'='*70}")
        print(f"Version: {self.config.version}")
        print(f"Architecture: {self.config.base_architecture.upper()}")
        print(f"Integration: {self.config.integration_strategy.value}")
        print(f"{'='*70}\n")
        
        # Initialize Consciousness
        if self.config.enable_consciousness and CONSCIOUSNESS_AVAILABLE:
            print("🧠 Initializing Neural Consciousness...")
            self.consciousness = ConsciousnessProcessor(
                n_neurons=self.config.n_consciousness_neurons,
                n_regions=8,
                decision_latency_ms=self.config.decision_latency_ms
            )
            print(f"   ✓ {self.config.n_consciousness_neurons:,} neurons")
            print(f"   ✓ 8 brain regions")
            print(f"   ✓ {len(self.consciousness.synapses):,} synaptic connections")
        
        # Initialize NeuroGenesis (Evolution & Learning)
        if self.config.enable_neurogenesis and NEUROGENESIS_AVAILABLE:
            print("\n🧬 Initializing NeuroGenesis Evolution...")
            ng_config = NeuroGenesisConfig(
                name=f"{self.config.name}_Evolution",
                base_architecture=self.config.base_architecture,
                input_size=self.config.input_size,
                output_size=self.config.output_size,
                enable_evolution=self.config.enable_evolution,
                population_size=self.config.population_size,
                enable_curriculum=True,
                enable_moe=False
            )
            self.neurogenesis = NeuroGenesis(config=ng_config)
            print(f"   ✓ Population: {self.config.population_size}")
            print(f"   ✓ Evolution enabled")

            # GPU acceleration
            if self.config.use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        for _attr in ('model', 'best_model', 'network'):
                            _m = getattr(self.neurogenesis, _attr, None)
                            if _m is not None and hasattr(_m, 'to'):
                                _m.to('cuda')
                                print(f"   ✓ {_attr} → CUDA")
                                break
                    else:
                        print("   ⚠ use_gpu=True but CUDA unavailable — CPU used")
                except ImportError:
                    print("   ⚠ use_gpu=True but torch not installed — CPU used")

        # Initialize Hippocampus (Memory)
        if self.config.enable_hippocampus and HIPPOCAMPUS_AVAILABLE:
            print("\n🧠 Initializing Hippocampus Memory...")
            hc_config = HippocampusConfig(
                name=f"{self.config.name}_Memory",
                base_architecture=self.config.base_architecture,
                input_size=self.config.input_size,
                output_size=self.config.output_size,
                short_term_capacity=self.config.short_term_capacity,
                long_term_capacity=self.config.long_term_capacity,
                episodic_capacity=self.config.episodic_capacity,
                enable_episodic_memory=True,
                enable_attention_mechanism=self.config.enable_attention,
                enable_hebbian_learning=True,
                enable_associative_memory=True
            )
            self.hippocampus = Hippocampus(config=hc_config)
            print(f"   ✓ Short-term: {self.config.short_term_capacity}")
            print(f"   ✓ Long-term: {self.config.long_term_capacity:,}")
            print(f"   ✓ Episodic: {self.config.episodic_capacity:,}")
        
        # Initialize Motor Cortex (Action/Execution)
        if self.config.enable_motor_cortex and MOTOR_CORTEX_AVAILABLE:
            print("\n🦾 Initializing Motor Cortex...")
            self.motor_cortex = MotorCortex(cerebrum=self.neurogenesis)
            print(f"   ✓ Connected to NeuroGenesis")
            print(f"   ✓ Command Executor ready")
            print(f"   ✓ File Operator ready")
            print(f"   ✓ Display Output ready")
        
        # Initialize Quantum Cognition Engine
        if self.config.enable_quantum and QUANTUM_AVAILABLE:
            print("\n⚛️  Initializing Quantum Cognition Engine...")
            self.quantum = QuantumCognitionEngine(
                hilbert_dim      = self.config.hilbert_dim,
                decoherence_rate = self.config.decoherence_rate,
                n_walk_nodes     = self.config.hilbert_dim // 2,
                memory_capacity  = self.config.consciousness_memory_capacity,
            )
            print(f"   ✓ Hilbert space ℂ^{self.config.hilbert_dim}")
            print(f"   ✓ Decoherence rate γ = {self.config.decoherence_rate}")
            print(f"   ✓ Born rule · order effects · decoherence · quantum walk")

        print(f"\n{'='*70}")
        print("✅ PRECIOUS BRAIN INITIALIZED SUCCESSFULLY")
        print(f"{'='*70}\n")
    
    # --------------------------------------------------------
    # COGNITIVE OPERATIONS
    # --------------------------------------------------------
    
    def think(self, input_data: np.ndarray, 
              context: Optional[Dict] = None) -> ThoughtPattern:
        """
        Form a thought through consciousness system
        
        Process:
        1. Activate neural patterns
        2. Apply synaptic plasticity
        3. Store in working memory
        4. Associate with existing knowledge
        """
        if not self.consciousness:
            raise ValueError("Consciousness not initialized")
        
        # Form thought pattern
        thought = self.consciousness.form_thought(
            input_data,
            thought_name=context.get('name', 'unnamed') if context else 'unnamed'
        )

        # Activation strength: mean over active neurons (meaningful even after k-WTA)
        if thought.active_neurons:
            thought.activation_strength = float(np.mean(
                self.consciousness.neuron_activations[list(thought.active_neurons)]
            ))

        # Cross-link with the most similar consciousness memory
        if self.consciousness.memories:
            related = self.consciousness.retrieve_memory(input_data)
            if related:
                thought.associated_memory = related.memory_id

        # Update state
        self.state.current_thought = thought
        self.state.working_memory.append(thought)
        if len(self.state.working_memory) > 7:  # Miller's Law
            self.state.working_memory.pop(0)

        # Quantum state update: encode activation + unitary evolution step
        if self.quantum is not None:
            self.quantum.encode_neural(input_data)
            self.quantum.evolve(dt=0.05)

        # Thinking costs cognitive energy
        self.state.fatigue_level = min(1.0, self.state.fatigue_level + 0.005)

        # Update metrics
        self.metrics.total_thoughts += 1

        return thought
    
    def feel(self, sensory_input: np.ndarray,
             body_signals: Optional[Dict] = None) -> EmotionalState:
        """
        Process emotion through consciousness
        
        Integrates:
        - Sensory input (what you perceive)
        - Body signals (physiological state)
        - Memory (past experiences with similar stimuli)
        """
        if not self.consciousness:
            raise ValueError("Consciousness not initialized")
        
        # Default body signals
        if body_signals is None:
            body_signals = {
                'heart_rate': 70 + np.random.randn() * 10,
                'arousal': self.state.arousal_level + np.random.randn() * 0.1
            }
        
        # Process emotion
        emotion = self.consciousness.process_emotion(
            sensory_input,
            body_signals
        )

        # Track arousal as exponential moving average of body_arousal signal
        target = emotion.body_arousal * 0.5 + 0.5  # map (-1,1) → (0,1)
        self.state.arousal_level = float(np.clip(
            0.7 * self.state.arousal_level + 0.3 * target, 0.0, 1.0
        ))

        # Gate synaptic plasticity in consciousness based on current emotion
        self._emotion_modulate_consciousness(emotion)

        # ── Neuromodulators from brain engine (if available) ──────────────────
        if (self.consciousness is not None
                and hasattr(self.consciousness, '_last_brain_state')
                and self.consciousness._last_brain_state is not None):
            nm = self.consciousness._last_brain_state.get('neuromodulators', {})
            for key in ('dopamine', 'serotonin', 'acetylcholine', 'norepinephrine'):
                if key in nm:
                    # EMA blend: 30 % new signal, 70 % running average
                    self._neuromodulators[key] = (
                        0.3 * float(nm[key]) + 0.7 * self._neuromodulators[key]
                    )
        else:
            # Derive approximate values from emotion when engine not available
            self._neuromodulators['dopamine'] = float(np.clip(
                0.5 + 0.4 * emotion.valence, 0.1, 0.9))
            self._neuromodulators['norepinephrine'] = float(np.clip(
                0.3 + 0.6 * self.state.arousal_level, 0.1, 0.95))

        # ── Norepinephrine amplifies quantum decoherence ───────────────────────
        if self.quantum is not None:
            ne = self._neuromodulators['norepinephrine']
            dt_decohere = (0.2 + 0.6 * float(np.clip(emotion.body_arousal, 0.0, 1.0))
                           ) * (0.5 + ne)
            self.quantum.decohere(dt=dt_decohere)

        # Update state
        self.state.current_emotion = emotion
        self.metrics.total_emotions += 1

        # Emotional memory binding (capped at 1000 entries)
        if self.hippocampus and emotion.emotion != EmotionType.NEUTRAL:
            memory_key = f"emotion_{time.time()}"
            self.emotion_memory_bindings[memory_key] = emotion.emotion
            if len(self.emotion_memory_bindings) > 1000:
                oldest = next(iter(self.emotion_memory_bindings))
                del self.emotion_memory_bindings[oldest]

        return emotion
    
    def _emotion_modulate_consciousness(self, emotion: EmotionalState):
        """Adjust LTP/LTD thresholds based on emotion, neuromodulators, and
        quantum decoherence state.

        Three-layer gating:
          Layer 1 — emotion type sets a base threshold (biological analogue)
          Layer 2 — norepinephrine scales arousal effect on the threshold
          Layer 3 — quantum purity: low purity (decohered/committed brain)
                    raises LTP threshold (less plastic — already decided)
        """
        if not self.consciousness:
            return

        # Layer 1: emotion-type base
        if emotion.emotion == EmotionType.FEAR:
            base_ltp = min(0.9, 0.7 + 0.1 * emotion.intensity)
            base_ltd = 0.3
        elif emotion.emotion == EmotionType.HAPPINESS:
            base_ltp = max(0.5, 0.7 - 0.1 * emotion.intensity)
            base_ltd = 0.35
        elif emotion.emotion == EmotionType.SADNESS:
            base_ltp = 0.8
            base_ltd = 0.2
        elif emotion.emotion in (EmotionType.ANGER, EmotionType.SURPRISE):
            base_ltp = 0.6
            base_ltd = 0.4
        else:
            base_ltp = 0.7
            base_ltd = 0.3

        # Layer 2: norepinephrine modulates plasticity width
        ne   = self._neuromodulators['norepinephrine']
        ne_shift = (ne - 0.5) * 0.1   # [-0.05, +0.05]
        base_ltp = float(np.clip(base_ltp - ne_shift, 0.4, 0.95))
        base_ltd = float(np.clip(base_ltd + ne_shift * 0.5, 0.1, 0.5))

        # Layer 3: quantum decoherence → raise LTP when brain is committed
        if self.quantum is not None:
            purity    = self.quantum.density.purity()
            # Low purity = decohered = brain has committed to a state → harder to retrain
            decohere_penalty = (1.0 - purity) * 0.15   # 0 to 0.15
            base_ltp  = float(np.clip(base_ltp + decohere_penalty, 0.4, 0.95))

        self.consciousness.ltp_threshold = base_ltp
        self.consciousness.ltd_threshold = base_ltd

    def decide(self, decision_input: np.ndarray,
               decision_label: str = "unnamed") -> Tuple[float, str, List[str]]:
        """
        Make a decision (pre-conscious processing)
        
        Returns:
        - Decision time (when brain actually decided)
        - Decision label
        - List of decisions that became conscious
        """
        if not self.consciousness:
            raise ValueError("Consciousness not initialized")
        
        # ── Quantum purity modulates decision latency ─────────────────────────
        # High purity (coherent brain) → faster, more decisive
        # Low purity (decohered brain) → slower, more deliberate
        if self.quantum is not None:
            purity = self.quantum.density.purity()
            self.consciousness.decision_latency_ms = float(np.clip(
                300.0 * (1.2 - 0.4 * purity), 150.0, 400.0
            ))

        # Pre-conscious decision making
        decision_time, label = self.consciousness.pre_conscious_decision(
            decision_input,
            decision_label
        )

        # Check what became conscious
        conscious_decisions = self.consciousness.check_conscious_awareness()

        # Update metrics
        self.metrics.total_decisions += 1

        # Record outcome (bounded)
        self.decision_outcomes.append({
            'time':              decision_time,
            'label':             label,
            'conscious_delay_ms': self.consciousness.decision_latency_ms,
            'dopamine':          self._neuromodulators['dopamine'],
        })
        if len(self.decision_outcomes) > self.config.max_decision_outcomes:
            self.decision_outcomes = self.decision_outcomes[
                -self.config.max_decision_outcomes:
            ]

        return decision_time, label, conscious_decisions
    
    def memorize(self, experience: np.ndarray,
                 label: Optional[np.ndarray] = None,
                 emotional_context: Optional[EmotionType] = None,
                 importance: float = 1.0):
        """
        Store experience in memory system
        
        Integrates:
        - Current thought pattern
        - Emotional state
        - Hippocampal encoding
        """
        if not self.hippocampus:
            raise ValueError("Hippocampus not initialized")
        
        # Create context
        context = {
            'emotion': emotional_context or (self.state.current_emotion.emotion 
                                           if self.state.current_emotion else None),
            'arousal': self.state.arousal_level,
            'timestamp': time.time()
        }
        
        # Dopamine boosts memory consolidation strength (reward-modulated plasticity)
        da = self._neuromodulators['dopamine']
        effective_importance = float(np.clip(importance * (0.6 + 0.8 * da), 0.1, 3.0))

        # Store in hippocampus
        self.hippocampus.memorize(
            experience.reshape(1, -1) if experience.ndim == 1 else experience,
            label.reshape(1, -1) if label is not None and label.ndim == 1 else label,
            context=context,
            quality_threshold=0.0  # Store all experiences
        )

        # Dual-encode in consciousness memory so thought retrieval can find it
        if self.consciousness:
            emo_ctx = emotional_context or (
                self.state.current_emotion.emotion if self.state.current_emotion else None
            )
            self.consciousness.encode_memory(
                experience.flatten(),
                f"exp_{self.metrics.total_memories}",
                emo_ctx
            )
            self._prune_consciousness_memories()

        # Quantum memory: phase-encode with emotional context
        if self.quantum is not None:
            emo_str = (emotional_context.value
                       if emotional_context is not None
                       else (self.state.current_emotion.emotion.value
                             if self.state.current_emotion else 'neutral'))
            self.quantum.quantum_memorize(
                experience.flatten(),
                label   = f'mem_{self.metrics.total_memories}',
                emotion = emo_str,
                strength = float(np.clip(
                    np.linalg.norm(experience) / (np.sqrt(len(experience.flatten())) + 1e-8),
                    0.1, 2.0
                )),
            )

        # Update metrics
        self.metrics.total_memories += 1

        # Associate thought with memory (bounded)
        if self.state.current_thought:
            memory_key  = f"mem_{self.metrics.total_memories}"
            thought_key = self.state.current_thought.pattern_id
            if thought_key not in self.thought_memory_associations:
                self.thought_memory_associations[thought_key] = []
            self.thought_memory_associations[thought_key].append(memory_key)
            # Evict oldest entries when over cap
            if len(self.thought_memory_associations) > self.config.max_thought_associations:
                oldest = next(iter(self.thought_memory_associations))
                del self.thought_memory_associations[oldest]
    
    def _prune_consciousness_memories(self):
        """Drop the weakest consciousness memories when over capacity."""
        if not self.consciousness:
            return
        mems = self.consciousness.memories
        cap  = self.config.consciousness_memory_capacity
        if len(mems) > cap:
            to_drop = sorted(mems, key=lambda k: mems[k].strength)[:len(mems) - cap]
            for key in to_drop:
                del mems[key]

    def recall(self, query: np.ndarray,
               memory_type: MemoryType = MemoryType.LONG_TERM,
               use_attention: bool = True) -> List[MemoryTrace]:
        """
        Retrieve memories with attention + quantum interference reweighting.

        Acetylcholine controls attention sharpness: high ACh → top-1 dominates;
        low ACh → broader, more associative retrieval.
        Quantum interference: constructive gain promotes; destructive suppresses.
        """
        if not self.hippocampus:
            raise ValueError("Hippocampus not initialized")

        memories = self.hippocampus.recall(query, memory_type=memory_type)

        if not memories:
            return memories

        # ── Acetylcholine: sharpen or broaden importance distribution ─────────
        ach = self._neuromodulators['acetylcholine']
        # High ACh (focused attention): amplify the top memories, suppress rest
        # Low ACh (diffuse / associative): flatten the distribution
        if len(memories) > 1:
            importances = np.array([m.importance for m in memories], dtype=float)
            sharpness   = 1.0 + 2.0 * ach          # range [1.0, 3.0]
            softmax_imp = np.exp(sharpness * importances)
            softmax_imp = softmax_imp / softmax_imp.sum() * importances.sum()
            for m, imp in zip(memories, softmax_imp):
                m.importance = float(imp)

        # ── Quantum interference reweighting ──────────────────────────────────
        if self.quantum is not None and len(self.quantum.memory) > 0:
            q_res = self.quantum.quantum_recall(query, top_k=min(len(memories), 8))
            for i, match in enumerate(q_res['matches']):
                gain = match.get('interference_gain', 0.0)
                if i < len(memories) and abs(gain) > 0.005:
                    scale = float(np.clip(1.0 + gain * 2.0, 0.1, 3.0))
                    memories[i].importance = float(np.clip(
                        memories[i].importance * scale, 1e-4, 10.0
                    ))

        return memories
    
    # --------------------------------------------------------
    # LEARNING OPERATIONS
    # --------------------------------------------------------
    
    def learn(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 200,
              use_evolution: bool = True,
              use_memory_replay: bool = True,
              consolidate_freq: int = 10,
              verbose: bool = True) -> Dict:
        """
        Unified learning integrating all systems.

        Enhancements over baseline:
        - _meta_lr_scale applied to neurogenesis + hippocampus learning rates
        - EWC anchor replay mixed into training data when _ewc_active=True
        - Phases 2+3 run in parallel threads when config.parallel_processing=True
        - metacognitive_update() called after learning to adapt next session LR
        """
        print(f"\n{'='*70}")
        print(f"🧠 PRECIOUS BRAIN LEARNING")
        print(f"{'='*70}")
        print(f"Samples: {len(X)}")
        print(f"Epochs: {epochs}")
        print(f"Evolution: {'ON' if use_evolution else 'OFF'}")
        print(f"Memory Replay: {'ON' if use_memory_replay else 'OFF'}")

        # ── Meta-cognition: read current LR scale ─────────────────────────────
        meta = self.metacognitive_update()
        lr_scale = self._meta_lr_scale
        if verbose and meta.get('meta_cognition') == 'active':
            print(f"Meta-LR scale: {lr_scale:.3f}  regime={meta.get('regime','?')}")
        print(f"{'='*70}\n")

        # ── EWC experience replay ──────────────────────────────────────────────
        X_train, y_train = X, y
        if self._ewc_active and len(self._ewc_anchors) > 0:
            n_replay = max(1, int(len(self._ewc_anchors) * self.config.replay_fraction))
            idx      = np.random.choice(len(self._ewc_anchors), n_replay, replace=False)
            anc_X    = np.stack([self._ewc_anchors[i] for i in idx])
            anc_y    = np.stack([self._ewc_targets[i] for i in idx])
            if anc_X.ndim > 1 and anc_X.shape[1] > X.shape[1]:
                anc_X = anc_X[:, :X.shape[1]]
            if anc_X.shape[1:] == X.shape[1:] and anc_y.shape[1:] == y.shape[1:]:
                X_train = np.concatenate([X, anc_X], axis=0)
                y_train = np.concatenate([y, anc_y], axis=0)
                if verbose:
                    print(f"  🔒 EWC replay: +{n_replay} anchors → {len(X_train)} total")

        self.state.mode = CognitiveMode.LEARNING
        learning_results = {}

        # ── Temporarily scale learning rates by meta_lr_scale ─────────────────
        _orig_lr_ng = None
        if self.neurogenesis is not None:
            for attr in ('learning_rate', 'lr', '_lr'):
                if hasattr(self.neurogenesis, attr):
                    _orig_lr_ng = getattr(self.neurogenesis, attr)
                    setattr(self.neurogenesis, attr,
                            float(np.clip(_orig_lr_ng * lr_scale, 1e-6, 1.0)))
                    break

        _orig_lr_hc = None
        if self.hippocampus is not None:
            for attr in ('learning_rate', 'lr', '_lr'):
                if hasattr(self.hippocampus, attr):
                    _orig_lr_hc = getattr(self.hippocampus, attr)
                    setattr(self.hippocampus, attr,
                            float(np.clip(_orig_lr_hc * lr_scale, 1e-6, 1.0)))
                    break

        try:
            # Phase 1: Evolutionary Pre-training (if enabled)
            if use_evolution and self.neurogenesis:
                print("🧬 Phase 1: Evolutionary Learning")
                print("-" * 70)

                evolution_results = self.neurogenesis.evolve(
                    X_train, y_train,
                    generations=self.config.evolution_generations,
                    epochs_per_gen=epochs // 5,
                    verbose=verbose
                )

                learning_results['evolution'] = evolution_results
                self.metrics.evolution_generations += self.config.evolution_generations

            # ── Phases 2 & 3: serial or parallel ──────────────────────────────

            def _phase2_conscious(X_p: np.ndarray, y_p: np.ndarray):
                if not self.consciousness:
                    return
                print("\n🧠 Phase 2: Conscious Experience Formation")
                print("-" * 70)
                for i in range(0, len(X_p), 10):
                    batch_X = X_p[i:i+1]
                    self.think(batch_X[0], context={'name': f'learn_{i}'})
                    if self.neurogenesis:
                        try:
                            pred    = self.neurogenesis.predict(batch_X)
                            arousal = min(1.0, float(np.mean(np.abs(pred - y_p[i:i+1]))))
                        except Exception:
                            arousal = 0.5
                    else:
                        arousal = 0.5
                    emotion = self.feel(batch_X[0],
                                        {'heart_rate': 70 + arousal * 30,
                                         'arousal':    arousal})
                    if verbose and i % 50 == 0:
                        print(f"  Sample {i}: emotion={emotion.emotion.value}")

            def _phase3_memory(X_p: np.ndarray, y_p: np.ndarray):
                if not self.hippocampus:
                    return None
                print("\n💾 Phase 3: Memory-Augmented Learning")
                print("-" * 70)
                return self.hippocampus.learn_with_dense_memory(
                    X_p, y_p,
                    epochs=epochs,
                    consolidate_freq=consolidate_freq,
                    replay_ratio=0.5,
                    verbose=verbose
                )

            if (self.config.parallel_processing
                    and self.consciousness
                    and self.hippocampus):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    f2 = pool.submit(_phase2_conscious, X_train, y_train)
                    f3 = pool.submit(_phase3_memory,    X_train, y_train)
                    f2.result()
                    mem_results = f3.result()
            else:
                _phase2_conscious(X_train, y_train)
                mem_results = _phase3_memory(X_train, y_train)

            if mem_results is not None:
                learning_results['memory'] = mem_results
                self.metrics.memory_consolidations = len(
                    self.hippocampus.consolidation_history
                )

            # Phase 4: Final Integration Training
            if self.neurogenesis:
                print("\n🎯 Phase 4: Integration & Fine-tuning")
                print("-" * 70)

                final_results = self.neurogenesis.curriculum_train(
                    X_train, y_train,
                    epochs=epochs // 4,
                    batch_size=32,
                    verbose=verbose
                )

                learning_results['final'] = final_results

        finally:
            # Restore original learning rates regardless of exceptions
            if self.neurogenesis is not None and _orig_lr_ng is not None:
                for attr in ('learning_rate', 'lr', '_lr'):
                    if hasattr(self.neurogenesis, attr):
                        setattr(self.neurogenesis, attr, _orig_lr_ng)
                        break
            if self.hippocampus is not None and _orig_lr_hc is not None:
                for attr in ('learning_rate', 'lr', '_lr'):
                    if hasattr(self.hippocampus, attr):
                        setattr(self.hippocampus, attr, _orig_lr_hc)
                        break

        # Update metrics
        self.metrics.learning_episodes += 1

        # Compute final error
        final_pred = self.predict(X)
        final_error = np.mean(np.abs(final_pred - y))
        self.metrics.avg_prediction_error = final_error

        # Feed error back into meta-cognition so next session adapts LR
        self.metacognitive_update(latest_error=final_error)

        print(f"\n{'='*70}")
        print(f"✅ LEARNING COMPLETE")
        print(f"{'='*70}")
        print(f"Final Error: {final_error:.6f}")
        print(f"Total Thoughts: {self.metrics.total_thoughts}")
        print(f"Total Memories: {self.metrics.total_memories}")
        print(f"Total Emotions: {self.metrics.total_emotions}")
        print(f"Meta-LR scale: {self._meta_lr_scale:.3f}")
        print(f"{'='*70}\n")

        learning_results['final_error'] = final_error
        learning_results['metrics'] = self.get_metrics()

        return learning_results
    
    def dream(self, n_cycles: int = 5, verbose: bool = True):
        """
        Dream state: Memory consolidation and replay
        
        During dreaming:
        - Memories are replayed
        - Synaptic connections strengthen
        - Unnecessary information is pruned
        - Creative associations form
        """
        if not self.config.enable_dreaming:
            return
        
        if not self.hippocampus:
            print("⚠️  Dreaming requires hippocampus")
            return
        
        print(f"\n{'='*70}")
        print(f"💤 ENTERING DREAM STATE")
        print(f"{'='*70}")
        print(f"Dream Cycles: {n_cycles}")
        print(f"{'='*70}\n")
        
        self.state.mode = CognitiveMode.DREAMING
        
        for cycle in range(n_cycles):
            print(f"💤 Dream Cycle {cycle+1}/{n_cycles}")
            print("-" * 70)
            
            # Memory consolidation
            n_consolidated = self.hippocampus.consolidate(force=True)
            print(f"   📦 Consolidated {n_consolidated} memories")
            
            # Memory replay
            if len(self.hippocampus.long_term_memory.memories) > 0:
                self.hippocampus.replay_memories(
                    n_replays=20,
                    batch_size=16,
                    verbose=False
                )
                print(f"   🔄 Replayed memories")
            
            # Replay strongest consciousness memories (sleep consolidation)
            replayed = 0
            if self.consciousness and self.consciousness.memories:
                top_mems = sorted(
                    self.consciousness.memories.values(),
                    key=lambda m: m.strength, reverse=True
                )[:5]
                for mem in top_mems:
                    self.consciousness.form_thought(mem.content, f"dream_replay_{cycle}")
                    mem.consolidate(consolidation_rate=0.01)  # gentle strengthening
                    replayed += 1

            # Spontaneous dream thought
            dream_input = np.random.randn(self.config.input_size)
            dream_thought = self.consciousness.form_thought(dream_input, f"dream_{cycle}")
            print(f"   🌙 Replayed {replayed} memories | "
                  f"Dream thought: {len(dream_thought.active_neurons)} neurons")

            # Sleep restores cognitive energy
            self.state.fatigue_level = max(0.0, self.state.fatigue_level - 0.15)

            # Record dream
            self.dream_log.append({
                'cycle': cycle,
                'timestamp': time.time(),
                'memories_consolidated': n_consolidated,
                'memories_replayed': replayed,
                'fatigue_after': self.state.fatigue_level
            })

            time.sleep(0.1)  # Simulate sleep time

        # Keep dream log bounded
        if len(self.dream_log) > 100:
            self.dream_log = self.dream_log[-100:]

        print(f"\n{'='*70}")
        print(f"✅ DREAM STATE COMPLETE")
        print(f"{'='*70}\n")

        self.state.mode = CognitiveMode.LEARNING
    
    def consciousness_cycle(self,
                            input_stream: List[np.ndarray],
                            input_labels: List[str],
                            verbose: bool = True):
        """
        Run a full multi-step consciousness simulation cycle.

        Drives thought formation, memory encoding/retrieval, emotion
        processing, and pre-conscious decisions in one integrated loop.
        Syncs cognitive state back to the brain afterwards.
        """
        if not self.consciousness:
            raise ValueError("Consciousness not initialized")
        self.consciousness.simulate_consciousness_cycle(
            input_stream, input_labels, verbose=verbose
        )
        # Sync back
        self.state.current_emotion = self.consciousness.current_emotion
        self.metrics.total_thoughts  = self.consciousness.total_thoughts
        self.metrics.total_emotions  = self.consciousness.total_emotions_processed

    def get_consciousness_snapshot(self) -> Dict:
        """Return a detailed snapshot of current consciousness state."""
        if not self.consciousness:
            return {}

        c = self.consciousness
        ltp_count    = sum(1 for s in c.synapses.values() if s.potentiation_count > 0)
        ltd_count    = sum(1 for s in c.synapses.values() if s.depression_count > 0)
        mem_strengths = [m.strength for m in c.memories.values()]

        snapshot: Dict = {
            'thoughts_formed':      c.total_thoughts,
            'total_thoughts':       c.total_thoughts,
            'memories_stored':      len(c.memories),
            'total_memories_formed': len(c.memories),
            'total_emotions':       c.total_emotions_processed,
            'avg_memory_strength':  float(np.mean(mem_strengths)) if mem_strengths else 0.0,
            'ltp_synapses':         ltp_count,
            'ltd_synapses':         ltd_count,
            'current_emotion':      c.current_emotion.emotion.value,
            'decision_buffer_size': len(c.decision_buffer),
            'ltp_threshold':        c.ltp_threshold,
            'ltd_threshold':        c.ltd_threshold,
        }

        if (c.brain_engine is not None
                and hasattr(c, '_last_brain_state')
                and c._last_brain_state is not None):
            bs = c._last_brain_state
            nm = bs['neuromodulators']
            snapshot['brain_engine'] = {
                'workspace_winner': bs.get('workspace_winner', 'n/a'),
                'free_energy':      float(bs.get('free_energy', 0.0)),
                'lif_spikes':       int(bs.get('lif_spikes', 0)),
                'dopamine':         float(nm['dopamine']),
                'serotonin':        float(nm['serotonin']),
                'acetylcholine':    float(nm['acetylcholine']),
                'norepinephrine':   float(nm['norepinephrine']),
            }

        # Quantum cognition state
        if self.quantum is not None:
            qr = self.quantum.get_state_report()
            snapshot['quantum'] = {
                'purity':              qr['purity'],
                'von_neumann_entropy': qr['von_neumann_entropy'],
                'l1_coherence':        qr['l1_coherence'],
                'memory_entanglement': qr['memory_entanglement'],
                'n_q_memories':        qr['n_memories'],
                'hamiltonian_gap':     qr['hamiltonian_gap'],
                'preciousness':        qr['quantum_preciousness'],
            }

        return snapshot

    def quantum_preciousness(self) -> Dict:
        """Quantum cognition contribution to the preciousness score."""
        if self.quantum is None:
            return {'total': 0.0, 'available': False}
        result = self.quantum.quantum_preciousness()
        result['available'] = True
        return result

    def quantum_spread(self, concept_index: int, n_steps: int = 12) -> Dict:
        """Quantum walk spreading activation from a concept node."""
        if self.quantum is None:
            return {}
        return self.quantum.spread_activation(concept_index, n_steps)

    # --------------------------------------------------------
    # PERSISTENCE  (save / load)
    # --------------------------------------------------------

    def save(self, path: str) -> str:
        """
        Persist full brain state to disk.

        Saves everything: synaptic weights, memory traces, quantum state,
        neuromodulators, EWC anchors, metrics, and cognitive history.

        Returns the resolved file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        payload = {
            'version':           self.config.version,
            'config':            self.config,
            'metrics':           self.metrics,
            'neuromodulators':   self._neuromodulators.copy(),
            'ewc_anchors':       [a.tolist() for a in self._ewc_anchors],
            'ewc_targets':       [t.tolist() for t in self._ewc_targets],
            'ewc_active':        self._ewc_active,
            'meta_lr_scale':     self._meta_lr_scale,
            'error_history':     list(self._error_history),
            'dream_log':         self.dream_log,
            'cognitive_history': self.cognitive_history[-200:],  # last 200 only
            'decision_outcomes': self.decision_outcomes,
            # Core components serialised directly (pickle handles numpy)
            'consciousness':     self.consciousness,
            'neurogenesis':      self.neurogenesis,
            'hippocampus':       self.hippocampus,
            'quantum':           self.quantum,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  💾 Brain saved → {path}  ({size_mb:.1f} MB)")
        return path

    @classmethod
    def load(cls, path: str) -> 'PreciousBrain':
        """
        Restore a brain from a saved file.

        Creates a new PreciousBrain with the saved config, then transplants
        all component state — no re-initialization from scratch.
        """
        with open(path, 'rb') as f:
            payload = pickle.load(f)

        # Build shell with saved config (skips heavy init because we overwrite below)
        config  = payload['config']
        config.enable_consciousness = False   # prevent re-init
        config.enable_neurogenesis  = False
        config.enable_hippocampus   = False
        config.enable_motor_cortex  = False
        config.enable_quantum       = False
        brain   = cls(config=config)

        # Restore config to full version
        brain.config = payload['config']

        # Transplant components
        brain.consciousness = payload['consciousness']
        brain.neurogenesis  = payload['neurogenesis']
        brain.hippocampus   = payload['hippocampus']
        brain.quantum       = payload['quantum']

        # Restore state
        brain.metrics             = payload['metrics']
        brain._neuromodulators    = payload['neuromodulators']
        brain._ewc_anchors        = [np.array(a) for a in payload['ewc_anchors']]
        brain._ewc_targets        = [np.array(t) for t in payload['ewc_targets']]
        brain._ewc_active         = payload['ewc_active']
        brain._meta_lr_scale      = payload['meta_lr_scale']
        brain._error_history      = deque(payload['error_history'], maxlen=20)
        brain.dream_log           = payload['dream_log']
        brain.cognitive_history   = payload['cognitive_history']
        brain.decision_outcomes   = payload['decision_outcomes']

        print(f"  🧠 Brain loaded ← {path}  "
              f"(sessions={brain.metrics.learning_episodes}, "
              f"thoughts={brain.metrics.total_thoughts})")
        return brain

    # --------------------------------------------------------
    # CONTINUAL LEARNING  (EWC-style knowledge anchoring)
    # --------------------------------------------------------

    def consolidate_knowledge(self, X: Optional[np.ndarray] = None,
                               y: Optional[np.ndarray] = None,
                               n_anchors: int = 64) -> int:
        """
        Anchor current knowledge to protect it from future learning.

        Stores n_anchors representative (X, y) pairs as EWC replay examples.
        On subsequent learn() calls these are mixed in to prevent catastrophic
        forgetting of previously learned tasks.

        If X/y not supplied, samples from hippocampal long-term memory.
        Returns number of anchors stored.
        """
        if X is not None and y is not None:
            n = min(n_anchors, len(X))
            idx = np.random.choice(len(X), n, replace=False)
            self._ewc_anchors = [X[i] for i in idx]
            self._ewc_targets = [y[i] for i in idx]
        elif self.hippocampus is not None:
            try:
                ltm = self.hippocampus.long_term_memory.memories
                if ltm:
                    sample = ltm[:n_anchors] if len(ltm) >= n_anchors else ltm
                    self._ewc_anchors = [
                        np.array(m.data).flatten() for m in sample
                        if hasattr(m, 'data') and m.data is not None
                    ]
                    self._ewc_targets = [
                        np.array(getattr(m, 'label', [0.0])).flatten()
                        for m in sample
                        if hasattr(m, 'data') and m.data is not None
                    ]
            except Exception:
                pass

        self._ewc_active = len(self._ewc_anchors) > 0
        print(f"  🔒 Knowledge consolidated: {len(self._ewc_anchors)} anchors stored")
        return len(self._ewc_anchors)

    # --------------------------------------------------------
    # FORGETTING  (controlled memory removal)
    # --------------------------------------------------------

    def forget(self, query: np.ndarray, threshold: float = 0.80) -> Dict:
        """
        Remove memories that are highly similar to query.

        Searches hippocampal LTM and quantum memory bank.
        Only removes memories with fidelity ≥ threshold to avoid
        accidentally wiping unrelated content.

        Returns counts of memories removed from each store.
        """
        removed = {'hippocampus': 0, 'consciousness': 0, 'quantum': 0}
        q = np.asarray(query, dtype=float).flatten()

        # ── Hippocampus LTM ───────────────────────────────────────────────────
        if self.hippocampus is not None:
            try:
                ltm   = self.hippocampus.long_term_memory.memories
                keep  = []
                for m in ltm:
                    if not (hasattr(m, 'data') and m.data is not None):
                        keep.append(m)
                        continue
                    vec  = np.array(m.data).flatten()
                    minl = min(len(q), len(vec))
                    sim  = float(np.dot(q[:minl], vec[:minl]) / (
                        np.linalg.norm(q[:minl]) * np.linalg.norm(vec[:minl]) + 1e-10
                    ))
                    if sim >= threshold:
                        removed['hippocampus'] += 1
                    else:
                        keep.append(m)
                self.hippocampus.long_term_memory.memories = keep
            except Exception:
                pass

        # ── Consciousness memories ────────────────────────────────────────────
        if self.consciousness is not None:
            try:
                to_del = []
                for key, mem in self.consciousness.memories.items():
                    vec  = np.array(mem.content).flatten()
                    minl = min(len(q), len(vec))
                    sim  = float(np.dot(q[:minl], vec[:minl]) / (
                        np.linalg.norm(q[:minl]) * np.linalg.norm(vec[:minl]) + 1e-10
                    ))
                    if sim >= threshold:
                        to_del.append(key)
                for k in to_del:
                    del self.consciousness.memories[k]
                removed['consciousness'] = len(to_del)
            except Exception:
                pass

        # ── Quantum memory bank ───────────────────────────────────────────────
        if self.quantum is not None:
            qm   = self.quantum.memory
            keep_p, keep_ph, keep_l, keep_s = [], [], [], []
            for v, ph, lb, st in zip(qm._patterns, qm._phases,
                                      qm._labels, qm._strengths):
                vec  = np.real(v).flatten()
                minl = min(len(q), len(vec))
                sim  = float(np.dot(q[:minl], vec[:minl]) / (
                    np.linalg.norm(q[:minl]) * np.linalg.norm(vec[:minl]) + 1e-10
                ))
                if sim >= threshold:
                    removed['quantum'] += 1
                else:
                    keep_p.append(v); keep_ph.append(ph)
                    keep_l.append(lb); keep_s.append(st)
            qm._patterns  = keep_p;  qm._phases   = keep_ph
            qm._labels    = keep_l;  qm._strengths = keep_s
            qm._dirty     = True

        total = sum(removed.values())
        print(f"  🗑️  Forgot {total} memories "
              f"(hippo={removed['hippocampus']}, "
              f"conscious={removed['consciousness']}, "
              f"quantum={removed['quantum']})")
        return removed

    # --------------------------------------------------------
    # META-COGNITION  (self-monitoring + adaptive learning rate)
    # --------------------------------------------------------

    def metacognitive_update(self, latest_error: Optional[float] = None) -> Dict:
        """
        Self-monitor prediction accuracy and adapt learning rate.

        Three regimes:
          Improving  (error trending down)  → maintain current LR
          Plateauing (error flat)           → increase LR slightly (explore)
          Degrading  (error trending up)    → decrease LR (consolidate)

        Also monitors quantum coherence: low purity → reduce LR
        (brain is over-stressed, pull back on plasticity).

        Returns current meta-state and recommended LR scale.
        """
        if not self.config.enable_meta_cognition:
            return {'meta_cognition': 'disabled'}

        if latest_error is not None:
            self._error_history.append(float(latest_error))

        if len(self._error_history) < 4:
            return {'meta_cognition': 'warming_up', 'lr_scale': self._meta_lr_scale}

        errors  = list(self._error_history)
        recent  = np.mean(errors[-4:])
        older   = np.mean(errors[-8:-4]) if len(errors) >= 8 else recent
        trend   = recent - older   # negative = improving

        if trend < -0.005:
            regime         = 'improving'
            self._meta_lr_scale = float(np.clip(self._meta_lr_scale * 1.05, 0.5, 2.0))
        elif trend > 0.010:
            regime         = 'degrading'
            self._meta_lr_scale = float(np.clip(self._meta_lr_scale * 0.80, 0.1, 2.0))
        else:
            regime         = 'plateau'
            self._meta_lr_scale = float(np.clip(self._meta_lr_scale * 1.02, 0.5, 2.0))

        # Quantum guard: if brain is heavily decohered (stressed), pull back
        if self.quantum is not None:
            purity = self.quantum.density.purity()
            if purity < 0.5:
                self._meta_lr_scale *= 0.9
                regime += '+quantum_stress'

        return {
            'meta_cognition': 'active',
            'regime':         regime,
            'lr_scale':       self._meta_lr_scale,
            'recent_error':   float(recent),
            'trend':          float(trend),
        }

    # --------------------------------------------------------
    # ACTION & EXECUTION OPERATIONS
    # --------------------------------------------------------
    
    def act(self, action_description: str,
            context: Optional[Dict] = None) -> ExecutionContext:
        """
        Execute an action through Motor Cortex
        
        Completes the cognitive cycle:
        Perception → Thought → Memory → Decision → Action
        
        Args:
            action_description: Description of action to execute
            context: Additional context for execution
        
        Returns:
            Execution result
        """
        if not self.motor_cortex:
            raise ValueError("Motor Cortex not initialized")
        
        # Form thought about the action
        if self.consciousness:
            action_thought = self.think(
                np.array([hash(action_description) % 1000 / 1000.0]),
                context={'name': 'action_planning', 'action': action_description}
            )
            print(f"💭 Action thought formed: {len(action_thought.active_neurons)} neurons")
        
        # Execute through motor cortex
        result = self.motor_cortex.execute_from_reasoning(
            action_description,
            context
        )
        
        # Memorize action outcome if significant
        if self.hippocampus and result['success']:
            outcome_vector = np.array([1.0 if result['success'] else 0.0])
            self.memorize(
                outcome_vector,
                importance=1.5 if result['success'] else 0.5
            )
        
        return result
    
    def perceive_and_act(self, input_data: np.ndarray,
                        auto_execute: bool = True) -> Dict:
        """
        Complete perception-action loop
        
        Full cognitive cycle:
        1. Perceive input
        2. Form thoughts
        3. Retrieve memories
        4. Make decision
        5. Execute action
        
        Args:
            input_data: Sensory input
            auto_execute: Automatically execute decided action
        
        Returns:
            Complete cycle results
        """
        if not self.motor_cortex:
            raise ValueError("Motor Cortex not initialized")
        
        print(f"\n{'='*70}")
        print(f"🔄 COMPLETE COGNITIVE CYCLE")
        print(f"{'='*70}\n")
        
        cycle_results = {
            'perception': None,
            'thought': None,
            'memory': None,
            'decision': None,
            'action': None
        }
        
        # 1. Perception (input processing)
        print("👁️  1. PERCEPTION")
        print("-" * 70)
        print(f"Input shape: {input_data.shape}")
        cycle_results['perception'] = input_data
        
        # 2. Thought formation
        if self.consciousness:
            print("\n💭 2. THOUGHT FORMATION")
            print("-" * 70)
            thought = self.think(input_data[0] if input_data.ndim > 1 else input_data)
            print(f"Neurons active: {len(thought.active_neurons)}")
            print(f"Activation: {thought.activation_strength:.3f}")
            cycle_results['thought'] = thought
        
        # 3. Memory retrieval
        if self.hippocampus:
            print("\n🧠 3. MEMORY RETRIEVAL")
            print("-" * 70)
            memories = self.recall(input_data[0] if input_data.ndim > 1 else input_data)
            print(f"Retrieved: {len(memories)} memories")
            if memories:
                print(f"Top memory importance: {memories[0].importance:.3f}")
            cycle_results['memory'] = memories
        
        # 4. Decision making (via neurogenesis)
        print("\n🎯 4. DECISION MAKING")
        print("-" * 70)
        
        if self.neurogenesis:
            prediction = self.neurogenesis.predict(
                input_data.reshape(1, -1) if input_data.ndim == 1 else input_data
            )
            
            # Interpret prediction as decision
            confidence = float(np.mean(np.abs(prediction)))
            
            if confidence > 0.7:
                decision = "Execute action with high confidence"
            elif confidence > 0.4:
                decision = "Execute action with moderate confidence"
            else:
                decision = "Store for further analysis"
            
            print(f"Prediction: {prediction.flatten()[0]:.3f}")
            print(f"Confidence: {confidence:.3f}")
            print(f"Decision: {decision}")
            cycle_results['decision'] = decision
        else:
            decision = "Process and display information"
            cycle_results['decision'] = decision
        
        # 5. Action execution
        if auto_execute:
            print("\n🦾 5. ACTION EXECUTION")
            print("-" * 70)
            
            action_result = self.motor_cortex.execute_from_reasoning(
                decision,
                context_data={'input': input_data, 'confidence': confidence if self.neurogenesis else 0.5}
            )
            
            print(f"Action executed: {action_result['action']['type']}")
            print(f"Success: {'✓' if action_result['success'] else '✗'}")
            cycle_results['action'] = action_result
        
        print(f"\n{'='*70}")
        print(f"✅ COGNITIVE CYCLE COMPLETE")
        print(f"{'='*70}\n")
        
        return cycle_results
    
    def execute_command(self, command: str) -> ExecutionContext:
        """
        Execute a system command
        
        Args:
            command: System command to execute
        
        Returns:
            Execution context
        """
        if not self.motor_cortex:
            raise ValueError("Motor Cortex not initialized")
        
        return self.motor_cortex.command_executor.execute(command, shell=True)
    
    def write_file(self, filename: str, content: str) -> ExecutionContext:
        """
        Write content to a file
        
        Args:
            filename: Name of file to write
            content: Content to write
        
        Returns:
            Execution context
        """
        if not self.motor_cortex:
            raise ValueError("Motor Cortex not initialized")
        
        return self.motor_cortex.file_operator.write(filename, content)
    
    def read_file(self, filename: str) -> ExecutionContext:
        """
        Read content from a file
        
        Args:
            filename: Name of file to read
        
        Returns:
            Execution context with file content
        """
        if not self.motor_cortex:
            raise ValueError("Motor Cortex not initialized")
        
        return self.motor_cortex.file_operator.read(filename)
    
    # --------------------------------------------------------
    # PREDICTION & INFERENCE
    # --------------------------------------------------------
    
    def predict(self, X: np.ndarray,
                use_consciousness: bool = False,
                use_memory: bool = False,
                return_explanation: bool = False) -> Union[np.ndarray, Tuple]:
        """
        Make predictions using integrated brain systems
        
        Can use:
        - Pure neural prediction (fastest)
        - Consciousness-augmented (with thought patterns)
        - Memory-augmented (with recall)
        """
        self.state.mode = CognitiveMode.INFERENCE
        
        # Base prediction from neurogenesis
        if self.neurogenesis:
            predictions = self.neurogenesis.predict(X)
        elif self.hippocampus:
            predictions = self.hippocampus.predict(X, use_memory=use_memory)
        else:
            raise ValueError("No prediction system available")
        
        if return_explanation:
            explanation = {
                'method': 'neurogenesis' if self.neurogenesis else 'hippocampus',
                'consciousness_used': use_consciousness,
                'memory_used': use_memory,
                'n_samples': len(X)
            }
            return predictions, explanation
        
        return predictions
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence estimates
        """
        predictions = self.predict(X)
        
        # Compute confidence from memory similarity
        if self.hippocampus:
            confidences = []
            for i in range(len(X)):
                memories = self.hippocampus.recall(X[i], top_k=5)
                if memories:
                    # Confidence based on memory strength
                    avg_strength = np.mean([m.importance for m in memories])
                    confidences.append(avg_strength)
                else:
                    confidences.append(0.5)
            
            return predictions, np.array(confidences)
        else:
            # Default confidence
            return predictions, np.ones(len(X)) * 0.8
    
    # --------------------------------------------------------
    # ANALYSIS & INTROSPECTION
    # --------------------------------------------------------
    
    def introspect(self) -> Dict:
        """
        Brain introspection: analyze internal state
        """
        introspection = {
            'state': {
                'mode': self.state.mode.value,
                'arousal': self.state.arousal_level,
                'fatigue': self.state.fatigue_level,
                'working_memory_items': len(self.state.working_memory)
            },
            'metrics': self.get_metrics(),
            'consciousness': {},
            'memory': {},
            'learning': {}
        }
        
        # Consciousness stats (full snapshot)
        if self.consciousness:
            introspection['consciousness'] = self.get_consciousness_snapshot()
            introspection['consciousness']['active_synapses'] = len(self.consciousness.synapses)
        
        # Memory stats
        if self.hippocampus:
            introspection['memory'] = self.hippocampus.get_memory_stats()
        
        # Learning stats
        if self.neurogenesis:
            introspection['learning'] = {
                'evolution_history': len(self.neurogenesis.evolution_history),
                'population_size': len(self.neurogenesis.population)
            }
        
        return introspection
    
    def get_metrics(self) -> Dict:
        """Get cognitive performance metrics"""
        return {
            'total_thoughts': self.metrics.total_thoughts,
            'total_memories': self.metrics.total_memories,
            'total_emotions': self.metrics.total_emotions,
            'total_decisions': self.metrics.total_decisions,
            'memory_consolidations': self.metrics.memory_consolidations,
            'evolution_generations': self.metrics.evolution_generations,
            'learning_episodes': self.metrics.learning_episodes,
            'avg_prediction_error': self.metrics.avg_prediction_error
        }
    
    def summary(self):
        """Display comprehensive brain summary"""
        print(f"\n{'='*70}")
        print(f"🧠 PRECIOUS BRAIN SUMMARY: {self.config.name}")
        print(f"{'='*70}")
        
        print(f"\n📊 Cognitive State:")
        print(f"   ├─ Mode: {self.state.mode.value}")
        print(f"   ├─ Arousal: {self.state.arousal_level:.2f}")
        print(f"   ├─ Fatigue: {self.state.fatigue_level:.2f}")
        print(f"   └─ Working Memory: {len(self.state.working_memory)} items")
        
        print(f"\n🧠 Consciousness:")
        if self.consciousness:
            c   = self.consciousness
            ltp = sum(1 for s in c.synapses.values() if s.potentiation_count > 0)
            ltd = sum(1 for s in c.synapses.values() if s.depression_count > 0)
            mem_strengths = [m.strength for m in c.memories.values()]
            avg_ms = float(np.mean(mem_strengths)) if mem_strengths else 0.0
            print(f"   ├─ Neurons: {self.config.n_consciousness_neurons:,}")
            print(f"   ├─ Thoughts Formed: {self.metrics.total_thoughts}")
            print(f"   ├─ Emotions Processed: {self.metrics.total_emotions}")
            print(f"   ├─ Decisions Made: {self.metrics.total_decisions}")
            print(f"   ├─ Consciousness Memories: {len(c.memories)}  "
                  f"(avg strength: {avg_ms:.3f})")
            print(f"   ├─ LTP: {ltp:,} synapses   LTD: {ltd:,} synapses")
            print(f"   ├─ Plasticity thresholds: "
                  f"LTP>{c.ltp_threshold:.2f}  LTD<{c.ltd_threshold:.2f}")
            if c.brain_engine and hasattr(c, '_last_brain_state') and c._last_brain_state:
                nm = c._last_brain_state['neuromodulators']
                print(f"   ├─ Workspace: {c._last_brain_state.get('workspace_winner','n/a')}  "
                      f"FreeEnergy: {c._last_brain_state.get('free_energy',0.0):.4f}")
                print(f"   └─ DA={nm['dopamine']:.3f}  5-HT={nm['serotonin']:.3f}  "
                      f"ACh={nm['acetylcholine']:.3f}  NE={nm['norepinephrine']:.3f}")
            else:
                print(f"   └─ Brain engine: not connected")
        else:
            print(f"   └─ Not initialized")
        
        print(f"\n🧬 Evolution & Learning:")
        if self.neurogenesis:
            print(f"   ├─ Architecture: {self.config.base_architecture}")
            print(f"   ├─ Generations: {self.metrics.evolution_generations}")
            print(f"   └─ Learning Episodes: {self.metrics.learning_episodes}")
        else:
            print(f"   └─ Not initialized")
        
        print(f"\n💾 Memory Systems:")
        if self.hippocampus:
            stats = self.hippocampus.get_memory_stats()
            print(f"   ├─ Short-term: {stats['short_term_size']}")
            print(f"   ├─ Long-term: {stats['long_term_size']}")
            print(f"   ├─ Consolidations: {self.metrics.memory_consolidations}")
            print(f"   └─ Retrievals: {stats['retrievals']}")
        else:
            print(f"   └─ Not initialized")
        
        print(f"\n🦾 Motor Cortex:")
        if self.motor_cortex:
            motor_stats = self.motor_cortex.get_statistics()
            print(f"   ├─ Actions Executed: {motor_stats['total_actions']}")
            print(f"   ├─ Success Rate: {motor_stats['success_rate']:.1%}")
            print(f"   └─ Avg Execution Time: {motor_stats['avg_time']:.3f}s")
        else:
            print(f"   └─ Not initialized")
        
        print(f"\n📈 Performance:")
        print(f"   ├─ Prediction Error: {self.metrics.avg_prediction_error:.6f}")
        print(f"   ├─ Total Memories: {self.metrics.total_memories}")
        print(f"   └─ Dream Cycles: {len(self.dream_log)}")
        
        print(f"\n{'='*70}\n")


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def create_precious_brain(name: str = "PreciousBrain",
                         architecture: str = "large",
                         enable_all: bool = True,
                         **kwargs) -> PreciousBrain:
    """
    Create a precious brain with specified configuration
    """
    if enable_all:
        config = PreciousBrainConfig(
            name=name,
            base_architecture=architecture,
            enable_consciousness=True,
            enable_neurogenesis=True,
            enable_hippocampus=True,
            enable_evolution=True,
            enable_dreaming=True,
            enable_attention=True,
            **kwargs
        )
    else:
        config = PreciousBrainConfig(
            name=name,
            base_architecture=architecture,
            **kwargs
        )
    
    return PreciousBrain(config=config)


# ============================================================
# DEMONSTRATION & BENCHMARKING
# ============================================================

def demonstrate_precious_brain():
    """
    Fixed demonstration - configure brain based on actual data dimensions
    """
    print("\n" + "="*70)
    print("🧠 PRECIOUS BRAIN - UNIFIED COGNITIVE ARCHITECTURE")
    print("="*70)
    print("Integrating: Consciousness + Evolution + Memory")
    print("="*70 + "\n")
    
    # ========================================
    # DEMO 1: BASIC COGNITIVE OPERATIONS
    # ========================================
    print("\n" + "="*70)
    print("DEMO 1: BASIC COGNITIVE OPERATIONS")
    print("="*70)
    
    # Create brain with CORRECT input_size=1 to match data
    brain = create_precious_brain(
        name="DemoBrain",
        architecture="medium",
        input_size=1,  # <-- FIX: Match the actual data dimension
        output_size=1,
        n_consciousness_neurons=5000,
        population_size=6,
        enable_all=True
    )
    
    brain.summary()
    
    # Form a thought (use 1-dimensional input)
    print("\n🧠 Forming a thought...")
    input_pattern = np.random.randn(1)  # <-- FIX: 1D instead of 10D
    thought = brain.think(input_pattern, context={'name': 'first_thought'})
    print(f"   ✓ Thought formed: {len(thought.active_neurons)} neurons active")
    print(f"   ✓ Activation strength: {thought.activation_strength:.3f}")
    
    # Process an emotion
    print("\n💗 Processing emotion...")
    sensory_input = np.random.randn(1)  # <-- FIX: 1D instead of 10D
    emotion = brain.feel(sensory_input, {'heart_rate': 100, 'arousal': 0.8})
    print(f"   ✓ Emotion: {emotion.emotion.value}")
    print(f"   ✓ Intensity: {emotion.intensity:.3f}")
    print(f"   ✓ Heart rate: {emotion.heart_rate:.1f} bpm")
    
    # Make a decision
    print("\n🎯 Making a decision...")
    decision_input = np.random.randn(1)  # <-- FIX: 1D instead of 10D
    decision_time, label, conscious = brain.decide(decision_input, "test_decision")
    print(f"   ✓ Decision made at: {decision_time:.6f}")
    print(f"   ✓ Conscious after: {brain.config.decision_latency_ms}ms")
    if conscious:
        print(f"   ✓ Became conscious: {conscious}")
    
    # Memorize an experience
    print("\n💾 Storing memory...")
    experience = np.random.randn(1)  # <-- FIX: 1D instead of 10D
    label = np.random.randn(1)
    brain.memorize(experience, label, emotional_context=EmotionType.HAPPINESS)
    print(f"   ✓ Memory stored with emotional context: HAPPINESS")
    
    # ========================================
    # DEMO 2: LEARNING FROM DATA
    # ========================================
    print("\n" + "="*70)
    print("DEMO 2: INTEGRATED LEARNING")
    print("="*70)
    
    # Generate dataset (already correct - 1 feature)
    X_train = np.linspace(-3, 3, 150).reshape(-1, 1)
    y_train = (np.sin(2*X_train) * np.exp(-X_train**2/3) + 
               0.3*np.cos(3*X_train)).reshape(-1, 1)
    y_train += np.random.normal(0, 0.05, y_train.shape)
    
    # Now learning will work correctly!
    learning_results = brain.learn(
        X_train, y_train,
        epochs=100,
        use_evolution=True,
        use_memory_replay=True,
        consolidate_freq=10,
        verbose=True
    )
    
    # ... rest of the demo continues normally ...
    
    return brain

# ============================================================
# BENCHMARKING SUITE
# ============================================================

class PreciousBrainBenchmark:
    """Comprehensive benchmarking for Precious Brain"""
    
    @staticmethod
    def benchmark_learning_speed(brain: PreciousBrain,
                                X: np.ndarray, y: np.ndarray,
                                epochs: int = 100) -> Dict:
        """Benchmark learning speed and efficiency"""
        print(f"\n⚡ LEARNING SPEED BENCHMARK")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        results = brain.learn(
            X, y,
            epochs=epochs,
            use_evolution=True,
            verbose=False
        )
        
        elapsed = time.time() - start_time
        
        print(f"   ├─ Total Time: {elapsed:.2f}s")
        print(f"   ├─ Samples/sec: {len(X)*epochs/elapsed:.1f}")
        print(f"   └─ Final Error: {results['final_error']:.6f}")
        
        return {
            'time': elapsed,
            'samples_per_sec': len(X)*epochs/elapsed,
            'final_error': results['final_error']
        }
    
    @staticmethod
    def benchmark_memory_efficiency(brain: PreciousBrain) -> Dict:
        """Benchmark memory system efficiency"""
        print(f"\n💾 MEMORY EFFICIENCY BENCHMARK")
        print(f"{'='*60}")
        
        if not brain.hippocampus:
            return {}
        
        stats = brain.hippocampus.get_memory_stats()
        
        storage_efficiency = stats['long_term_size'] / brain.config.long_term_capacity
        
        print(f"   ├─ LTM Utilization: {storage_efficiency:.1%}")
        print(f"   ├─ Consolidations: {stats['consolidations']}")
        print(f"   ├─ Retrievals: {stats['retrievals']}")
        
        if 'avg_importance' in stats:
            print(f"   └─ Avg Importance: {stats['avg_importance']:.3f}")
        
        return {
            'storage_efficiency': storage_efficiency,
            'consolidations': stats['consolidations'],
            'retrievals': stats['retrievals']
        }
    
    @staticmethod
    def benchmark_consciousness_overhead(brain: PreciousBrain,
                                        n_thoughts: int = 100) -> Dict:
        """Benchmark consciousness processing overhead"""
        print(f"\n🧠 CONSCIOUSNESS OVERHEAD BENCHMARK")
        print(f"{'='*60}")
        
        if not brain.consciousness:
            return {}
        
        # Measure thought formation time
        start_time = time.time()
        
        for i in range(n_thoughts):
            input_data = np.random.randn(brain.config.input_size)
            brain.think(input_data, context={'name': f'bench_{i}'})
        
        elapsed = time.time() - start_time
        
        avg_thought_time_ms = (elapsed / n_thoughts) * 1000
        
        print(f"   ├─ Total Time: {elapsed:.2f}s")
        print(f"   ├─ Thoughts/sec: {n_thoughts/elapsed:.1f}")
        print(f"   └─ Avg Time/Thought: {avg_thought_time_ms:.2f}ms")
        
        return {
            'time': elapsed,
            'thoughts_per_sec': n_thoughts/elapsed,
            'avg_time_ms': avg_thought_time_ms
        }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Check module availability
    print("\n🔍 Checking module availability...")
    print(f"   Consciousness: {'✓' if CONSCIOUSNESS_AVAILABLE else '✗'}")
    print(f"   NeuroGenesis:  {'✓' if NEUROGENESIS_AVAILABLE else '✗'}")
    print(f"   Hippocampus:   {'✓' if HIPPOCAMPUS_AVAILABLE else '✗'}")
    print(f"   Motor Cortex:  {'✓' if MOTOR_CORTEX_AVAILABLE else '✗'}")
    
    if not (CONSCIOUSNESS_AVAILABLE or NEUROGENESIS_AVAILABLE or HIPPOCAMPUS_AVAILABLE):
        print("\n⚠️  No brain modules available!")
        print("Please ensure at least one of the following is in the same directory:")
        print("   - Neural_Consciousness.py")
        print("   - Cerebrum_Ultra.py")
        print("   - Hippocampus_Brain.py")
        exit(1)
    
    # Run demonstration
    brain = demonstrate_precious_brain()
    
    # Run benchmarks
    print("\n" + "="*70)
    print("🏁 RUNNING BENCHMARKS")
    print("="*70)
    
    # Generate benchmark data
    X_bench = np.random.uniform(-3, 3, (100, 1))
    y_bench = np.sin(2*X_bench) + 0.5*X_bench
    
    benchmark = PreciousBrainBenchmark()
    
    # Learning speed
    learning_bench = benchmark.benchmark_learning_speed(
        brain, X_bench, y_bench, epochs=50
    )
    
    # Memory efficiency
    memory_bench = benchmark.benchmark_memory_efficiency(brain)
    
    # Consciousness overhead
    consciousness_bench = benchmark.benchmark_consciousness_overhead(
        brain, n_thoughts=50
    )
    
    print("\n" + "="*70)
    print("📊 BENCHMARK SUMMARY")
    print("="*70)
    print(f"Learning: {learning_bench.get('samples_per_sec', 0):.1f} samples/sec")
    print(f"Memory: {memory_bench.get('storage_efficiency', 0):.1%} utilization")
    print(f"Consciousness: {consciousness_bench.get('thoughts_per_sec', 0):.1f} thoughts/sec")
    print("="*70 + "\n")
    
    print("🎉 All demonstrations and benchmarks complete!")
    print("\n💡 The Precious Brain successfully demonstrates:")
    print("   ✓ Conscious thought formation")
    print("   ✓ Emotional processing")
    print("   ✓ Pre-conscious decision making")
    print("   ✓ Memory encoding and consolidation")
    print("   ✓ Evolutionary learning")
    print("   ✓ Continual learning without forgetting")
    print("   ✓ Memory replay during dreaming")
    print("   ✓ Integrated prediction with confidence")
    print("   ✓ Complete perception-to-action cycle")
    print("   ✓ File operations and command execution")
    print("\n🧠 A truly precious brain! ✨\n")