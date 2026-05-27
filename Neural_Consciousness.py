"""
Advanced Neural Consciousness Algorithm
Implements biological principles of:
- Thought pattern formation (distributed firing patterns)
- Memory encoding via synaptic plasticity (LTP/LTD)
- Emotional signal processing (body-brain integration)
- Deterministic neural dynamics (pre-conscious decision making)

Author: Based on Neural_Algorithm.py framework
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# Import base neural processor
try:
    from Neural_Algorithm import (
        NeuralDataProcessor,
        LayerConfig,
        ActivationFunction,
        LossFunction
    )
    BASE_NEURAL_AVAILABLE = True
except ImportError:
    BASE_NEURAL_AVAILABLE = False
    print("Warning: Neural_Algorithm module not available")

# Import brain-like neural engine
try:
    from neural_network import (
        BrainLikeNeuralEngine,
        BrainConfig,
        NeuromodulatorState,
    )
    BRAIN_ENGINE_AVAILABLE = True
except ImportError:
    BRAIN_ENGINE_AVAILABLE = False
    print("Warning: neural_network module not available")


class EmotionType(Enum):
    """Primary emotion types with neural correlates"""
    FEAR = "fear"  # Amygdala activation
    HAPPINESS = "happiness"  # Nucleus accumbens, dopamine
    SADNESS = "sadness"  # Anterior cingulate
    ANGER = "anger"  # Hypothalamus
    LOVE = "love"  # Oxytocin pathways
    SURPRISE = "surprise"  # Rapid amygdala response
    DISGUST = "disgust"  # Insula activation
    NEUTRAL = "neutral"  # Baseline state


class NeuralRegion(Enum):
    """Brain regions involved in consciousness"""
    AMYGDALA = "amygdala"
    HIPPOCAMPUS = "hippocampus"
    PREFRONTAL_CORTEX = "prefrontal_cortex"
    NUCLEUS_ACCUMBENS = "nucleus_accumbens"
    ANTERIOR_CINGULATE = "anterior_cingulate"
    HYPOTHALAMUS = "hypothalamus"
    INSULA = "insula"
    SENSORY_CORTEX = "sensory_cortex"


@dataclass
class SynapticConnection:
    """Represents a single synaptic connection between neurons"""
    pre_neuron_id: int
    post_neuron_id: int
    weight: float
    last_activation: float = 0.0
    potentiation_count: int = 0  # LTP counter
    depression_count: int = 0  # LTD counter
    
    def strengthen(self, ltp_rate: float = 0.1):
        """Long-Term Potentiation - strengthen connection"""
        self.weight = min(1.0, self.weight + ltp_rate)
        self.potentiation_count += 1
    
    def weaken(self, ltd_rate: float = 0.05):
        """Long-Term Depression - weaken connection"""
        self.weight = max(0.0, self.weight - ltd_rate)
        self.depression_count += 1


@dataclass
class ThoughtPattern:
    """Represents a distributed firing pattern (a 'thought')"""
    pattern_id: str
    active_neurons: Set[int]
    activation_strength: float
    associated_memory: Optional[str] = None
    emotional_valence: float = 0.0  # -1 (negative) to +1 (positive)
    timestamp: float = 0.0
    
    def similarity(self, other: 'ThoughtPattern') -> float:
        """Compute overlap between two thought patterns"""
        if not self.active_neurons or not other.active_neurons:
            return 0.0
        intersection = len(self.active_neurons & other.active_neurons)
        union = len(self.active_neurons | other.active_neurons)
        return intersection / union if union > 0 else 0.0


@dataclass
class EmotionalState:
    """Represents current emotional state with physiological markers"""
    emotion: EmotionType
    intensity: float  # 0.0 to 1.0
    heart_rate: float = 70.0  # bpm
    cortisol_level: float = 0.5  # normalized
    dopamine_level: float = 0.5  # normalized
    oxytocin_level: float = 0.5  # normalized
    body_arousal: float = 0.0  # -1 (calm) to +1 (aroused)
    
    def update_physiology(self):
        """Update physiological markers based on emotion"""
        if self.emotion == EmotionType.FEAR:
            self.heart_rate = 70 + 50 * self.intensity
            self.cortisol_level = 0.5 + 0.5 * self.intensity
            self.body_arousal = self.intensity
        elif self.emotion == EmotionType.HAPPINESS:
            self.dopamine_level = 0.5 + 0.5 * self.intensity
            self.body_arousal = 0.3 * self.intensity
        elif self.emotion == EmotionType.SADNESS:
            self.heart_rate = 70 - 10 * self.intensity
            self.dopamine_level = 0.5 - 0.3 * self.intensity
            self.body_arousal = -0.5 * self.intensity
        elif self.emotion == EmotionType.ANGER:
            self.heart_rate = 70 + 40 * self.intensity
            self.cortisol_level = 0.5 + 0.4 * self.intensity
            self.body_arousal = 0.8 * self.intensity
        elif self.emotion == EmotionType.LOVE:
            self.oxytocin_level = 0.5 + 0.5 * self.intensity
            self.dopamine_level = 0.5 + 0.3 * self.intensity


@dataclass
class Memory:
    """Represents a memory stored as synaptic pattern"""
    memory_id: str
    content: np.ndarray
    synaptic_pattern: Dict[Tuple[int, int], float]  # (pre, post) -> weight
    strength: float  # How consolidated the memory is
    retrieval_count: int = 0
    creation_time: float = 0.0
    last_access_time: float = 0.0
    emotional_tag: Optional[EmotionType] = None
    
    def decay(self, decay_rate: float = 0.01):
        """Memory decay over time (unused memories weaken)"""
        self.strength = max(0.0, self.strength - decay_rate)
    
    def consolidate(self, consolidation_rate: float = 0.05):
        """Memory consolidation (strengthening through retrieval)"""
        self.strength = min(1.0, self.strength + consolidation_rate)
        self.retrieval_count += 1


class ConsciousnessProcessor:
    """
    Advanced neural processor simulating consciousness mechanisms:
    - Distributed thought patterns
    - Synaptic plasticity-based memory
    - Emotion-body integration
    - Deterministic pre-conscious processing
    """
    
    def __init__(self,
                 n_neurons: int = 10000,
                 n_regions: int = 8,
                 decision_latency_ms: float = 300.0,
                 ltp_threshold: float = 0.7,
                 ltd_threshold: float = 0.3):
        """
        Initialize consciousness processor
        
        Args:
            n_neurons: Total number of neurons in network
            n_regions: Number of brain regions
            decision_latency_ms: Time before conscious awareness (200-500ms)
            ltp_threshold: Activation threshold for LTP
            ltd_threshold: Activation threshold below which LTD occurs
        """
        self.n_neurons = n_neurons
        self.n_regions = n_regions
        self.decision_latency_ms = decision_latency_ms
        self.ltp_threshold = ltp_threshold
        self.ltd_threshold = ltd_threshold
        
        # Neural state
        self.neuron_activations = np.zeros(n_neurons)
        self.neuron_regions = self._assign_regions()
        
        # Synaptic connections (sparse representation)
        self.synapses: Dict[Tuple[int, int], SynapticConnection] = {}
        self._initialize_synapses()
        
        # Memory storage
        self.memories: Dict[str, Memory] = {}
        self.thought_patterns: List[ThoughtPattern] = []
        
        # Emotional system
        self.current_emotion = EmotionalState(
            emotion=EmotionType.NEUTRAL,
            intensity=0.0
        )
        
        # Decision buffer (pre-conscious processing)
        self.decision_buffer: List[Tuple[float, str, np.ndarray]] = []
        
        # Metrics
        self.total_thoughts = 0
        self.total_memories_formed = 0
        self.total_emotions_processed = 0

        # ── BrainLikeNeuralEngine integration ─────────────────────────────────
        # Provides: real cortex output, neuromodulators, global workspace,
        # free energy, working memory, and spiking dynamics.
        if BRAIN_ENGINE_AVAILABLE:
            _cfg = BrainConfig(
                input_size=16,        # compressed sensory dim fed to engine
                hidden_sizes=[32, 16],
                output_size=8,
                learning_rate=0.005,
                stdp_lr=0.01,
                attention_heads=4,
                attention_dim=16,
                memory_size=8,
                dopamine_decay=0.92,
            )
            self.brain_engine = BrainLikeNeuralEngine(_cfg)
            self._brain_tick_reward = 0.0   # updated by process_emotion()
        else:
            self.brain_engine = None
        
    def _assign_regions(self) -> np.ndarray:
        """Assign neurons to different brain regions"""
        regions = np.zeros(self.n_neurons, dtype=int)
        neurons_per_region = self.n_neurons // self.n_regions
        
        for i in range(self.n_regions):
            start_idx = i * neurons_per_region
            end_idx = start_idx + neurons_per_region
            regions[start_idx:end_idx] = i
        
        return regions
    
    def _initialize_synapses(self, connection_probability: float = 0.01):
        """Initialize sparse synaptic connections"""
        n_connections = int(self.n_neurons * self.n_neurons * connection_probability)

        print(f"Initializing {n_connections:,} synaptic connections...")

        # Generate pairs in batches (vectorized) to avoid a slow Python loop
        rng = np.random.default_rng()
        while len(self.synapses) < n_connections:
            remaining = n_connections - len(self.synapses)
            batch = max(remaining, 1000)
            pre_arr  = rng.integers(0, self.n_neurons, size=batch)
            post_arr = rng.integers(0, self.n_neurons, size=batch)
            w_arr    = rng.uniform(0.1, 0.5, size=batch)
            for pre, post, w in zip(pre_arr, post_arr, w_arr):
                key = (int(pre), int(post))
                if key[0] != key[1] and key not in self.synapses:
                    self.synapses[key] = SynapticConnection(
                        pre_neuron_id=key[0], post_neuron_id=key[1], weight=float(w)
                    )
                    if len(self.synapses) >= n_connections:
                        break

        self._rebuild_syn_arrays()
    
    def _rebuild_syn_arrays(self):
        """Build numpy arrays from synapses dict for vectorized propagation."""
        keys = list(self.synapses.keys())
        self._syn_keys = keys
        if keys:
            self._syn_pre     = np.array([k[0] for k in keys], dtype=np.int32)
            self._syn_post    = np.array([k[1] for k in keys], dtype=np.int32)
            self._syn_weights = np.array([self.synapses[k].weight for k in keys], dtype=np.float64)
        else:
            self._syn_pre     = np.empty(0, dtype=np.int32)
            self._syn_post    = np.empty(0, dtype=np.int32)
            self._syn_weights = np.empty(0, dtype=np.float64)

    def form_thought(self,
                     input_pattern: np.ndarray,
                     thought_name: str = "unnamed") -> ThoughtPattern:
        """
        Create a thought as a distributed firing pattern
        
        Process:
        1. Input activates initial neurons
        2. Activation spreads through synaptic connections
        3. Pattern stabilizes into coherent thought
        4. Pattern is recorded
        """
        # Reset activations
        self.neuron_activations = np.zeros(self.n_neurons)

        # ── Brain engine: compress input → cortex → fused representation ──────
        # Use cortex_vec to SELECT which neurons activate rather than adding
        # uniformly to all neurons.  Each of the 8 cortex dims gates one
        # brain-region block (~625 neurons each), producing a sparse, input-
        # specific activation pattern (~500–1500 neurons) that makes thought
        # patterns distinguishable and memory retrieval meaningful.
        if self.brain_engine is not None:
            # Compress/pad input_pattern to engine's input_size (16 dims)
            eng_in = np.zeros(16)
            chunk = min(len(input_pattern), 16)
            eng_in[:chunk] = input_pattern[:chunk] / (np.linalg.norm(input_pattern[:chunk]) + 1e-8)
            brain_state = self.brain_engine.process(eng_in, reward=self._brain_tick_reward)
            cortex_vec  = brain_state["cortex_output"]          # shape (8,)
            # Gate each brain-region block by its corresponding cortex dimension.
            # A positive cortex value activates that region; negative suppresses it.
            block = self.n_neurons // 8
            region_mask = np.zeros(self.n_neurons)
            for i, cv in enumerate(cortex_vec):
                start, end = i * block, min((i + 1) * block, self.n_neurons)
                region_mask[start:end] = float(np.tanh(cv))   # in (-1, 1)
            # Initialize activations: raw input ONLY in first input_size slots,
            # then multiply by region_mask so inactive regions stay near 0.
            input_size = min(len(input_pattern), self.n_neurons)
            self.neuron_activations[:input_size] = input_pattern[:input_size]
            # Apply region gating — suppresses neurons in cortex-inhibited regions
            self.neuron_activations *= np.where(region_mask > 0, region_mask, 0.05)
            # Store brain state for use by process_emotion / decision
            self._last_brain_state = brain_state
        else:
            self._last_brain_state = None
            # Initialize with input pattern (sensory input, memory retrieval, etc.)
            input_size = min(len(input_pattern), self.n_neurons)
            self.neuron_activations[:input_size] = input_pattern[:input_size]
        
        # Propagate activation through network (vectorized over synapses)
        for _ in range(5):
            new_activations = self.neuron_activations.copy()
            firing_mask = self.neuron_activations[self._syn_pre] > 0.5
            if firing_mask.any():
                contributions = (
                    self.neuron_activations[self._syn_pre] * self._syn_weights * firing_mask
                )
                np.add.at(new_activations, self._syn_post, contributions)
            self.neuron_activations = 1.0 / (1.0 + np.exp(-new_activations))

        # ── Brain engine: apply region gating AFTER propagation ───────────────
        # After sigmoid spreading, apply the cortex region mask so that only
        # the brain-region blocks selected by cortex_vec remain active.
        # Guarantee at least one region open so thoughts are never empty.
        if self.brain_engine is not None and hasattr(self, '_last_brain_state') \
                and self._last_brain_state is not None:
            cortex_vec = self._last_brain_state["cortex_output"]
            block = self.n_neurons // 8
            region_gate = np.zeros(self.n_neurons)
            open_count = 0
            for i, cv in enumerate(cortex_vec):
                start, end = i * block, min((i + 1) * block, self.n_neurons)
                # Continuous gate: tanh maps cortex output to (0.05, 1.0)
                # so patterns vary within regions, not just by region count.
                gate_val = float(np.clip(np.tanh(cv), 0.05, 1.0))
                region_gate[start:end] = gate_val
                if cv > 0.0:
                    open_count += 1
            if open_count == 0:
                best_region = int(np.argmax(cortex_vec))
                s, e = best_region * block, min((best_region + 1) * block, self.n_neurons)
                region_gate[s:e] = 1.0
            self.neuron_activations *= region_gate

        # k-Winner-Take-All: keep top ~5% of neurons active.
        # Pure relative threshold — no absolute floor — so patterns remain
        # non-empty even after continuous region gating suppresses activations.
        # ~250 active neurons → only ~0.25% of synapses get LTP per step.
        k = max(50, self.n_neurons // 20)
        kth_val = float(np.partition(self.neuron_activations, -k)[-k])
        active_neurons = set(np.where(self.neuron_activations >= kth_val)[0].tolist())

        # Mean over the ACTIVE neurons only — gives a meaningful strength signal
        # regardless of how many neurons k-WTA selects.
        activation_strength = (
            float(np.mean(self.neuron_activations[list(active_neurons)]))
            if active_neurons else 0.0
        )
        
        # Create thought pattern
        thought = ThoughtPattern(
            pattern_id=f"thought_{self.total_thoughts}_{thought_name}",
            active_neurons=active_neurons,
            activation_strength=activation_strength,
            timestamp=time.time()
        )
        
        self.thought_patterns.append(thought)
        self.total_thoughts += 1
        
        # Apply synaptic plasticity
        self._apply_plasticity(active_neurons)
        
        return thought
    
    def _apply_plasticity(self, active_neurons: Set[int]):
        """Apply LTP/LTD based on correlated firing — "cells that fire together wire together"."""
        if not active_neurons:
            return

        active_arr = np.zeros(self.n_neurons, dtype=bool)
        active_arr[list(active_neurons)] = True

        pre_active  = active_arr[self._syn_pre]
        post_active = active_arr[self._syn_post]

        ltp_mask = pre_active & post_active  & (self._syn_weights < self.ltp_threshold)
        ltd_mask = pre_active & ~post_active & (self._syn_weights > self.ltd_threshold)

        self._syn_weights[ltp_mask] = np.minimum(1.0, self._syn_weights[ltp_mask] + 0.1)
        self._syn_weights[ltd_mask] = np.maximum(0.0, self._syn_weights[ltd_mask] - 0.05)

        # Sync changed weights and update counters in SynapticConnection objects
        changed_indices = np.where(ltp_mask | ltd_mask)[0]
        for i in changed_indices:
            key = self._syn_keys[i]
            syn = self.synapses[key]
            syn.weight = float(self._syn_weights[i])
            if ltp_mask[i]:
                syn.potentiation_count += 1
            if ltd_mask[i]:
                syn.depression_count += 1
    
    def encode_memory(self,
                     memory_content: np.ndarray,
                     memory_name: str,
                     emotional_context: Optional[EmotionType] = None) -> Memory:
        """
        Encode a memory through synaptic pattern strengthening
        
        Process:
        1. Form thought pattern from content
        2. Strengthen synaptic connections in pattern
        3. Store as retrievable memory
        """
        # Form thought pattern
        thought = self.form_thought(memory_content, memory_name)
        
        # Extract synaptic pattern
        synaptic_pattern = {}
        for (pre, post), synapse in self.synapses.items():
            if pre in thought.active_neurons and post in thought.active_neurons:
                synaptic_pattern[(pre, post)] = synapse.weight
        
        # Create memory
        memory = Memory(
            memory_id=f"mem_{self.total_memories_formed}_{memory_name}",
            content=memory_content,
            synaptic_pattern=synaptic_pattern,
            strength=0.5,  # Initial strength
            creation_time=time.time(),
            last_access_time=time.time(),
            emotional_tag=emotional_context
        )
        
        self.memories[memory.memory_id] = memory
        self.total_memories_formed += 1
        
        # Emotional memories are strengthened more
        if emotional_context and emotional_context != EmotionType.NEUTRAL:
            memory.consolidate(consolidation_rate=0.2)
        
        return memory
    
    def retrieve_memory(self, query_pattern: np.ndarray) -> Optional[Memory]:
        """
        Retrieve memory by pattern matching
        
        Process:
        1. Form thought from query
        2. Find memory with most similar synaptic pattern
        3. Reactivate and strengthen retrieved memory
        """
        if not self.memories:
            return None
        
        # Form query thought
        query_thought = self.form_thought(query_pattern, "query")
        
        # Find best matching memory
        best_memory = None
        best_similarity = 0.0
        
        for memory in self.memories.values():
            # Reconstruct memory thought pattern
            memory_neurons = set()
            for (pre, post) in memory.synaptic_pattern.keys():
                memory_neurons.add(pre)
                memory_neurons.add(post)
            
            memory_thought = ThoughtPattern(
                pattern_id=memory.memory_id,
                active_neurons=memory_neurons,
                activation_strength=memory.strength,
                timestamp=memory.last_access_time
            )
            
            # Compute similarity
            similarity = query_thought.similarity(memory_thought)
            
            # Weight by memory strength
            weighted_similarity = similarity * memory.strength
            
            if weighted_similarity > best_similarity:
                best_similarity = weighted_similarity
                best_memory = memory
        
        # Consolidate retrieved memory
        if best_memory and best_similarity > 0.3:
            best_memory.consolidate()
            best_memory.last_access_time = time.time()
            return best_memory
        
        return None
    
    def process_emotion(self,
                       sensory_input: np.ndarray,
                       body_signal: Dict[str, float]) -> EmotionalState:
        """
        Process emotion through body-brain integration
        
        Steps:
        1. Sensory input triggers amygdala (fast pathway)
        2. Body signals change (heart rate, hormones)
        3. Cortex interprets combined signals
        4. Emotion is labeled
        """
        # Step 1: Fast amygdala response
        amygdala_neurons = np.where(self.neuron_regions == 0)[0]  # Region 0 = amygdala
        amygdala_activation = np.mean(self.neuron_activations[amygdala_neurons])
        
        # Step 2: Process body signals
        heart_rate = body_signal.get('heart_rate', 70.0)
        arousal = body_signal.get('arousal', 0.0)

        # ── Brain engine: modulate arousal with real dopamine level ───────────
        if self.brain_engine is not None and hasattr(self, '_last_brain_state') \
                and self._last_brain_state is not None:
            nm = self._last_brain_state["neuromodulators"]
            da_bias  = (float(nm["dopamine"])       - 0.5) * 0.8   # [-0.4, +0.4]
            ne_boost = (float(nm["norepinephrine"])  - 0.3) * 0.4   # arousal drive
            ser_calm = (0.7 - float(nm["serotonin"])) * 0.3         # low ser → more negative
            arousal  = float(np.clip(arousal + da_bias + ne_boost + ser_calm, -1.0, 1.0))

        # Step 3: Cortical interpretation
        if arousal > 0.7 and amygdala_activation > 0.7:
            emotion_type = EmotionType.FEAR
            intensity    = min(1.0, (arousal + amygdala_activation) / 2)
        elif arousal > 0.6 and heart_rate > 90:
            emotion_type = EmotionType.ANGER
            intensity    = arousal
        elif amygdala_activation > 0.6 and 0.3 < arousal <= 0.7:
            # Rapid amygdala response without high HR → surprise
            emotion_type = EmotionType.SURPRISE
            intensity    = amygdala_activation
        elif arousal > 0.5 and amygdala_activation < 0.3:
            emotion_type = EmotionType.HAPPINESS
            intensity    = arousal
        elif 0.2 < arousal <= 0.5 and amygdala_activation < 0.3:
            # Calm positive arousal, low threat → love/bonding
            emotion_type = EmotionType.LOVE
            intensity    = arousal
        elif arousal < -0.3 and amygdala_activation > 0.4:
            # Negative arousal with threat response → disgust (insula-driven)
            emotion_type = EmotionType.DISGUST
            intensity    = min(1.0, amygdala_activation * abs(arousal))
        elif arousal < 0.3:
            emotion_type = EmotionType.SADNESS
            intensity    = abs(arousal)
        else:
            emotion_type = EmotionType.NEUTRAL
            intensity    = 0.0
        
        # Create emotional state
        emotion = EmotionalState(
            emotion=emotion_type,
            intensity=intensity,
            heart_rate=heart_rate,
            body_arousal=arousal
        )

        # ── Brain engine: override physiology with real neuromodulator levels ─
        # dopamine, serotonin, ACh from the engine are more accurate than the
        # rule-based estimates above.  free_energy maps to cortisol (surprise).
        if self.brain_engine is not None and hasattr(self, '_last_brain_state') \
                and self._last_brain_state is not None:
            nm  = self._last_brain_state["neuromodulators"]
            fe  = float(self._last_brain_state["free_energy"])
            emotion.dopamine_level  = float(nm["dopamine"])
            emotion.oxytocin_level  = float(nm["acetylcholine"])   # ACh ~ bonding
            emotion.cortisol_level  = min(1.0, fe / 3.0)           # FE ~ stress
            # reward signal for next brain engine tick: positive = pleasant emotion
            self._brain_tick_reward = (
                0.5 if emotion_type == EmotionType.HAPPINESS else
               -0.5 if emotion_type in (EmotionType.FEAR, EmotionType.ANGER) else
                0.0
            )
        
        emotion.update_physiology()
        self.current_emotion = emotion
        self.total_emotions_processed += 1
        
        return emotion
    
    def pre_conscious_decision(self,
                              decision_input: np.ndarray,
                              decision_label: str) -> Tuple[float, str]:
        """
        Simulate pre-conscious decision making
        
        Brain decides 200-500ms BEFORE conscious awareness
        
        Process:
        1. Neural processing begins immediately
        2. Decision forms in unconscious networks
        3. After latency period, "decision" enters awareness
        4. Person feels they "made" the decision
        """
        # Record decision start time
        decision_start = time.time()
        
        # Process decision unconsciously
        decision_thought = self.form_thought(decision_input, decision_label)

        # ── Brain engine: workspace winner biases decision confidence ─────────
        # "spiking"   winner → fast/intuitive decision (lower latency)
        # "attention" winner → deliberate/reflective   (normal latency)
        # "memory"    winner → experience-based        (memory-weighted)
        if self.brain_engine is not None and hasattr(self, '_last_brain_state') \
                and self._last_brain_state is not None:
            winner      = self._last_brain_state.get("workspace_winner", "attention")
            mem_norm    = float(np.linalg.norm(self._last_brain_state.get("memory_state",
                                                                           np.zeros(8))))
            lif_spikes  = int(self._last_brain_state.get("lif_spikes", 0))

            if winner == "spiking":
                # fast intuitive path: shorten latency by spike activity
                self.decision_latency_ms = max(150.0, 300.0 - lif_spikes * 5)
            elif winner == "memory":
                # memory-based: weight decision by memory consolidation
                decision_thought.emotional_valence = float(np.tanh(mem_norm * 0.5))
                self.decision_latency_ms = 300.0
            else:
                # deliberate attention path: normal latency
                self.decision_latency_ms = 300.0
        
        # Compute "decision" based on neural activity
        decision_activation = decision_thought.activation_strength
        
        # Add to decision buffer (not yet conscious)
        self.decision_buffer.append((
            decision_start,
            decision_label,
            decision_input
        ))
        
        return decision_start, decision_label
    
    def check_conscious_awareness(self) -> List[str]:
        """
        Check which decisions have crossed into conscious awareness
        """
        current_time = time.time()
        conscious_decisions = []
        
        # Remove decisions that are now "conscious"
        remaining_buffer = []
        
        for decision_time, label, _ in self.decision_buffer:
            time_since_decision = (current_time - decision_time) * 1000  # ms
            
            if time_since_decision >= self.decision_latency_ms:
                # Decision is now conscious
                conscious_decisions.append(label)
            else:
                # Still pre-conscious
                remaining_buffer.append((decision_time, label, _))
        
        self.decision_buffer = remaining_buffer
        
        return conscious_decisions
    
    def simulate_consciousness_cycle(self,
                                    input_stream: List[np.ndarray],
                                    input_labels: List[str],
                                    verbose: bool = True):
        """
        Simulate a complete consciousness cycle:
        - Thought formation
        - Memory encoding/retrieval
        - Emotional processing
        - Pre-conscious decision making
        """
        if verbose:
            print("\n" + "=" * 70)
            print("CONSCIOUSNESS SIMULATION CYCLE")
            print("=" * 70)
        
        for i, (input_data, label) in enumerate(zip(input_stream, input_labels)):
            if verbose:
                print(f"\n[Timestep {i+1}] Processing: {label}")
            
            # 1. Form thought
            thought = self.form_thought(input_data, label)
            if verbose:
                print(f"  → Thought formed: {len(thought.active_neurons)} neurons active "
                      f"(strength: {thought.activation_strength:.3f})")
            
            # 2. Try to retrieve related memory
            memory = self.retrieve_memory(input_data)
            if memory and verbose:
                print(f"  → Memory retrieved: '{memory.memory_id}' "
                      f"(strength: {memory.strength:.3f}, "
                      f"retrievals: {memory.retrieval_count})")
            elif verbose:
                print(f"  → No matching memory found")
            
            # 3. Encode as new memory (with probability)
            if np.random.random() < 0.3:  # 30% chance to form memory
                emotion = np.random.choice(list(EmotionType))
                mem = self.encode_memory(input_data, label, emotion)
                if verbose:
                    print(f"  → New memory encoded: '{mem.memory_id}' "
                          f"with {emotion.value} tag")
            
            # 4. Process emotion
            body_signal = {
                'heart_rate': 70 + np.random.randn() * 10,
                'arousal': np.random.uniform(-1, 1)
            }
            emotion = self.process_emotion(input_data, body_signal)
            if verbose:
                print(f"  → Emotion: {emotion.emotion.value} "
                      f"(intensity: {emotion.intensity:.3f}, "
                      f"HR: {emotion.heart_rate:.1f})")
            
            # 5. Pre-conscious decision
            decision_time, decision = self.pre_conscious_decision(input_data, f"decision_{label}")
            if verbose:
                print(f"  → Decision formed pre-consciously at t={decision_time:.6f}")
            
            # Decay all memories each timestep so unused ones gradually weaken
            for mem in self.memories.values():
                mem.decay(decay_rate=0.005)

            # Small delay
            time.sleep(0.05)

            # 6. Check conscious awareness
            conscious = self.check_conscious_awareness()
            if conscious and verbose:
                print(f"  → Conscious awareness: {conscious}")
        
        if verbose:
            print("\n" + "=" * 70)
            print("CONSCIOUSNESS STATISTICS")
            print("=" * 70)
            print(f"Total thoughts formed: {self.total_thoughts}")
            print(f"Total memories stored: {self.total_memories_formed}")
            print(f"Total emotions processed: {self.total_emotions_processed}")
            print(f"Active synapses: {len(self.synapses):,}")
            
            # Synaptic plasticity stats
            ltp_count = sum(1 for s in self.synapses.values() if s.potentiation_count > 0)
            ltd_count = sum(1 for s in self.synapses.values() if s.depression_count > 0)
            print(f"Synapses strengthened (LTP): {ltp_count:,}")
            print(f"Synapses weakened (LTD): {ltd_count:,}")

            # ── Brain engine stats ─────────────────────────────────────────────
            if self.brain_engine is not None and hasattr(self, '_last_brain_state') \
                    and self._last_brain_state is not None:
                bs  = self._last_brain_state
                nm  = bs["neuromodulators"]
                print("\n── BrainEngine State ──────────────────────────────────")
                print(f"  Workspace winner : {bs.get('workspace_winner', 'n/a')}")
                print(f"  Free energy      : {bs.get('free_energy', 0.0):.4f}")
                print(f"  LIF spikes       : {bs.get('lif_spikes', 0)}")
                print(f"  SNN spikes       : {bs.get('spike_count', 0)}")
                print(f"  Memory norm      : {float(np.linalg.norm(bs.get('memory_state', np.zeros(8)))):.4f}")
                print(f"  Dopamine         : {nm['dopamine']:.4f}")
                print(f"  Serotonin        : {nm['serotonin']:.4f}")
                print(f"  Acetylcholine    : {nm['acetylcholine']:.4f}")
                print(f"  Norepinephrine   : {nm['norepinephrine']:.4f}")
                print(f"  Plasticity grown : {self.brain_engine.plasticity.grown_count}")
                print(f"  Plasticity pruned: {self.brain_engine.plasticity.pruned_count}")
            print("=" * 70)


def demonstrate_consciousness():
    """Demonstrate consciousness algorithm"""
    
    print("\n" + "🧠" * 35)
    print("ADVANCED NEURAL CONSCIOUSNESS ALGORITHM")
    print("🧠" * 35)
    
    # Create consciousness processor
    processor = ConsciousnessProcessor(
        n_neurons=5000,
        n_regions=8,
        decision_latency_ms=300.0
    )
    
    # Generate input stream (simulating sensory experiences)
    n_inputs = 10
    input_dim = 100
    
    input_stream = [np.random.randn(input_dim) for _ in range(n_inputs)]
    input_labels = [
        "visual_apple",
        "sound_music",
        "memory_childhood",
        "face_recognition",
        "language_word",
        "motor_command",
        "abstract_math",
        "social_emotion",
        "spatial_navigation",
        "decision_choice"
    ]
    
    # Run consciousness simulation
    processor.simulate_consciousness_cycle(
        input_stream,
        input_labels,
        verbose=True
    )
    
    print("\n✅ Consciousness simulation complete!\n")


if __name__ == "__main__":
    demonstrate_consciousness()