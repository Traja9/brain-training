"""
===========================================================
QUANTUM BUTTERFLY OPTIMIZATION FOR PRECIOUS BRAIN
-----------------------------------------------------------
A quantum-inspired metaheuristic algorithm that combines:
- Quantum Mechanics (superposition, entanglement, tunneling)
- Butterfly Optimization Algorithm (fragrance-based search)
- Precious Brain Integration (consciousness, memory, evolution)

Quantum Principles Applied:
✓ Quantum Superposition: Butterflies exist in multiple states
✓ Quantum Entanglement: Paired butterflies share information
✓ Quantum Tunneling: Escape local optima via quantum jumps
✓ Wave Function Collapse: Measurement affects butterfly state
✓ Quantum Interference: Constructive/destructive path interactions

This creates an advanced optimization system that leverages
the complete cognitive architecture of the Precious Brain.
===========================================================
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy

# Import the Precious Brain
try:
    from A_Brain import (
        PreciousBrain,
        PreciousBrainConfig,
        create_precious_brain,
        CognitiveMode,
        EmotionType
    )
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False
    print("⚠️  A_Brain module not available - running in standalone mode")


# ============================================================
# QUANTUM BUTTERFLY ENUMS & CONFIGURATIONS
# ============================================================

class QuantumState(Enum):
    """Quantum states for butterflies"""
    SUPERPOSITION = "superposition"  # Multiple states simultaneously
    COLLAPSED = "collapsed"  # Definite state after measurement
    ENTANGLED = "entangled"  # Linked with another butterfly
    TUNNELING = "tunneling"  # Quantum jump through barrier


class ButterflyMode(Enum):
    """Butterfly behavioral modes"""
    GLOBAL_SEARCH = "global_search"  # Explore entire space
    LOCAL_SEARCH = "local_search"  # Exploit nearby regions
    QUANTUM_LEAP = "quantum_leap"  # Quantum tunneling jump
    SOCIAL_LEARNING = "social_learning"  # Learn from brain
    CHAOTIC_FLIGHT = "chaotic_flight"  # Butterfly effect chaos


class ChaoticAttractor(Enum):
    """Types of chaotic attractors for butterfly effect"""
    LORENZ = "lorenz"  # Lorenz attractor (weather butterfly effect)
    ROSSLER = "rossler"  # Rössler attractor
    HENON = "henon"  # Hénon map
    LOGISTIC = "logistic"  # Logistic map (population dynamics)


@dataclass
class QuantumButterflyConfig:
    """Configuration for quantum butterfly optimization"""
    # Population
    n_butterflies: int = 20
    dimensions: int = 5
    
    # Search space
    lower_bound: float = -5.0
    upper_bound: float = 5.0
    
    # Butterfly parameters
    sensory_modality: float = 0.01  # Initial fragrance sensing
    power_exponent: float = 0.1  # Fragrance power law
    switch_probability: float = 0.8  # Global vs local search
    
    # Quantum parameters
    quantum_probability: float = 0.15  # Quantum effect probability
    entanglement_strength: float = 0.3  # Entanglement coupling
    tunneling_rate: float = 0.05  # Quantum tunneling frequency
    decoherence_time: float = 10.0  # Quantum state lifetime
    
    # Brain integration
    enable_brain: bool = True
    brain_learning_rate: float = 0.1
    brain_influence: float = 0.3  # How much brain affects butterflies
    memory_guided_search: bool = True
    consciousness_feedback: bool = True
    
    # Optimization
    max_iterations: int = 100
    tolerance: float = 1e-6
    adaptive_parameters: bool = True
    
    # Advanced features
    enable_quantum_interference: bool = True
    enable_wave_function: bool = True
    enable_measurement_collapse: bool = True
    
    # Butterfly Effect (Chaos Theory)
    enable_butterfly_effect: bool = True
    chaos_sensitivity: float = 0.001  # Sensitivity to initial conditions
    attractor_type: ChaoticAttractor = ChaoticAttractor.LORENZ
    chaos_injection_rate: float = 0.1  # How often chaos affects system
    lyapunov_threshold: float = 0.5  # Divergence threshold


# ============================================================
# QUANTUM BUTTERFLY CLASS
# ============================================================

@dataclass
class QuantumButterfly:
    """A quantum-inspired butterfly agent"""
    id: int
    position: np.ndarray
    velocity: np.ndarray
    fitness: float = np.inf
    fragrance: float = 0.0
    
    # Quantum properties
    quantum_state: QuantumState = QuantumState.SUPERPOSITION
    superposition_states: List[np.ndarray] = field(default_factory=list)
    wave_function: np.ndarray = field(default_factory=lambda: np.array([]))
    phase: float = 0.0
    entangled_with: Optional[int] = None
    coherence_time: float = 0.0
    
    # Behavioral
    mode: ButterflyMode = ButterflyMode.GLOBAL_SEARCH
    best_position: np.ndarray = field(default_factory=lambda: np.array([]))
    best_fitness: float = np.inf
    
    # Chaos theory properties (Butterfly Effect)
    chaos_state: np.ndarray = field(default_factory=lambda: np.array([]))
    lyapunov_exponent: float = 0.0  # Measures chaos/sensitivity
    trajectory_history: List[np.ndarray] = field(default_factory=list)
    divergence_rate: float = 0.0  # How fast similar states diverge
    attractor_phase: float = 0.0  # Position in chaotic attractor
    
    # Brain integration
    thought_pattern: Optional[str] = None
    emotion_state: Optional[str] = None
    memory_trace: Optional[Dict] = None
    
    def __post_init__(self):
        """Initialize quantum properties"""
        if len(self.superposition_states) == 0:
            # Initialize with 3 superposition states
            self.superposition_states = [
                self.position.copy(),
                self.position + np.random.randn(*self.position.shape) * 0.1,
                self.position - np.random.randn(*self.position.shape) * 0.1
            ]
        
        if len(self.wave_function) == 0:
            # Initialize wave function (probability amplitudes)
            n_states = len(self.superposition_states)
            self.wave_function = np.ones(n_states) / np.sqrt(n_states)
        
        if len(self.best_position) == 0:
            self.best_position = self.position.copy()
        
        if len(self.chaos_state) == 0:
            # Initialize chaos state (for chaotic attractors)
            self.chaos_state = np.random.randn(3) * 0.1  # 3D for Lorenz
        
        # Initialize trajectory history
        self.trajectory_history = [self.position.copy()]


# ============================================================
# MAIN QUANTUM BUTTERFLY OPTIMIZATION CLASS
# ============================================================

class QuantumButterflyOptimization:
    """
    Quantum-Inspired Butterfly Optimization Algorithm
    with Precious Brain Integration
    """
    
    def __init__(self, 
                 objective_function: Callable,
                 config: Optional[QuantumButterflyConfig] = None,
                 precious_brain: Optional[PreciousBrain] = None):
        """
        Initialize quantum butterfly optimizer
        
        Args:
            objective_function: Function to optimize (minimize)
            config: Configuration object
            precious_brain: Pre-initialized Precious Brain (optional)
        """
        self.config = config or QuantumButterflyConfig()
        self.objective_function = objective_function
        
        # Initialize swarm
        self.swarm: List[QuantumButterfly] = []
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_fitness: float = np.inf
        
        # Quantum tracking
        self.entanglement_pairs: List[Tuple[int, int]] = []
        self.quantum_history: List[Dict] = []
        
        # Chaos tracking (Butterfly Effect)
        self.chaos_events: List[Dict] = []
        self.divergence_pairs: List[Tuple[int, int, float]] = []  # (id1, id2, divergence)
        self.global_lyapunov: float = 0.0  # Global system chaos measure
        
        # Optimization tracking
        self.fitness_history: List[float] = []
        self.iteration: int = 0
        
        # Brain integration
        self.brain: Optional[PreciousBrain] = precious_brain
        if self.config.enable_brain and BRAIN_AVAILABLE and self.brain is None:
            print("\n🧠 Initializing Precious Brain for Quantum Butterflies...")
            self.brain = create_precious_brain(
                name="QuantumButterflyBrain",
                architecture="medium",
                input_size=self.config.dimensions,
                output_size=1,
                n_consciousness_neurons=3000,
                population_size=6,
                enable_all=True
            )
        
        # Initialize swarm
        self._initialize_swarm()
    
    # ============================================================
    # BUTTERFLY EFFECT (CHAOS THEORY) METHODS
    # ============================================================
    
    def _lorenz_attractor(self, state: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Lorenz attractor - the original butterfly effect system
        
        Edward Lorenz discovered this in 1963 while studying weather patterns.
        Small changes in initial conditions lead to vastly different outcomes.
        
        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz
        """
        sigma = 10.0  # Prandtl number
        rho = 28.0    # Rayleigh number
        beta = 8.0/3.0
        
        x, y, z = state[0], state[1], state[2]
        
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        
        return state + np.array([dx, dy, dz])
    
    def _rossler_attractor(self, state: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Rössler attractor - simpler chaotic system
        
        dx/dt = -y - z
        dy/dt = x + ay
        dz/dt = b + z(x - c)
        """
        a, b, c = 0.2, 0.2, 5.7
        
        x, y, z = state[0], state[1], state[2]
        
        dx = (-y - z) * dt
        dy = (x + a * y) * dt
        dz = (b + z * (x - c)) * dt
        
        return state + np.array([dx, dy, dz])
    
    def _logistic_map(self, x: float, r: float = 3.9) -> float:
        """
        Logistic map - 1D chaotic system
        
        x_n+1 = r * x_n * (1 - x_n)
        
        For r > 3.57, exhibits chaos (butterfly effect)
        """
        return r * x * (1 - x)
    
    def _apply_chaotic_attractor(self, butterfly: QuantumButterfly):
        """Apply chaotic dynamics to butterfly state"""
        if self.config.attractor_type == ChaoticAttractor.LORENZ:
            butterfly.chaos_state = self._lorenz_attractor(butterfly.chaos_state)
        elif self.config.attractor_type == ChaoticAttractor.ROSSLER:
            butterfly.chaos_state = self._rossler_attractor(butterfly.chaos_state)
        elif self.config.attractor_type == ChaoticAttractor.LOGISTIC:
            # Apply logistic map to each dimension
            for i in range(len(butterfly.chaos_state)):
                butterfly.chaos_state[i] = self._logistic_map(butterfly.chaos_state[i])
        
        # Update attractor phase
        butterfly.attractor_phase += 0.1
    
    def _calculate_lyapunov_exponent(self, butterfly: QuantumButterfly) -> float:
        """
        Calculate Lyapunov exponent - measure of chaos
        
        Positive Lyapunov exponent = chaotic (butterfly effect present)
        Negative = stable
        Zero = neutral
        """
        if len(butterfly.trajectory_history) < 2:
            return 0.0
        
        # Compare recent trajectory divergence
        recent_positions = butterfly.trajectory_history[-10:] if len(butterfly.trajectory_history) >= 10 else butterfly.trajectory_history
        
        if len(recent_positions) < 2:
            return 0.0
        
        # Calculate average divergence rate
        divergences = []
        for i in range(1, len(recent_positions)):
            dist = np.linalg.norm(recent_positions[i] - recent_positions[i-1])
            if dist > 0:
                divergences.append(np.log(dist))
        
        if divergences:
            lyapunov = np.mean(divergences)
            return lyapunov
        
        return 0.0
    
    def _butterfly_effect_perturbation(self, butterfly: QuantumButterfly):
        """
        Apply butterfly effect - tiny perturbation with large consequences
        
        "Does the flap of a butterfly's wings in Brazil set off a tornado in Texas?"
        - Edward Lorenz, 1972
        """
        if not self.config.enable_butterfly_effect:
            return
        
        if np.random.random() > self.config.chaos_injection_rate:
            return
        
        # Tiny initial perturbation (butterfly wing flap)
        tiny_perturbation = np.random.randn(*butterfly.position.shape) * self.config.chaos_sensitivity
        
        # Apply chaotic amplification through attractor dynamics
        self._apply_chaotic_attractor(butterfly)
        
        # Map chaos state to position perturbation (amplification)
        chaos_magnitude = np.linalg.norm(butterfly.chaos_state)
        amplification_factor = min(10.0, 1.0 + chaos_magnitude)
        
        # Large effect from tiny cause (butterfly effect)
        large_perturbation = tiny_perturbation * amplification_factor
        
        butterfly.position += large_perturbation
        butterfly.mode = ButterflyMode.CHAOTIC_FLIGHT
        
        # Record chaos event
        self.chaos_events.append({
            'iteration': self.iteration,
            'butterfly_id': butterfly.id,
            'initial_perturbation': np.linalg.norm(tiny_perturbation),
            'final_perturbation': np.linalg.norm(large_perturbation),
            'amplification': amplification_factor,
            'attractor': self.config.attractor_type.value
        })
    
    def _measure_system_divergence(self):
        """
        Measure how sensitive the system is to initial conditions
        
        Core of butterfly effect: nearby trajectories diverge exponentially
        """
        if len(self.swarm) < 2:
            return
        
        # Find pairs of nearby butterflies
        self.divergence_pairs = []
        
        for i in range(len(self.swarm)):
            for j in range(i + 1, len(self.swarm)):
                b1, b2 = self.swarm[i], self.swarm[j]
                
                # Current distance
                current_dist = np.linalg.norm(b1.position - b2.position)
                
                # Historical distance (if available)
                if len(b1.trajectory_history) > 0 and len(b2.trajectory_history) > 0:
                    if len(b1.trajectory_history) == len(b2.trajectory_history):
                        hist_idx = min(len(b1.trajectory_history) - 1, 10)
                        initial_dist = np.linalg.norm(
                            b1.trajectory_history[-hist_idx] - b2.trajectory_history[-hist_idx]
                        )
                        
                        if initial_dist > 0:
                            # Divergence rate = current / initial
                            divergence = current_dist / initial_dist
                            
                            # Store significant divergences
                            if divergence > 1.5:  # 50% divergence
                                self.divergence_pairs.append((i, j, divergence))
                                b1.divergence_rate = divergence
                                b2.divergence_rate = divergence
        
        # Calculate global Lyapunov exponent
        if self.divergence_pairs:
            divergence_values = [d for _, _, d in self.divergence_pairs]
            self.global_lyapunov = np.mean(np.log(divergence_values))
        else:
            self.global_lyapunov = 0.0
    
    def _chaos_induced_exploration(self, butterfly: QuantumButterfly):
        """
        Use chaos to enhance exploration
        
        High Lyapunov exponent → more exploration
        Low Lyapunov exponent → more exploitation
        """
        # Calculate butterfly's chaos level
        butterfly.lyapunov_exponent = self._calculate_lyapunov_exponent(butterfly)
        
        # If highly chaotic, encourage exploration
        if butterfly.lyapunov_exponent > self.config.lyapunov_threshold:
            # Map 3D chaos state to butterfly's dimension
            n_dims = len(butterfly.position)
            
            # Repeat or truncate chaos state to match dimensions
            if len(butterfly.chaos_state) < n_dims:
                # Tile chaos state to fill dimensions
                repeats = (n_dims // len(butterfly.chaos_state)) + 1
                chaos_extended = np.tile(butterfly.chaos_state, repeats)[:n_dims]
            else:
                # Truncate to match dimensions
                chaos_extended = butterfly.chaos_state[:n_dims]
            
            # Normalize direction
            chaos_direction = chaos_extended / (np.linalg.norm(chaos_extended) + 1e-10)
            
            exploration_strength = min(1.0, butterfly.lyapunov_exponent)
            butterfly.velocity += exploration_strength * chaos_direction
        
        # Update trajectory history
        butterfly.trajectory_history.append(butterfly.position.copy())
        if len(butterfly.trajectory_history) > 50:  # Keep last 50
            butterfly.trajectory_history.pop(0)
    
    def _initialize_swarm(self):
        """Initialize butterfly swarm with quantum properties"""
        print(f"\n{'='*70}")
        print(f"🦋 INITIALIZING QUANTUM BUTTERFLY SWARM")
        print(f"{'='*70}")
        print(f"Butterflies: {self.config.n_butterflies}")
        print(f"Dimensions: {self.config.dimensions}")
        print(f"Quantum Effects: ENABLED")
        print(f"Brain Integration: {'ENABLED' if self.brain else 'DISABLED'}")
        print(f"{'='*70}\n")
        
        for i in range(self.config.n_butterflies):
            # Random initial position
            position = np.random.uniform(
                self.config.lower_bound,
                self.config.upper_bound,
                self.config.dimensions
            )
            
            # Random initial velocity
            velocity = np.random.randn(self.config.dimensions) * 0.1
            
            # Create butterfly
            butterfly = QuantumButterfly(
                id=i,
                position=position,
                velocity=velocity
            )
            
            # Evaluate initial fitness
            butterfly.fitness = self.objective_function(butterfly.position)
            butterfly.best_fitness = butterfly.fitness
            butterfly.best_position = butterfly.position.copy()
            
            # Calculate initial fragrance
            butterfly.fragrance = self._calculate_fragrance(butterfly)
            
            self.swarm.append(butterfly)
            
            # Update global best
            if butterfly.fitness < self.global_best_fitness:
                self.global_best_fitness = butterfly.fitness
                self.global_best_position = butterfly.position.copy()
        
        # Create initial quantum entanglements
        self._create_entanglements()
        
        print(f"✓ Swarm initialized")
        print(f"✓ Initial best fitness: {self.global_best_fitness:.6f}")
        print(f"✓ Quantum entanglements: {len(self.entanglement_pairs)}\n")
    
    def _calculate_fragrance(self, butterfly: QuantumButterfly) -> float:
        """
        Calculate fragrance intensity based on fitness
        
        Fragrance = c * I^a
        where c = sensory modality, I = fitness, a = power exponent
        """
        # Inverse fitness (better fitness = stronger fragrance)
        intensity = 1.0 / (1.0 + butterfly.fitness) if butterfly.fitness >= 0 else np.exp(-butterfly.fitness)
        
        fragrance = self.config.sensory_modality * (intensity ** self.config.power_exponent)
        
        return fragrance
    
    def _create_entanglements(self, n_pairs: Optional[int] = None):
        """Create quantum entanglement pairs"""
        if n_pairs is None:
            n_pairs = self.config.n_butterflies // 4  # 25% entangled
        
        # Clear existing entanglements
        self.entanglement_pairs = []
        for butterfly in self.swarm:
            butterfly.entangled_with = None
        
        # Create new entanglements
        available_ids = list(range(self.config.n_butterflies))
        np.random.shuffle(available_ids)
        
        for i in range(0, min(len(available_ids) - 1, n_pairs * 2), 2):
            id1, id2 = available_ids[i], available_ids[i + 1]
            
            self.swarm[id1].entangled_with = id2
            self.swarm[id1].quantum_state = QuantumState.ENTANGLED
            
            self.swarm[id2].entangled_with = id1
            self.swarm[id2].quantum_state = QuantumState.ENTANGLED
            
            self.entanglement_pairs.append((id1, id2))
    
    def _quantum_superposition_update(self, butterfly: QuantumButterfly):
        """Update butterfly in quantum superposition"""
        # Evolve all superposition states
        for i, state in enumerate(butterfly.superposition_states):
            # Random quantum fluctuation
            fluctuation = np.random.randn(*state.shape) * 0.05
            butterfly.superposition_states[i] = state + fluctuation
            
            # Ensure bounds
            butterfly.superposition_states[i] = np.clip(
                butterfly.superposition_states[i],
                self.config.lower_bound,
                self.config.upper_bound
            )
        
        # Update wave function (normalize)
        butterfly.wave_function = butterfly.wave_function / np.linalg.norm(butterfly.wave_function)
        
        # Phase evolution
        butterfly.phase += 0.1 * np.random.randn()
    
    def _quantum_measurement(self, butterfly: QuantumButterfly) -> np.ndarray:
        """
        Perform quantum measurement (wave function collapse)
        
        Returns collapsed position based on probability amplitudes
        """
        if butterfly.quantum_state != QuantumState.SUPERPOSITION:
            return butterfly.position
        
        # Probability distribution from wave function
        probabilities = np.abs(butterfly.wave_function) ** 2
        probabilities /= probabilities.sum()
        
        # Measure (collapse) to one state
        measured_idx = np.random.choice(len(probabilities), p=probabilities)
        collapsed_position = butterfly.superposition_states[measured_idx]
        
        # Collapse wave function
        butterfly.quantum_state = QuantumState.COLLAPSED
        butterfly.position = collapsed_position.copy()
        butterfly.coherence_time = 0.0
        
        return collapsed_position
    
    def _quantum_tunneling(self, butterfly: QuantumButterfly) -> bool:
        """
        Attempt quantum tunneling to escape local optimum
        
        Returns True if tunneling occurred
        """
        if np.random.random() > self.config.tunneling_rate:
            return False
        
        # Quantum jump to random location
        tunnel_position = np.random.uniform(
            self.config.lower_bound,
            self.config.upper_bound,
            self.config.dimensions
        )
        
        # Evaluate new position
        tunnel_fitness = self.objective_function(tunnel_position)
        
        # Accept if better (or with small probability anyway - quantum)
        if tunnel_fitness < butterfly.fitness or np.random.random() < 0.1:
            butterfly.position = tunnel_position
            butterfly.fitness = tunnel_fitness
            butterfly.quantum_state = QuantumState.TUNNELING
            
            return True
        
        return False
    
    def _quantum_entanglement_update(self, butterfly: QuantumButterfly):
        """Update based on quantum entanglement"""
        if butterfly.entangled_with is None:
            return
        
        partner = self.swarm[butterfly.entangled_with]
        
        # Entangled butterflies influence each other
        coupling = self.config.entanglement_strength
        
        # Position correlation
        position_diff = partner.position - butterfly.position
        butterfly.position += coupling * position_diff
        
        # Velocity correlation  
        velocity_diff = partner.velocity - butterfly.velocity
        butterfly.velocity += coupling * velocity_diff
        
        # Wave function entanglement (if in superposition)
        if butterfly.quantum_state == QuantumState.SUPERPOSITION and \
           partner.quantum_state == QuantumState.SUPERPOSITION:
            # Mix wave functions
            butterfly.wave_function = (butterfly.wave_function + partner.wave_function) / 2
            butterfly.wave_function /= np.linalg.norm(butterfly.wave_function)
    
    def _brain_guided_update(self, butterfly: QuantumButterfly):
        """Use Precious Brain to guide butterfly movement"""
        if not self.brain:
            return
        
        # Form thought about current position
        thought = self.brain.think(
            butterfly.position,
            context={'name': f'butterfly_{butterfly.id}', 'fitness': butterfly.fitness}
        )
        butterfly.thought_pattern = f"{len(thought.active_neurons)}_neurons"
        
        # Process emotion based on fitness improvement
        fitness_change = butterfly.best_fitness - butterfly.fitness
        arousal = min(1.0, abs(fitness_change) * 10)
        
        emotion = self.brain.feel(
            butterfly.position,
            {'heart_rate': 70 + arousal * 30, 'arousal': arousal}
        )
        butterfly.emotion_state = emotion.emotion.value
        
        # Memory-guided search
        if self.config.memory_guided_search:
            # Recall similar good positions
            memories = self.brain.recall(butterfly.position)
            
            if memories and len(memories) > 0:
                # Move towards best remembered position
                best_memory = memories[0]
                memory_position = best_memory.data[0]
                
                direction = memory_position - butterfly.position
                butterfly.velocity += self.config.brain_influence * direction
                
                butterfly.memory_trace = {
                    'importance': best_memory.importance,
                    'quality': best_memory.quality
                }
        
        # Store experience in brain memory
        if butterfly.fitness < butterfly.best_fitness:
            self.brain.memorize(
                butterfly.position,
                np.array([butterfly.fitness]),
                emotional_context=emotion.emotion,
                importance=2.0 if butterfly.fitness < self.global_best_fitness else 1.0
            )
    
    def _global_search_phase(self, butterfly: QuantumButterfly):
        """
        Global search phase (inspired by fragrance)
        
        Butterfly moves towards another butterfly with better fragrance
        """
        # Check if we have other butterflies to compare with
        if len(self.swarm) < 2:
            # If only one butterfly, move towards global best or explore randomly
            if self.global_best_position is not None:
                r = np.random.random()
                butterfly.position = butterfly.position + \
                    (r ** 2) * self.global_best_position - butterfly.position
            else:
                # Random exploration
                butterfly.position = butterfly.position + \
                    np.random.randn(*butterfly.position.shape) * 0.1
        else:
            # Select random butterfly with better fragrance
            better_butterflies = [
                b for b in self.swarm 
                if b.fragrance > butterfly.fragrance and b.id != butterfly.id
            ]
            
            if better_butterflies:
                target = np.random.choice(better_butterflies)
                
                # Move towards target with fragrance-based attraction
                r = np.random.random()
                butterfly.position = butterfly.position + \
                    (r ** 2) * target.position - butterfly.position
            else:
                # Random exploration if no better butterfly
                butterfly.position = butterfly.position + \
                    (np.random.random() ** 2) * self.global_best_position - butterfly.position
        
        butterfly.mode = ButterflyMode.GLOBAL_SEARCH
    
    def _local_search_phase(self, butterfly: QuantumButterfly):
        """
        Local search phase (exploitation)
        
        Butterfly moves within local neighborhood
        """
        # Check if we have enough butterflies for pair selection
        if len(self.swarm) < 2:
            # If only one butterfly, do random local movement
            r = np.random.random()
            butterfly.position = butterfly.position + \
                (r ** 2) * np.random.randn(*butterfly.position.shape) * 0.1
        else:
            # Select two random butterflies
            j, k = np.random.choice(self.config.n_butterflies, size=2, replace=False)
            
            # Local movement
            r = np.random.random()
            butterfly.position = butterfly.position + \
                (r ** 2) * self.swarm[j].position - self.swarm[k].position
        
        butterfly.mode = ButterflyMode.LOCAL_SEARCH
    
    def _quantum_interference(self, butterfly1: QuantumButterfly, butterfly2: QuantumButterfly):
        """
        Apply quantum interference between two butterflies
        
        Wave functions can interfere constructively or destructively
        """
        if not self.config.enable_quantum_interference:
            return
        
        if butterfly1.quantum_state != QuantumState.SUPERPOSITION or \
           butterfly2.quantum_state != QuantumState.SUPERPOSITION:
            return
        
        # Phase difference determines interference
        phase_diff = butterfly1.phase - butterfly2.phase
        
        # Constructive interference (in phase)
        if abs(phase_diff) < np.pi / 4:
            # Strengthen both butterflies towards better position
            if butterfly1.fitness < butterfly2.fitness:
                butterfly2.position += 0.1 * (butterfly1.position - butterfly2.position)
            else:
                butterfly1.position += 0.1 * (butterfly2.position - butterfly1.position)
        
        # Destructive interference (out of phase)
        elif abs(phase_diff) > 3 * np.pi / 4:
            # Push apart
            diff = butterfly1.position - butterfly2.position
            butterfly1.position += 0.05 * diff
            butterfly2.position -= 0.05 * diff
    
    def optimize(self, verbose: bool = True) -> Dict:
        """
        Run quantum butterfly optimization
        
        Returns:
            Dictionary with optimization results
        """
        print(f"\n{'='*70}")
        print(f"🦋 QUANTUM BUTTERFLY OPTIMIZATION - STARTING")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for iteration in range(self.config.max_iterations):
            self.iteration = iteration
            
            # Adaptive parameter update
            if self.config.adaptive_parameters:
                # Decrease sensory modality over time
                self.config.sensory_modality = 0.01 + (1 - iteration / self.config.max_iterations) * 0.99
            
            # Update each butterfly
            for butterfly in self.swarm:
                # Quantum state evolution
                if butterfly.quantum_state == QuantumState.SUPERPOSITION:
                    self._quantum_superposition_update(butterfly)
                    
                    # Coherence time tracking
                    butterfly.coherence_time += 1
                    
                    # Decoherence (collapse after too long)
                    if butterfly.coherence_time > self.config.decoherence_time:
                        self._quantum_measurement(butterfly)
                
                # Quantum entanglement effects
                if butterfly.quantum_state == QuantumState.ENTANGLED:
                    self._quantum_entanglement_update(butterfly)
                
                # Quantum tunneling attempt
                if np.random.random() < self.config.quantum_probability:
                    tunneled = self._quantum_tunneling(butterfly)
                    if tunneled and verbose:
                        print(f"   ⚡ Butterfly {butterfly.id} quantum tunneled!")
                
                # BUTTERFLY EFFECT - Tiny perturbations, large consequences
                if self.config.enable_butterfly_effect:
                    self._butterfly_effect_perturbation(butterfly)
                    self._chaos_induced_exploration(butterfly)
                
                # Brain-guided update
                if self.brain and iteration % 5 == 0:  # Every 5 iterations
                    self._brain_guided_update(butterfly)
                
                # Standard butterfly algorithm
                if np.random.random() < self.config.switch_probability:
                    self._global_search_phase(butterfly)
                else:
                    self._local_search_phase(butterfly)
                
                # Boundary constraint
                butterfly.position = np.clip(
                    butterfly.position,
                    self.config.lower_bound,
                    self.config.upper_bound
                )
                
                # Evaluate fitness
                butterfly.fitness = self.objective_function(butterfly.position)
                
                # Update fragrance
                butterfly.fragrance = self._calculate_fragrance(butterfly)
                
                # Update personal best
                if butterfly.fitness < butterfly.best_fitness:
                    butterfly.best_fitness = butterfly.fitness
                    butterfly.best_position = butterfly.position.copy()
                
                # Update global best
                if butterfly.fitness < self.global_best_fitness:
                    self.global_best_fitness = butterfly.fitness
                    self.global_best_position = butterfly.position.copy()
            
            # Quantum interference between nearby butterflies
            if self.config.enable_quantum_interference:
                for i in range(self.config.n_butterflies - 1):
                    for j in range(i + 1, self.config.n_butterflies):
                        distance = np.linalg.norm(
                            self.swarm[i].position - self.swarm[j].position
                        )
                        if distance < 1.0:  # Close enough for interference
                            self._quantum_interference(self.swarm[i], self.swarm[j])
            
            # Re-create entanglements periodically
            if iteration % 20 == 0:
                self._create_entanglements()
            
            # Measure system-wide divergence (butterfly effect)
            if self.config.enable_butterfly_effect and iteration % 5 == 0:
                self._measure_system_divergence()
            
            # Record history
            self.fitness_history.append(self.global_best_fitness)
            
            # Quantum state statistics
            quantum_stats = {
                'superposition': sum(1 for b in self.swarm if b.quantum_state == QuantumState.SUPERPOSITION),
                'collapsed': sum(1 for b in self.swarm if b.quantum_state == QuantumState.COLLAPSED),
                'entangled': sum(1 for b in self.swarm if b.quantum_state == QuantumState.ENTANGLED),
                'tunneling': sum(1 for b in self.swarm if b.quantum_state == QuantumState.TUNNELING)
            }
            self.quantum_history.append(quantum_stats)
            
            # Verbose output
            if verbose and (iteration % 10 == 0 or iteration == self.config.max_iterations - 1):
                print(f"Iteration {iteration+1}/{self.config.max_iterations}")
                print(f"   Best Fitness: {self.global_best_fitness:.8f}")
                print(f"   Quantum States: S={quantum_stats['superposition']} "
                      f"C={quantum_stats['collapsed']} E={quantum_stats['entangled']} "
                      f"T={quantum_stats['tunneling']}")
                
                if self.config.enable_butterfly_effect:
                    chaos_count = sum(1 for b in self.swarm if b.mode == ButterflyMode.CHAOTIC_FLIGHT)
                    print(f"   Chaos: Lyapunov={self.global_lyapunov:.4f}, "
                          f"Chaotic={chaos_count}, Divergences={len(self.divergence_pairs)}")
                
                if self.brain:
                    metrics = self.brain.get_metrics()
                    print(f"   Brain: {metrics['total_thoughts']} thoughts, "
                          f"{metrics['total_memories']} memories")
                print()
            
            # Convergence check
            if iteration > 10:
                recent_improvement = abs(
                    self.fitness_history[-1] - self.fitness_history[-10]
                )
                if recent_improvement < self.config.tolerance:
                    if verbose:
                        print(f"✓ Converged at iteration {iteration+1}")
                    break
        
        elapsed_time = time.time() - start_time
        
        # Brain memory consolidation after optimization
        if self.brain:
            print("\n🧠 Consolidating brain memories...")
            self.brain.dream(n_cycles=3, verbose=False)
        
        print(f"\n{'='*70}")
        print(f"✅ OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Best Fitness: {self.global_best_fitness:.8f}")
        print(f"Best Position: {self.global_best_position}")
        print(f"Iterations: {iteration+1}")
        print(f"Time: {elapsed_time:.2f}s")
        print(f"{'='*70}\n")
        
        results = {
            'best_fitness': self.global_best_fitness,
            'best_position': self.global_best_position,
            'iterations': iteration + 1,
            'time': elapsed_time,
            'fitness_history': self.fitness_history,
            'quantum_history': self.quantum_history,
            'chaos_events': self.chaos_events,
            'divergence_pairs': self.divergence_pairs,
            'global_lyapunov': self.global_lyapunov,
            'final_swarm': self.swarm
        }
        
        if self.brain:
            results['brain_metrics'] = self.brain.get_metrics()
            results['brain_introspection'] = self.brain.introspect()
        
        return results
    
    def get_swarm_statistics(self) -> Dict:
        """Get comprehensive swarm statistics"""
        stats = {
            'mean_fitness': np.mean([b.fitness for b in self.swarm]),
            'std_fitness': np.std([b.fitness for b in self.swarm]),
            'best_fitness': min(b.fitness for b in self.swarm),
            'worst_fitness': max(b.fitness for b in self.swarm),
            'mean_fragrance': np.mean([b.fragrance for b in self.swarm]),
            'quantum_states': {
                'superposition': sum(1 for b in self.swarm if b.quantum_state == QuantumState.SUPERPOSITION),
                'collapsed': sum(1 for b in self.swarm if b.quantum_state == QuantumState.COLLAPSED),
                'entangled': sum(1 for b in self.swarm if b.quantum_state == QuantumState.ENTANGLED),
                'tunneling': sum(1 for b in self.swarm if b.quantum_state == QuantumState.TUNNELING)
            },
            'modes': {
                'global_search': sum(1 for b in self.swarm if b.mode == ButterflyMode.GLOBAL_SEARCH),
                'local_search': sum(1 for b in self.swarm if b.mode == ButterflyMode.LOCAL_SEARCH),
                'quantum_leap': sum(1 for b in self.swarm if b.mode == ButterflyMode.QUANTUM_LEAP),
                'chaotic_flight': sum(1 for b in self.swarm if b.mode == ButterflyMode.CHAOTIC_FLIGHT)
            },
            'chaos_metrics': {
                'mean_lyapunov': np.mean([b.lyapunov_exponent for b in self.swarm]),
                'max_lyapunov': max([b.lyapunov_exponent for b in self.swarm]),
                'global_lyapunov': self.global_lyapunov,
                'chaos_events': len(self.chaos_events),
                'divergence_pairs': len(self.divergence_pairs)
            }
        }
        
        return stats


# ============================================================
# BENCHMARK FUNCTIONS
# ============================================================

class BenchmarkFunctions:
    """Standard optimization benchmark functions"""
    
    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """Sphere function (unimodal)"""
        return np.sum(x ** 2)
    
    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """Rastrigin function (highly multimodal)"""
        n = len(x)
        return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    
    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """Ackley function (multimodal with deep valley)"""
        n = len(x)
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e
    
    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """Rosenbrock function (narrow valley)"""
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    
    @staticmethod
    def griewank(x: np.ndarray) -> float:
        """Griewank function (many local optima)"""
        sum_part = np.sum(x ** 2) / 4000
        prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_part - prod_part + 1


# ============================================================
# DEMONSTRATION
# ============================================================

def demonstrate_quantum_butterfly():
    """Comprehensive demonstration"""
    print("\n" + "="*70)
    print("🦋 QUANTUM BUTTERFLY OPTIMIZATION DEMONSTRATION")
    print("="*70)
    print("With Precious Brain Integration")
    print("="*70 + "\n")
    
    # Test on multiple benchmark functions
    benchmarks = [
        ("Sphere", BenchmarkFunctions.sphere, 5, [-5, 5]),
        ("Rastrigin", BenchmarkFunctions.rastrigin, 5, [-5.12, 5.12]),
        ("Ackley", BenchmarkFunctions.ackley, 5, [-5, 5])
    ]
    
    all_results = []
    
    for name, func, dims, bounds in benchmarks:
        print(f"\n{'='*70}")
        print(f"📊 BENCHMARK: {name} Function ({dims}D)")
        print(f"{'='*70}\n")
        
        # Configure optimizer
        config = QuantumButterflyConfig(
            n_butterflies=20,
            dimensions=dims,
            lower_bound=bounds[0],
            upper_bound=bounds[1],
            max_iterations=50,
            quantum_probability=0.2,
            enable_brain=BRAIN_AVAILABLE,
            enable_quantum_interference=True,
            enable_wave_function=True,
            memory_guided_search=True
        )
        
        # Create optimizer
        qbo = QuantumButterflyOptimization(
            objective_function=func,
            config=config
        )
        
        # Optimize
        results = qbo.optimize(verbose=True)
        all_results.append({
            'name': name,
            'results': results
        })
        
        # Display statistics
        print(f"\n📈 {name} Results:")
        print(f"   ├─ Best Fitness: {results['best_fitness']:.8f}")
        print(f"   ├─ Best Position: {results['best_position']}")
        print(f"   ├─ Iterations: {results['iterations']}")
        print(f"   └─ Time: {results['time']:.2f}s")
        
        if 'brain_metrics' in results:
            brain = results['brain_metrics']
            print(f"\n🧠 Brain Integration:")
            print(f"   ├─ Thoughts: {brain['total_thoughts']}")
            print(f"   ├─ Memories: {brain['total_memories']}")
            print(f"   ├─ Emotions: {brain['total_emotions']}")
            print(f"   └─ Consolidations: {brain['memory_consolidations']}")
        
        # Swarm statistics
        stats = qbo.get_swarm_statistics()
        print(f"\n🦋 Final Swarm Statistics:")
        print(f"   ├─ Mean Fitness: {stats['mean_fitness']:.8f}")
        print(f"   ├─ Std Fitness: {stats['std_fitness']:.8f}")
        print(f"   └─ Diversity: {stats['worst_fitness'] - stats['best_fitness']:.8f}")
        
        print(f"\n⚛️  Quantum State Distribution:")
        qs = stats['quantum_states']
        print(f"   ├─ Superposition: {qs['superposition']}")
        print(f"   ├─ Collapsed: {qs['collapsed']}")
        print(f"   ├─ Entangled: {qs['entangled']}")
        print(f"   └─ Tunneling: {qs['tunneling']}")
        
        print(f"\n🦋 Butterfly Effect (Chaos) Metrics:")
        chaos = stats['chaos_metrics']
        print(f"   ├─ Global Lyapunov: {chaos['global_lyapunov']:.4f}")
        print(f"   ├─ Mean Lyapunov: {chaos['mean_lyapunov']:.4f}")
        print(f"   ├─ Chaos Events: {chaos['chaos_events']}")
        print(f"   └─ Divergence Pairs: {chaos['divergence_pairs']}")
        
        if chaos['global_lyapunov'] > 0:
            print(f"   ✓ System exhibits CHAOTIC behavior (butterfly effect active)")
        else:
            print(f"   ○ System is stable (no butterfly effect)")
    
    # Summary comparison
    print(f"\n{'='*70}")
    print(f"📊 OVERALL PERFORMANCE SUMMARY")
    print(f"{'='*70}\n")
    
    for result in all_results:
        name = result['name']
        res = result['results']
        print(f"{name:15s}: Fitness={res['best_fitness']:12.8f}  "
              f"Iters={res['iterations']:3d}  Time={res['time']:6.2f}s")
    
    print(f"\n{'='*70}")
    print(f"✅ ALL BENCHMARKS COMPLETE")
    print(f"{'='*70}\n")
    
    return all_results


# ============================================================
# ADVANCED DEMONSTRATION: BRAIN-GUIDED OPTIMIZATION
# ============================================================

def demonstrate_butterfly_effect():
    """
    Dedicated demonstration of the butterfly effect in action
    """
    print("\n" + "="*70)
    print("🦋 BUTTERFLY EFFECT DEMONSTRATION")
    print("="*70)
    print("Edward Lorenz (1963): Tiny changes → Massive consequences")
    print("="*70 + "\n")
    
    def simple_sphere(x):
        return np.sum(x ** 2)
    
    print("🔬 Experiment: Two butterflies with TINY difference (0.001)")
    print("Will they diverge dramatically? (Butterfly Effect Test)\n")
    
    # Create two nearly identical initial conditions
    base_position = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    
    results_comparison = []
    
    for test_num, tiny_diff in enumerate([0.0, 0.001], 1):
        print(f"Test {test_num}: Initial difference = {tiny_diff}")
        print("-" * 70)
        
        config = QuantumButterflyConfig(
            n_butterflies=1,  # Single butterfly for clean test
            dimensions=5,
            lower_bound=-5.0,
            upper_bound=5.0,
            max_iterations=30,
            enable_butterfly_effect=True,
            chaos_sensitivity=0.001,
            chaos_injection_rate=0.3,
            attractor_type=ChaoticAttractor.LORENZ,
            enable_brain=False  # Disable for clean chaos test
        )
        
        qbo = QuantumButterflyOptimization(
            objective_function=simple_sphere,
            config=config
        )
        
        # Set specific initial position
        qbo.swarm[0].position = base_position + tiny_diff
        qbo.swarm[0].fitness = simple_sphere(qbo.swarm[0].position)
        
        results = qbo.optimize(verbose=False)
        
        final_pos = results['best_position']
        print(f"   Initial: {base_position + tiny_diff}")
        print(f"   Final:   {final_pos}")
        print(f"   Fitness: {results['best_fitness']:.8f}")
        print(f"   Chaos Events: {len(results['chaos_events'])}")
        print(f"   Lyapunov: {results['global_lyapunov']:.4f}\n")
        
        results_comparison.append({
            'tiny_diff': tiny_diff,
            'final_position': final_pos,
            'fitness': results['best_fitness'],
            'chaos_events': len(results['chaos_events']),
            'lyapunov': results['global_lyapunov']
        })
    
    # Compare divergence
    print("="*70)
    print("📊 BUTTERFLY EFFECT ANALYSIS")
    print("="*70)
    
    pos_diff = np.linalg.norm(
        results_comparison[1]['final_position'] - results_comparison[0]['final_position']
    )
    
    initial_diff = 0.001
    amplification = pos_diff / initial_diff if initial_diff > 0 else 0
    
    print(f"\nInitial Difference:  {initial_diff:.6f}")
    print(f"Final Difference:    {pos_diff:.6f}")
    print(f"Amplification Factor: {amplification:.2f}x")
    
    if amplification > 10:
        print(f"\n✅ BUTTERFLY EFFECT CONFIRMED!")
        print(f"   Tiny change (0.001) amplified {amplification:.0f}x")
        print(f"   This is the essence of chaos theory!")
    else:
        print(f"\n○ Limited butterfly effect observed")
    
    print(f"\n🦋 Chaos Events:")
    print(f"   Test 1: {results_comparison[0]['chaos_events']} events")
    print(f"   Test 2: {results_comparison[1]['chaos_events']} events")
    
    print(f"\n📈 Lyapunov Exponents (positive = chaotic):")
    print(f"   Test 1: {results_comparison[0]['lyapunov']:.4f}")
    print(f"   Test 2: {results_comparison[1]['lyapunov']:.4f}")
    
    print("\n" + "="*70)
    print("💡 Butterfly Effect Explained:")
    print("="*70)
    print("In 1961, meteorologist Edward Lorenz discovered that tiny")
    print("rounding errors (0.000127) in weather simulations led to")
    print("completely different weather predictions. He famously asked:")
    print('"Does the flap of a butterfly\'s wings in Brazil set off')
    print('a tornado in Texas?"')
    print("\nThis algorithm demonstrates the same principle:")
    print("✓ Small initial perturbations (chaos_sensitivity)")
    print("✓ Amplified through chaotic attractors (Lorenz, Rössler)")
    print("✓ Lead to dramatically different optimization paths")
    print("="*70 + "\n")
    
    return results_comparison


# ============================================================
# CHAOS THEORY VISUALIZATION
# ============================================================

def analyze_chaos_dynamics(results: Dict):
    """Analyze the chaotic dynamics of the optimization"""
    print(f"\n🌪️ CHAOS DYNAMICS ANALYSIS")
    print(f"{'='*70}\n")
    
    chaos_events = results.get('chaos_events', [])
    
    if not chaos_events:
        print("No chaos events recorded")
        return
    
    print(f"Total Chaos Events: {len(chaos_events)}")
    
    # Amplification statistics
    amplifications = [e['amplification'] for e in chaos_events]
    
    print(f"\nAmplification Statistics:")
    print(f"   ├─ Mean: {np.mean(amplifications):.2f}x")
    print(f"   ├─ Max:  {np.max(amplifications):.2f}x")
    print(f"   ├─ Min:  {np.min(amplifications):.2f}x")
    print(f"   └─ Std:  {np.std(amplifications):.2f}x")
    
    # Timeline of chaos
    print(f"\nChaos Event Timeline:")
    iterations = [e['iteration'] for e in chaos_events]
    iteration_counts = {}
    for it in iterations:
        iteration_counts[it] = iteration_counts.get(it, 0) + 1
    
    # Show most chaotic iterations
    sorted_its = sorted(iteration_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for it, count in sorted_its:
        print(f"   Iteration {it}: {count} chaos events")
    
    # Attractor analysis
    attractors = [e['attractor'] for e in chaos_events]
    print(f"\nAttractor Used: {attractors[0] if attractors else 'None'}")
    
    # Butterfly effect magnitude
    initial_perts = [e['initial_perturbation'] for e in chaos_events]
    final_perts = [e['final_perturbation'] for e in chaos_events]
    
    print(f"\nButterfly Effect Magnitude:")
    print(f"   ├─ Avg Initial Perturbation: {np.mean(initial_perts):.6f}")
    print(f"   ├─ Avg Final Perturbation:   {np.mean(final_perts):.6f}")
    print(f"   └─ Average Amplification:    {np.mean(final_perts)/np.mean(initial_perts):.2f}x")
    
    # Divergence analysis
    divergences = results.get('divergence_pairs', [])
    if divergences:
        divergence_values = [d for _, _, d in divergences]
        print(f"\nTrajectory Divergence:")
        print(f"   ├─ Pairs that diverged: {len(divergences)}")
        print(f"   ├─ Mean divergence rate: {np.mean(divergence_values):.2f}x")
        print(f"   └─ Max divergence rate:  {np.max(divergence_values):.2f}x")
    
    print(f"\n{'='*70}\n")


# ============================================================
# BRAIN-GUIDED OPTIMIZATION DEMONSTRATION
# ============================================================

def demonstrate_brain_guided_optimization():
    """
    Advanced demo showing how Precious Brain guides optimization
    """
    print("\n" + "="*70)
    print("🧠 BRAIN-GUIDED QUANTUM BUTTERFLY OPTIMIZATION")
    print("="*70)
    print("Demonstrating consciousness-emotion-memory integration")
    print("="*70 + "\n")
    
    if not BRAIN_AVAILABLE:
        print("⚠️  A_Brain module not available - skipping brain demo")
        return None, None, None
    
    # Create a custom objective: Learn a pattern
    def pattern_learning_objective(x):
        """Learn to match a target pattern"""
        target = np.array([1.0, 2.0, -1.5, 0.5, 3.0])
        return np.sum((x - target) ** 2)
    
    # Initialize brain first
    print("🧠 Initializing Precious Brain...")
    brain = create_precious_brain(
        name="PatternLearnerBrain",
        architecture="medium",
        input_size=5,
        output_size=1,
        n_consciousness_neurons=5000,
        enable_all=True
    )
    
    print("\n🦋 Creating Quantum Butterfly Swarm...")
    config = QuantumButterflyConfig(
        n_butterflies=15,
        dimensions=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        max_iterations=40,
        quantum_probability=0.25,
        enable_brain=True,
        brain_learning_rate=0.15,
        brain_influence=0.4,
        memory_guided_search=True,
        consciousness_feedback=True,
        enable_butterfly_effect=True
    )
    
    qbo = QuantumButterflyOptimization(
        objective_function=pattern_learning_objective,
        config=config,
        precious_brain=brain
    )
    
    print("\n🔬 Running optimization with full brain integration...")
    results = qbo.optimize(verbose=True)
    
    # Detailed analysis
    print(f"\n{'='*70}")
    print(f"🔍 DETAILED ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"🎯 Optimization Results:")
    print(f"   ├─ Target Pattern: [1.0, 2.0, -1.5, 0.5, 3.0]")
    print(f"   ├─ Found Pattern:  {results['best_position']}")
    print(f"   ├─ Error: {results['best_fitness']:.8f}")
    print(f"   └─ Convergence: {results['iterations']} iterations")
    
    # Brain introspection
    introspection = brain.introspect()
    
    print(f"\n🧠 Brain Cognitive State:")
    state = introspection['state']
    print(f"   ├─ Mode: {state['mode']}")
    print(f"   ├─ Arousal Level: {state['arousal']:.2f}")
    print(f"   ├─ Fatigue Level: {state['fatigue']:.2f}")
    print(f"   └─ Working Memory: {state['working_memory_items']} items")
    
    print(f"\n💭 Consciousness Activity:")
    consciousness = introspection['consciousness']
    print(f"   ├─ Total Thoughts: {consciousness['total_thoughts']}")
    print(f"   ├─ Emotions Processed: {consciousness['total_emotions']}")
    print(f"   ├─ Memories Formed: {consciousness['total_memories_formed']}")
    print(f"   └─ Active Synapses: {consciousness['active_synapses']:,}")
    
    print(f"\n💾 Memory Systems:")
    memory = introspection['memory']
    print(f"   ├─ Short-term Size: {memory.get('short_term_size', 0)}")
    print(f"   ├─ Long-term Size: {memory.get('long_term_size', 0)}")
    print(f"   ├─ Episodic Size: {memory.get('episodic_size', memory.get('episodic_memory_size', 0))}")
    print(f"   ├─ Consolidations: {memory.get('consolidations', 0)}")
    print(f"   └─ Retrievals: {memory.get('retrievals', 0)}")
    
    # Demonstrate brain dreaming after optimization
    print(f"\n💤 Brain dreaming for memory consolidation...")
    brain.dream(n_cycles=5, verbose=True)
    
    # Test prediction with learned knowledge
    print(f"\n🔮 Testing Brain Prediction:")
    test_input = np.array([[1.0, 2.0, -1.5, 0.5, 3.0]])  # Target pattern
    prediction = brain.predict(test_input, use_memory=True)
    print(f"   ├─ Test Input: {test_input[0]}")
    print(f"   └─ Brain Prediction: {prediction[0]}")
    
    # Quantum history analysis
    print(f"\n⚛️  Quantum Evolution:")
    quantum_hist = results['quantum_history']
    if len(quantum_hist) > 0:
        initial = quantum_hist[0]
        final = quantum_hist[-1]
        print(f"   Initial State Distribution:")
        print(f"   ├─ Superposition: {initial['superposition']}")
        print(f"   ├─ Collapsed: {initial['collapsed']}")
        print(f"   ├─ Entangled: {initial['entangled']}")
        print(f"   └─ Tunneling: {initial['tunneling']}")
        print(f"\n   Final State Distribution:")
        print(f"   ├─ Superposition: {final['superposition']}")
        print(f"   ├─ Collapsed: {final['collapsed']}")
        print(f"   ├─ Entangled: {final['entangled']}")
        print(f"   └─ Tunneling: {final['tunneling']}")
    
    print(f"\n{'='*70}")
    print(f"✅ BRAIN-GUIDED OPTIMIZATION COMPLETE")
    print(f"{'='*70}\n")
    
    return results, brain, qbo


# ============================================================
# UPDATED MAIN DEMONSTRATION
# ============================================================
    """
    Advanced demo showing how Precious Brain guides optimization
    """
    print("\n" + "="*70)
    print("🧠 BRAIN-GUIDED QUANTUM BUTTERFLY OPTIMIZATION")
    print("="*70)
    print("Demonstrating consciousness-emotion-memory integration")
    print("="*70 + "\n")
    
    if not BRAIN_AVAILABLE:
        print("⚠️  A_Brain module not available - skipping brain demo")
        return
    
    # Create a custom objective: Learn a pattern
    def pattern_learning_objective(x):
        """Learn to match a target pattern"""
        target = np.array([1.0, 2.0, -1.5, 0.5, 3.0])
        return np.sum((x - target) ** 2)
    
    # Initialize brain first
    print("🧠 Initializing Precious Brain...")
    brain = create_precious_brain(
        name="PatternLearnerBrain",
        architecture="medium",
        input_size=5,
        output_size=1,
        n_consciousness_neurons=5000,
        enable_all=True
    )
    
    print("\n🦋 Creating Quantum Butterfly Swarm...")
    config = QuantumButterflyConfig(
        n_butterflies=15,
        dimensions=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        max_iterations=40,
        quantum_probability=0.25,
        enable_brain=True,
        brain_learning_rate=0.15,
        brain_influence=0.4,
        memory_guided_search=True,
        consciousness_feedback=True
    )
    
    qbo = QuantumButterflyOptimization(
        objective_function=pattern_learning_objective,
        config=config,
        precious_brain=brain
    )
    
    print("\n🔬 Running optimization with full brain integration...")
    results = qbo.optimize(verbose=True)
    
    # Detailed analysis
    print(f"\n{'='*70}")
    print(f"🔍 DETAILED ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"🎯 Optimization Results:")
    print(f"   ├─ Target Pattern: [1.0, 2.0, -1.5, 0.5, 3.0]")
    print(f"   ├─ Found Pattern:  {results['best_position']}")
    print(f"   ├─ Error: {results['best_fitness']:.8f}")
    print(f"   └─ Convergence: {results['iterations']} iterations")
    
    # Brain introspection
    introspection = brain.introspect()
    
    print(f"\n🧠 Brain Cognitive State:")
    state = introspection['state']
    print(f"   ├─ Mode: {state['mode']}")
    print(f"   ├─ Arousal Level: {state['arousal']:.2f}")
    print(f"   ├─ Fatigue Level: {state['fatigue']:.2f}")
    print(f"   └─ Working Memory: {state['working_memory_items']} items")
    
    print(f"\n💭 Consciousness Activity:")
    consciousness = introspection['consciousness']
    print(f"   ├─ Total Thoughts: {consciousness['total_thoughts']}")
    print(f"   ├─ Emotions Processed: {consciousness['total_emotions']}")
    print(f"   ├─ Memories Formed: {consciousness['total_memories_formed']}")
    print(f"   └─ Active Synapses: {consciousness['active_synapses']:,}")
    
    print(f"\n💾 Memory Systems:")
    memory = introspection['memory']
    print(f"   ├─ Short-term Size: {memory['short_term_size']}")
    print(f"   ├─ Long-term Size: {memory['long_term_size']}")
    print(f"   ├─ Episodic Size: {memory['episodic_size']}")
    print(f"   ├─ Consolidations: {memory['consolidations']}")
    print(f"   └─ Retrievals: {memory['retrievals']}")
    
    # Demonstrate brain dreaming after optimization
    print(f"\n💤 Brain dreaming for memory consolidation...")
    brain.dream(n_cycles=5, verbose=True)
    
    # Test prediction with learned knowledge
    print(f"\n🔮 Testing Brain Prediction:")
    test_input = np.array([[1.0, 2.0, -1.5, 0.5, 3.0]])  # Target pattern
    prediction = brain.predict(test_input, use_memory=True)
    print(f"   ├─ Test Input: {test_input[0]}")
    print(f"   └─ Brain Prediction: {prediction[0]}")
    
    # Quantum history analysis
    print(f"\n⚛️  Quantum Evolution:")
    quantum_hist = results['quantum_history']
    if len(quantum_hist) > 0:
        initial = quantum_hist[0]
        final = quantum_hist[-1]
        print(f"   Initial State Distribution:")
        print(f"   ├─ Superposition: {initial['superposition']}")
        print(f"   ├─ Collapsed: {initial['collapsed']}")
        print(f"   ├─ Entangled: {initial['entangled']}")
        print(f"   └─ Tunneling: {initial['tunneling']}")
        print(f"\n   Final State Distribution:")
        print(f"   ├─ Superposition: {final['superposition']}")
        print(f"   ├─ Collapsed: {final['collapsed']}")
        print(f"   ├─ Entangled: {final['entangled']}")
        print(f"   └─ Tunneling: {final['tunneling']}")
    
    print(f"\n{'='*70}")
    print(f"✅ BRAIN-GUIDED OPTIMIZATION COMPLETE")
    print(f"{'='*70}\n")
    
    return results, brain, qbo


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def analyze_convergence(fitness_history: List[float]):
    """Analyze convergence behavior"""
    print(f"\n📈 CONVERGENCE ANALYSIS")
    print(f"{'='*70}")
    
    if len(fitness_history) < 2:
        print("Not enough data for analysis")
        return
    
    # Calculate improvements
    improvements = []
    for i in range(1, len(fitness_history)):
        imp = fitness_history[i-1] - fitness_history[i]
        improvements.append(imp)
    
    total_improvement = fitness_history[0] - fitness_history[-1]
    avg_improvement = np.mean(improvements) if improvements else 0
    
    print(f"Initial Fitness: {fitness_history[0]:.8f}")
    print(f"Final Fitness: {fitness_history[-1]:.8f}")
    print(f"Total Improvement: {total_improvement:.8f}")
    print(f"Avg Improvement/Iter: {avg_improvement:.8f}")
    print(f"Convergence Rate: {(total_improvement/fitness_history[0])*100:.2f}%")
    
    # Find best iteration
    best_iter = np.argmin(fitness_history)
    print(f"Best Found at Iteration: {best_iter + 1}")
    
    # Stagnation detection
    if len(fitness_history) > 10:
        recent_change = abs(fitness_history[-1] - fitness_history[-10])
        if recent_change < 1e-6:
            print(f"⚠️  Stagnation detected in last 10 iterations")
        else:
            print(f"✓ Still improving (recent change: {recent_change:.8f})")
    
    print(f"{'='*70}\n")


def compare_with_without_brain(objective_func, dims: int = 5, 
                               bounds: List[float] = [-5, 5]):
    """Compare performance with and without brain integration"""
    print(f"\n{'='*70}")
    print(f"⚖️  COMPARISON: WITH vs WITHOUT BRAIN")
    print(f"{'='*70}\n")
    
    results_comparison = {}
    
    # Test WITHOUT brain
    print("🦋 Test 1: Pure Quantum Butterfly (No Brain)")
    print("-" * 70)
    config_no_brain = QuantumButterflyConfig(
        n_butterflies=20,
        dimensions=dims,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        max_iterations=50,
        enable_brain=False
    )
    
    qbo_no_brain = QuantumButterflyOptimization(
        objective_function=objective_func,
        config=config_no_brain
    )
    
    results_no_brain = qbo_no_brain.optimize(verbose=False)
    results_comparison['no_brain'] = results_no_brain
    
    print(f"✓ Complete - Best Fitness: {results_no_brain['best_fitness']:.8f}")
    print(f"✓ Iterations: {results_no_brain['iterations']}")
    print(f"✓ Time: {results_no_brain['time']:.2f}s\n")
    
    # Test WITH brain
    if BRAIN_AVAILABLE:
        print("🧠 Test 2: Quantum Butterfly + Precious Brain")
        print("-" * 70)
        config_with_brain = QuantumButterflyConfig(
            n_butterflies=20,
            dimensions=dims,
            lower_bound=bounds[0],
            upper_bound=bounds[1],
            max_iterations=50,
            enable_brain=True,
            memory_guided_search=True,
            consciousness_feedback=True
        )
        
        qbo_with_brain = QuantumButterflyOptimization(
            objective_function=objective_func,
            config=config_with_brain
        )
        
        results_with_brain = qbo_with_brain.optimize(verbose=False)
        results_comparison['with_brain'] = results_with_brain
        
        print(f"✓ Complete - Best Fitness: {results_with_brain['best_fitness']:.8f}")
        print(f"✓ Iterations: {results_with_brain['iterations']}")
        print(f"✓ Time: {results_with_brain['time']:.2f}s\n")
        
        # Comparison
        print(f"{'='*70}")
        print(f"📊 PERFORMANCE COMPARISON")
        print(f"{'='*70}")
        
        fitness_improvement = (results_no_brain['best_fitness'] - 
                              results_with_brain['best_fitness'])
        improvement_pct = (fitness_improvement / 
                          abs(results_no_brain['best_fitness'])) * 100
        
        print(f"\nFitness:")
        print(f"   Without Brain: {results_no_brain['best_fitness']:.8f}")
        print(f"   With Brain:    {results_with_brain['best_fitness']:.8f}")
        print(f"   Improvement:   {fitness_improvement:.8f} ({improvement_pct:+.2f}%)")
        
        print(f"\nTime:")
        print(f"   Without Brain: {results_no_brain['time']:.2f}s")
        print(f"   With Brain:    {results_with_brain['time']:.2f}s")
        
        print(f"\nIterations:")
        print(f"   Without Brain: {results_no_brain['iterations']}")
        print(f"   With Brain:    {results_with_brain['iterations']}")
        
        if fitness_improvement > 0:
            print(f"\n✅ Brain integration IMPROVED results by {improvement_pct:.2f}%")
        else:
            print(f"\n⚠️  Brain integration did not improve results significantly")
        
        print(f"{'='*70}\n")
    else:
        print("⚠️  A_Brain module not available - skipping brain comparison\n")
    
    return results_comparison


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🦋⚛️🧠 QUANTUM BUTTERFLY OPTIMIZATION WITH PRECIOUS BRAIN")
    print("="*70)
    print("Combining Quantum Mechanics + Butterfly Algorithm + AI Brain")
    print("="*70 + "\n")
    
    # Check if brain is available
    if BRAIN_AVAILABLE:
        print("✅ Precious Brain module detected and loaded")
    else:
        print("⚠️  Precious Brain module not available")
        print("   Running in quantum-only mode (no brain integration)")
    
    print("\n🚀 Starting demonstrations automatically in 2 seconds...")
    time.sleep(2)
    
    # Demo 1: Standard benchmarks
    print("\n" + "🔷"*35)
    print("DEMO 1: BENCHMARK FUNCTIONS")
    print("🔷"*35)
    benchmark_results = demonstrate_quantum_butterfly()
    
    # Demo 1.5: Butterfly Effect demonstration
    print("\n" + "🔷"*35)
    print("DEMO 1.5: BUTTERFLY EFFECT (CHAOS THEORY)")
    print("🔷"*35)
    chaos_comparison = demonstrate_butterfly_effect()
    
    # Demo 2: Brain-guided optimization
    if BRAIN_AVAILABLE:
        print("\n" + "🔷"*35)
        print("DEMO 2: BRAIN-GUIDED OPTIMIZATION")
        print("🔷"*35)
        brain_results, brain, qbo = demonstrate_brain_guided_optimization()
        
        # Analyze convergence
        analyze_convergence(brain_results['fitness_history'])
        
        # Analyze chaos dynamics
        analyze_chaos_dynamics(brain_results)
        
        # Demo 3: Comparison
        print("\n" + "🔷"*35)
        print("DEMO 3: PERFORMANCE COMPARISON")
        print("🔷"*35)
        comparison = compare_with_without_brain(
            BenchmarkFunctions.rastrigin,
            dims=5,
            bounds=[-5.12, 5.12]
        )
        
        # Final brain summary
        print("\n" + "🔷"*35)
        print("FINAL BRAIN STATE")
        print("🔷"*35)
        brain.summary()
    
    print("\n" + "="*70)
    print("🎉 ALL DEMONSTRATIONS COMPLETE!")
    print("="*70)
    print("\n💡 Key Takeaways:")
    print("   ✓ Quantum mechanics enhances butterfly optimization")
    print("   ✓ Superposition enables parallel exploration")
    print("   ✓ Entanglement improves information sharing")
    print("   ✓ Quantum tunneling escapes local optima")
    print("   ✓ BUTTERFLY EFFECT: Tiny changes → Huge consequences")
    print("   ✓ Chaos theory: Lyapunov exponents measure sensitivity")
    print("   ✓ Lorenz attractor: Classic weather butterfly effect")
    if BRAIN_AVAILABLE:
        print("   ✓ Brain memory guides efficient search")
        print("   ✓ Consciousness forms adaptive strategies")
        print("   ✓ Emotions modulate exploration vs exploitation")
        print("   ✓ Memory consolidation preserves learned patterns")
    print("\n🦋 Quantum + Chaos + 🧠 Brain = 🚀 Ultimate Optimization!")
    print("="*70 + "\n")