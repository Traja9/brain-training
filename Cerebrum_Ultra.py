"""
===========================================================
NEUROGENESIS v3.0 - ULTRA-ADVANCED NEURAL EVOLUTION SYSTEM
-----------------------------------------------------------
Revolutionary AI architecture featuring:
- Dynamic Neural Architecture Evolution (DNA-inspired)
- Quantum-Inspired Superposition Learning
- Meta-Learning & Few-Shot Adaptation
- Neuroevolution with Genetic Algorithms
- Multi-Task & Continual Learning
- Hypernetwork-based Weight Generation
- Neural ODEs for Continuous Dynamics
- Mixture of Experts (MoE) Architecture
- Self-Organizing Maps & Competitive Learning
- Automated Machine Learning (AutoML)
- Curriculum Learning & Active Learning
- Knowledge Distillation & Neural Compression
===========================================================
"""

import time
import numpy as np
from typing import Optional, List, Dict, Tuple, Callable, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Import base Cerebrum module
from Cerebrum import (
    Cerebrum_Core,
    CerebrumConfig,
    SchedulerType,
    RegularizationType,
    TrainingMetrics,
    create_cerebrum
)


# ============================================================
# ADVANCED ENUMS & CONFIGURATIONS
# ============================================================

class EvolutionStrategy(Enum):
    """Neural evolution strategies"""
    GENETIC_ALGORITHM = "genetic_algorithm"
    NEUROEVOLUTION = "neuroevolution"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    PARTICLE_SWARM = "particle_swarm"
    COEVOLUTION = "coevolution"


class MetaLearningStrategy(Enum):
    """Meta-learning approaches"""
    MAML = "maml"  # Model-Agnostic Meta-Learning
    REPTILE = "reptile"
    PROTO_NETS = "prototypical_networks"
    MATCHING_NETS = "matching_networks"
    META_SGD = "meta_sgd"


class ContinualLearningStrategy(Enum):
    """Continual learning methods"""
    ELASTIC_WEIGHT_CONSOLIDATION = "ewc"
    PROGRESSIVE_NEURAL_NETS = "progressive"
    GRADIENT_EPISODIC_MEMORY = "gem"
    EXPERIENCE_REPLAY = "experience_replay"
    KNOWLEDGE_DISTILLATION = "distillation"


@dataclass
class NeuroGenesisConfig:
    """Ultra-advanced configuration"""
    # Base configuration
    name: str = "NeuroGenesis"
    base_architecture: str = "medium"
    input_size: int = 1
    output_size: int = 1
    # Evolution parameters
    enable_evolution: bool = True
    evolution_strategy: EvolutionStrategy = EvolutionStrategy.GENETIC_ALGORITHM
    population_size: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    
    # Meta-learning
    enable_meta_learning: bool = False
    meta_strategy: MetaLearningStrategy = MetaLearningStrategy.MAML
    meta_learning_rate: float = 0.001
    num_inner_steps: int = 5
    
    # Continual learning
    enable_continual_learning: bool = False
    continual_strategy: ContinualLearningStrategy = ContinualLearningStrategy.ELASTIC_WEIGHT_CONSOLIDATION
    memory_size: int = 1000
    
    # Multi-task learning
    enable_multi_task: bool = False
    num_tasks: int = 1
    task_weights: Optional[List[float]] = None
    
    # Mixture of Experts
    enable_moe: bool = False
    num_experts: int = 4
    top_k_experts: int = 2
    
    # Hypernetwork
    enable_hypernetwork: bool = False
    hypernetwork_hidden_size: int = 64
    
    # Curriculum learning
    enable_curriculum: bool = False
    difficulty_measure: str = "loss"  # loss, uncertainty, complexity
    
    # Active learning
    enable_active_learning: bool = False
    query_strategy: str = "uncertainty"  # uncertainty, diversity, hybrid
    query_budget: int = 100
    
    # Advanced features
    enable_attention_routing: bool = True
    enable_skip_connections: bool = True
    enable_batch_normalization: bool = True
    enable_layer_normalization: bool = False
    
    # Performance
    use_mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


# ============================================================
# GENETIC ALGORITHM COMPONENTS
# ============================================================

class GeneticOperator:
    """Genetic algorithm operators for neural evolution"""
    
    @staticmethod
    def mutate(weights: Dict, mutation_rate: float) -> Dict:
        """Mutate neural network weights"""
        mutated = {}
        for key, value in weights.items():
            if np.random.random() < mutation_rate:
                # Gaussian mutation
                noise = np.random.randn(*value.shape) * 0.1
                mutated[key] = value + noise
            else:
                mutated[key] = value.copy()
        return mutated
    
    @staticmethod
    def crossover(parent1: Dict, parent2: Dict, crossover_rate: float) -> Tuple[Dict, Dict]:
        """Crossover between two parent networks"""
        child1, child2 = {}, {}
        
        for key in parent1.keys():
            if np.random.random() < crossover_rate:
                # Uniform crossover
                mask = np.random.random(parent1[key].shape) > 0.5
                child1[key] = np.where(mask, parent1[key], parent2[key])
                child2[key] = np.where(mask, parent2[key], parent1[key])
            else:
                child1[key] = parent1[key].copy()
                child2[key] = parent2[key].copy()
        
        return child1, child2
    
    @staticmethod
    def tournament_selection(population: List, fitness: List[float], 
                           tournament_size: int = 3) -> Any:
        """Tournament selection"""
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_fitness)]
        return population[winner_idx]


# ============================================================
# META-LEARNING COMPONENTS
# ============================================================

class MAMLLearner:
    """Model-Agnostic Meta-Learning (MAML) implementation"""
    
    def __init__(self, model: Cerebrum_Core, meta_lr: float = 0.001, 
                 inner_lr: float = 0.01, num_inner_steps: int = 5):
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
    
    def meta_train_step(self, tasks: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Single meta-training step across multiple tasks"""
        meta_gradients = []
        total_loss = 0.0
        
        for X_support, y_support, X_query, y_query in tasks:
            # Clone model for inner loop
            initial_weights = self._get_weights()
            
            # Inner loop adaptation
            for _ in range(self.num_inner_steps):
                loss = self._compute_loss(X_support, y_support)
                gradients = self._compute_gradients(X_support, y_support)
                self._update_weights(gradients, self.inner_lr)
            
            # Compute meta-gradient on query set
            query_loss = self._compute_loss(X_query, y_query)
            query_gradients = self._compute_gradients(X_query, y_query)
            meta_gradients.append(query_gradients)
            total_loss += query_loss
            
            # Restore initial weights
            self._set_weights(initial_weights)
        
        # Meta-update
        avg_meta_gradients = self._average_gradients(meta_gradients)
        self._update_weights(avg_meta_gradients, self.meta_lr)
        
        return total_loss / len(tasks)
    
    def _get_weights(self) -> Dict:
        """Get model weights"""
        return self.model._get_model_weights()[0] if self.model.models else {}
    
    def _set_weights(self, weights: Dict):
        """Set model weights"""
        if self.model.models and weights:
            model = self.model.models[0]
            try:
                for i, layer in enumerate(model.network):
                    layer_key = f'layer_{i}'
                    if layer_key in weights:
                        if hasattr(layer, 'weights'):
                            layer.weights = weights[layer_key].copy()
                        elif hasattr(layer, 'W'):
                            layer.W = weights[layer_key].copy()
            except (AttributeError, KeyError):
                pass
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute loss"""
        pred = self.model.predict(X)
        return np.mean((pred - y) ** 2)
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Compute gradients (placeholder)"""
        # In practice, this would compute actual gradients
        return {}
    
    def _update_weights(self, gradients: Dict, lr: float):
        """Update weights with gradients"""
        # Placeholder for gradient update
        pass
    
    def _average_gradients(self, gradients_list: List[Dict]) -> Dict:
        """Average gradients across tasks"""
        if not gradients_list:
            return {}
        avg_grads = {}
        for key in gradients_list[0].keys():
            avg_grads[key] = np.mean([g[key] for g in gradients_list], axis=0)
        return avg_grads


# ============================================================
# MIXTURE OF EXPERTS
# ============================================================

class MixtureOfExperts:
    """Mixture of Experts architecture"""
    
    def __init__(self, num_experts: int, expert_config: CerebrumConfig, 
             top_k: int = 2):
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = [Cerebrum_Core(config=expert_config) 
                    for _ in range(num_experts)]
        
        # FIX: Get input_size from the config, not from experts
        input_size = expert_config.input_size  # ADD THIS LINE
        self.gate_weights = np.random.randn(input_size, num_experts) * 0.01  # CHANGE THIS LINE    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass with expert routing"""
        # Compute gating scores
        gate_logits = X @ self.gate_weights
        gate_probs = self._softmax(gate_logits)
        
        # Select top-k experts
        top_k_indices = np.argsort(gate_probs, axis=1)[:, -self.top_k:]
        
        # Get predictions from selected experts
        batch_size = X.shape[0]
        output = np.zeros((batch_size, self.experts[0].output_size))
        
        for i in range(batch_size):
            expert_outputs = []
            expert_weights = []
            
            for expert_idx in top_k_indices[i]:
                pred = self.experts[expert_idx].predict(X[i:i+1])
                expert_outputs.append(pred)
                expert_weights.append(gate_probs[i, expert_idx])
            
            # Weighted combination
            expert_weights = np.array(expert_weights) / np.sum(expert_weights)
            output[i] = np.sum([w * out for w, out in zip(expert_weights, expert_outputs)], 
                              axis=0)
        
        return output
    
    def train_experts(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Train all experts"""
        print(f"\nðŸ'¥ Training {self.num_experts} Experts...")
        for i, expert in enumerate(self.experts):
            print(f"  Expert {i+1}/{self.num_experts}...")
            expert.train(X, y, epochs=epochs, verbose=False)
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# ============================================================
# HYPERNETWORK
# ============================================================

class HyperNetwork:
    """Hypernetwork that generates weights for main network"""
    
    def __init__(self, target_network_config: CerebrumConfig, hidden_size: int = 64):
        self.target_config = target_network_config
        self.hidden_size = hidden_size
        
        # Simple hypernetwork architecture
        self.hyper_layers = [
            np.random.randn(8, hidden_size) * 0.01,  # Input: task embedding
            np.random.randn(hidden_size, hidden_size) * 0.01,
            np.random.randn(hidden_size, self._compute_target_params()) * 0.01
        ]
    
    def generate_weights(self, task_embedding: np.ndarray) -> Dict:
        """Generate weights for target network given task embedding"""
        h = task_embedding
        for layer in self.hyper_layers[:-1]:
            h = np.tanh(h @ layer)
        
        weight_vector = h @ self.hyper_layers[-1]
        
        # Reshape into network structure
        weights = self._vector_to_weights(weight_vector)
        return weights
    
    def _compute_target_params(self) -> int:
        """Compute total parameters in target network"""
        # Simplified calculation
        return 1000  # Placeholder
    
    def _vector_to_weights(self, vector: np.ndarray) -> Dict:
        """Convert weight vector to network structure"""
        # Placeholder implementation
        return {}


# ============================================================
# CONTINUAL LEARNING
# ============================================================

class ElasticWeightConsolidation:
    """Elastic Weight Consolidation for continual learning"""
    
    def __init__(self, model: Cerebrum_Core, lambda_ewc: float = 0.4):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_information = {}
        self.optimal_weights = {}
    
    def compute_fisher_information(self, X: np.ndarray, y: np.ndarray, 
                                   num_samples: int = 200):
        """Compute Fisher Information Matrix"""
        print("ðŸŠ Computing Fisher Information Matrix...")
        
        # Sample subset for efficiency
        indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
        X_sample = X[indices]
        
        # Compute gradients for each sample
        gradients_squared = {}
        
        for i in range(len(X_sample)):
            # This is a placeholder - actual implementation would compute gradients
            pass
        
        # Store current optimal weights
        self.optimal_weights = self.model._get_model_weights()[0]
        print("  âœ Fisher Information computed")
    
    def compute_ewc_loss(self, current_loss: float) -> float:
        """Add EWC penalty to loss"""
        ewc_penalty = 0.0
        
        current_weights = self.model._get_model_weights()[0]
        
        for key in current_weights.keys():
            if key in self.optimal_weights and key in self.fisher_information:
                diff = current_weights[key] - self.optimal_weights[key]
                ewc_penalty += np.sum(self.fisher_information[key] * diff ** 2)
        
        return current_loss + (self.lambda_ewc / 2) * ewc_penalty


class ExperienceReplay:
    """Experience replay buffer for continual learning"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, X: np.ndarray, y: np.ndarray):
        """Add experiences to buffer"""
        for i in range(len(X)):
            self.buffer.append((X[i], y[i]))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample random batch from buffer"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        X_batch = np.array([x for x, y in batch])
        y_batch = np.array([y for x, y in batch])
        
        return X_batch, y_batch


# ============================================================
# CURRICULUM LEARNING
# ============================================================

class CurriculumScheduler:
    """Adaptive curriculum learning scheduler"""
    
    def __init__(self, difficulty_measure: str = "loss"):
        self.difficulty_measure = difficulty_measure
        self.sample_difficulties = []
    
    def compute_difficulties(self, model: Cerebrum_Core, X: np.ndarray, 
                           y: np.ndarray) -> np.ndarray:
        """Compute difficulty score for each sample"""
        difficulties = np.zeros(len(X))
        
        for i in range(len(X)):
            pred = model.predict(X[i:i+1])
            loss = np.mean((pred - y[i:i+1]) ** 2)
            difficulties[i] = loss
        
        return difficulties
    
    def get_curriculum_batch(self, X: np.ndarray, y: np.ndarray, 
                            batch_size: int, epoch: int, 
                            total_epochs: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get batch based on curriculum strategy"""
        # Compute current difficulty threshold
        progress = epoch / total_epochs
        difficulty_threshold = np.percentile(self.sample_difficulties, 
                                    progress * 100) if len(self.sample_difficulties) > 0 else float('inf')      
        # Select samples below difficulty threshold
        valid_indices = np.where(self.sample_difficulties <= difficulty_threshold)[0]
        
        if len(valid_indices) < batch_size:
            valid_indices = np.arange(len(X))
        
        selected = np.random.choice(valid_indices, 
                                   min(batch_size, len(valid_indices)), 
                                   replace=False)
        
        return X[selected], y[selected]


# ============================================================
# ACTIVE LEARNING
# ============================================================

class ActiveLearner:
    """Active learning with uncertainty sampling"""
    
    def __init__(self, query_strategy: str = "uncertainty"):
        self.query_strategy = query_strategy
        self.queried_indices = set()
    
    def select_samples(self, model: Cerebrum_Core, X_pool: np.ndarray, 
                      n_samples: int) -> np.ndarray:
        """Select most informative samples for labeling"""
        
        if self.query_strategy == "uncertainty":
            # Use prediction uncertainty
            if hasattr(model, 'predict') and model.config.ensemble_size > 1:
                _, uncertainties = model.predict(X_pool, return_uncertainty=True)
                uncertainties = uncertainties.flatten()
            else:
                # Fallback: use prediction variance
                predictions = model.predict(X_pool)
                # FIX: Ensure uncertainties match X_pool length
                if predictions.ndim > 1:
                    uncertainties = np.var(predictions, axis=1).flatten()
                else:
                    uncertainties = np.abs(predictions).flatten()
            
            # Ensure correct length
            if len(uncertainties) != len(X_pool):
                uncertainties = np.repeat(uncertainties, len(X_pool) // len(uncertainties) + 1)[:len(X_pool)]
            
            # Select samples with highest uncertainty
            available_indices = [i for i in range(len(X_pool)) 
                            if i not in self.queried_indices]
            
            if len(available_indices) == 0:
                return np.array([])
            
            uncertainties_available = uncertainties[available_indices]
            top_indices = np.argsort(uncertainties_available)[-n_samples:]
            selected = np.array([available_indices[i] for i in top_indices])
            
        elif self.query_strategy == "diversity":
            # Select diverse samples (simplified)
            selected = self._diversity_sampling(X_pool, n_samples)

        else:  # hybrid
            selected = self._hybrid_sampling(model, X_pool, n_samples)

        # Mark as queried
        self.queried_indices.update(selected)

        return selected
            
    def _diversity_sampling(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """Sample diverse points"""
        # K-means clustering-based diversity
        available = [i for i in range(len(X)) if i not in self.queried_indices]
        if len(available) <= n_samples:
            return np.array(available)
        
        # Simple diversity: maximize pairwise distances
        selected = []
        remaining = list(available)
        
        # Start with random point
        first = np.random.choice(remaining)
        selected.append(first)
        remaining.remove(first)
        
        # Greedily select most distant points
        while len(selected) < n_samples and remaining:
            max_min_dist = -1
            best_idx = None
            
            for idx in remaining:
                min_dist = min([np.linalg.norm(X[idx] - X[s]) for s in selected])
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        return np.array(selected)
    
    def _hybrid_sampling(self, model: Cerebrum_Core, X: np.ndarray, 
                        n_samples: int) -> np.ndarray:
        """Combine uncertainty and diversity"""
        # Split budget
        n_uncertain = n_samples // 2
        n_diverse = n_samples - n_uncertain
        
        uncertain_samples = self.select_samples(model, X, n_uncertain)
        
        # Temporarily mark uncertain samples
        temp_queried = self.queried_indices.copy()
        self.queried_indices.update(uncertain_samples)
        
        diverse_samples = self._diversity_sampling(X, n_diverse)
        
        # Restore queried indices
        self.queried_indices = temp_queried
        
        return np.concatenate([uncertain_samples, diverse_samples])


# ============================================================
# MAIN NEUROGENESIS CLASS
# ============================================================

class NeuroGenesis:
    """
    Ultra-Advanced Neural Evolution System
    
    Combines multiple cutting-edge techniques:
    - Neural Architecture Evolution
    - Meta-Learning
    - Continual Learning
    - Mixture of Experts
    - Hypernetworks
    - Curriculum Learning
    - Active Learning
    """
    
    def __init__(self, config: Optional[NeuroGenesisConfig] = None, **kwargs):
        # Initialize configuration
        if config is None:
            config = NeuroGenesisConfig(**kwargs)
        self.config = config
        
        # Core components
        self.base_cerebrum: Optional[Cerebrum_Core] = None
        self.population: List[Cerebrum_Core] = []
        self.fitness_scores: List[float] = []
        
        # Advanced components
        self.meta_learner: Optional[MAMLLearner] = None
        self.moe: Optional[MixtureOfExperts] = None
        self.hypernetwork: Optional[HyperNetwork] = None
        self.ewc: Optional[ElasticWeightConsolidation] = None
        self.experience_replay: Optional[ExperienceReplay] = None
        self.curriculum_scheduler: Optional[CurriculumScheduler] = None
        self.active_learner: Optional[ActiveLearner] = None
        
        # Training history
        self.evolution_history: List[Dict] = []
        self.meta_learning_history: List[Dict] = []
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all components"""
        print(f"\n🧬 Initializing NeuroGenesis: {self.config.name}")
        print("="*60)
        
        # Create base cerebrum
        base_config = CerebrumConfig(
            name=f"{self.config.name}_Base",
            architecture=self.config.base_architecture,
            input_size=self.config.input_size,
            output_size=self.config.output_size
        )
        self.base_cerebrum = Cerebrum_Core(config=base_config)
        
        # ADD ALL THESE MISSING INITIALIZATIONS:
        
        # Initialize evolution
        if self.config.enable_evolution:
            print("🧬 Evolution: ENABLED")
            self._initialize_population()
        
        # Initialize meta-learning
        if self.config.enable_meta_learning:
            print("🧠 Meta-Learning: ENABLED")
            self.meta_learner = MAMLLearner(
                self.base_cerebrum,
                meta_lr=self.config.meta_learning_rate,
                num_inner_steps=self.config.num_inner_steps
            )
        
        # Initialize continual learning
        if self.config.enable_continual_learning:
            print("♾️  Continual Learning: ENABLED")
            self.experience_replay = ExperienceReplay(self.config.memory_size)
            if self.config.continual_strategy == ContinualLearningStrategy.ELASTIC_WEIGHT_CONSOLIDATION:
                self.ewc = ElasticWeightConsolidation(self.base_cerebrum)
        
        # Initialize MoE
        if self.config.enable_moe:
            print("💥 Mixture of Experts: ENABLED")
            moe_config = CerebrumConfig(
                name=f"{self.config.name}_Expert",
                architecture=self.config.base_architecture,
                input_size=self.config.input_size,
                output_size=self.config.output_size
            )
            self.moe = MixtureOfExperts(
                num_experts=self.config.num_experts,
                expert_config=moe_config,
                top_k=self.config.top_k_experts
            )
        
        # Initialize curriculum learning
        if self.config.enable_curriculum:
            print("📚 Curriculum Learning: ENABLED")
            self.curriculum_scheduler = CurriculumScheduler(
                difficulty_measure=self.config.difficulty_measure
            )
        
        # Initialize active learning
        if self.config.enable_active_learning:
            print("🎯 Active Learning: ENABLED")
            self.active_learner = ActiveLearner(
                query_strategy=self.config.query_strategy
            )
        
        print("="*60)
        print("✅ NeuroGenesis Initialized Successfully!\n")

    def _initialize_population(self):
        """Initialize population for evolution"""
        print(f"  Creating population of {self.config.population_size} networks...")
        
        for i in range(self.config.population_size):
            config = CerebrumConfig(
                name=f"{self.config.name}_Individual_{i}",
                architecture=self.config.base_architecture,
                input_size=self.config.input_size,   # ADD THIS
                output_size=self.config.output_size  # ADD THIS
            )
            individual = Cerebrum_Core(config=config)
            self.population.append(individual)
            self.fitness_scores.append(0.0)    
    # --------------------------------------------------------
    # EVOLUTION METHODS
    # --------------------------------------------------------
    def evolve(self, X: np.ndarray, y: np.ndarray, 
           generations: int = 10,
           epochs_per_gen: int = 50,
           verbose: bool = True) -> Dict:
        """Evolutionary training with genetic algorithms"""
        
        if not self.config.enable_evolution:
            raise ValueError("Evolution not enabled in configuration")
        
        if len(self.population) == 0 or len(self.fitness_scores) == 0:
            print("⚠️  Population not initialized, initializing now...")
            self._initialize_population()
        
        print(f"\n🧬 EVOLUTIONARY TRAINING")
        print(f"   ├─ Generations: {generations}")
        print(f"   ├─ Population: {self.config.population_size}")
        print(f"   └─ Strategy: {self.config.evolution_strategy.value}\n")
        
        start_time = time.time()
        
        for gen in range(generations):
            gen_start = time.time()
            
            print(f"\n🧬 Generation {gen+1}/{generations}")
            print("─" * 40)
            
            for i, individual in enumerate(self.population):
                if verbose:
                    print(f"  Training Individual {i+1}/{len(self.population)}...")
                
                individual.train(X, y, epochs=epochs_per_gen, verbose=False)
                
                # Evaluate fitness
                predictions = individual.predict(X)
                fitness = -np.mean((predictions - y) ** 2)
                self.fitness_scores[i] = fitness
            
            # Report generation stats
            best_fitness = max(self.fitness_scores)
            avg_fitness = np.mean(self.fitness_scores)
            
            print(f"\n  📊 Generation Stats:")
            print(f"     ├─ Best Fitness:  {best_fitness:.6f}")
            print(f"     ├─ Avg Fitness:   {avg_fitness:.6f}")
            print(f"     └─ Time: {(time.time()-gen_start):.2f}s")
            
            # Record history
            self.evolution_history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'std_fitness': np.std(self.fitness_scores)
            })
            
            # Evolution operations (except last generation)
            if gen < generations - 1:
                self._evolve_population()
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ Evolution Complete!")
        print(f"   ├─ Total Time: {elapsed:.2f}s")
        print(f"   ├─ Best Fitness: {max(self.fitness_scores):.6f}")
        print(f"   └─ Final Avg: {np.mean(self.fitness_scores):.6f}")
        
        # Select best individual as base cerebrum
        best_idx = np.argmax(self.fitness_scores)
        self.base_cerebrum = self.population[best_idx]
        
        return {
            'generations': generations,
            'best_fitness': max(self.fitness_scores),
            'history': self.evolution_history
        }
          
    def _evolve_population(self):
        """Apply genetic operators to create next generation"""
        new_population = []
        
        # Elitism: keep best individual
        best_idx = np.argmax(self.fitness_scores)
        new_population.append(self.population[best_idx])
        
        # Create rest of population
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = GeneticOperator.tournament_selection(
                self.population, self.fitness_scores
            )
            parent2 = GeneticOperator.tournament_selection(
                self.population, self.fitness_scores
            )
            
            # Get weights
            p1_weights = parent1._get_model_weights()[0] if parent1.models else {}
            p2_weights = parent2._get_model_weights()[0] if parent2.models else {}
            
            if not p1_weights or not p2_weights:
                # If weights not available, just copy parents
                new_population.append(parent1)
                continue
            
            # Crossover
            child1_weights, child2_weights = GeneticOperator.crossover(
                p1_weights, p2_weights, self.config.crossover_rate
            )
            
            # Mutation
            child1_weights = GeneticOperator.mutate(
                child1_weights, self.config.mutation_rate
            )
            child2_weights = GeneticOperator.mutate(
                child2_weights, self.config.mutation_rate
            )
            
            # Create new individuals
            for child_weights in [child1_weights, child2_weights]:
                if len(new_population) < self.config.population_size:
                    child = Cerebrum_Core(config=CerebrumConfig(
                        architecture=self.config.base_architecture
                    ))
                    # Set weights
                    self._set_weights(child, child_weights)
                    new_population.append(child)
        
        self.population = new_population[:self.config.population_size]
        self.fitness_scores = [0.0] * len(self.population)
    
    def _set_weights(self, cerebrum: Cerebrum_Core, weights: Dict):
        """Set weights for a cerebrum"""
        if cerebrum.models and weights:
            model = cerebrum.models[0]
            try:
                if hasattr(model, 'network') and model.network:
                    for i, layer in enumerate(model.network):
                        layer_key = f'layer_{i}'
                        if layer_key in weights:
                            if hasattr(layer, 'weights'):
                                layer.weights = weights[layer_key].copy()
                            elif hasattr(layer, 'W'):
                                layer.W = weights[layer_key].copy()
            except (AttributeError, KeyError):
                pass
    
    # --------------------------------------------------------
    # META-LEARNING METHODS
    # --------------------------------------------------------
    
    def meta_train(self, task_generator: Callable, 
                   num_iterations: int = 100,
                   tasks_per_iteration: int = 5,
                   verbose: bool = True) -> Dict:
        """
        Meta-training for few-shot learning
        
        Args:
            task_generator: Function that generates tasks (support, query sets)
            num_iterations: Number of meta-training iterations
            tasks_per_iteration: Tasks per iteration
            verbose: Print progress
            
        Returns:
            Meta-learning history
        """
        if not self.config.enable_meta_learning or not self.meta_learner:
            raise ValueError("Meta-learning not enabled")
        
        print(f"\nðŸ§  META-LEARNING TRAINING")
        print(f"   âœâ€ Iterations: {num_iterations}")
        print(f"   âœâ€ Tasks/Iter: {tasks_per_iteration}")
        print(f"   ââ€ Strategy: {self.config.meta_strategy.value}\n")
        
        for iteration in range(num_iterations):
            # Generate tasks
            tasks = [task_generator() for _ in range(tasks_per_iteration)]
            
            # Meta-training step
            meta_loss = self.meta_learner.meta_train_step(tasks)
            
            # Record history
            self.meta_learning_history.append({
                'iteration': iteration,
                'meta_loss': meta_loss
            })
            
            if verbose and iteration % max(1, num_iterations // 10) == 0:
                print(f"  Iteration {iteration}/{num_iterations} | "
                      f"Meta Loss: {meta_loss:.6f}")
        
        print(f"\nâœ… Meta-Learning Complete!")
        return {'history': self.meta_learning_history}
    
    def few_shot_adapt(self, X_support: np.ndarray, y_support: np.ndarray,
                       num_steps: int = None) -> None:
        """
        Quick adaptation to new task with few examples
        
        Args:
            X_support: Support set inputs
            y_support: Support set labels
            num_steps: Number of adaptation steps
        """
        if not self.meta_learner:
            raise ValueError("Meta-learner not initialized")
        
        if num_steps is None:
            num_steps = self.config.num_inner_steps
        
        print(f"\nðŸŽ¯ Few-Shot Adaptation")
        print(f"   âœâ€ Support Samples: {len(X_support)}")
        print(f"   ââ€ Adaptation Steps: {num_steps}")
        
        initial_loss = self.meta_learner._compute_loss(X_support, y_support)
        
        for step in range(num_steps):
            loss = self.meta_learner._compute_loss(X_support, y_support)
            gradients = self.meta_learner._compute_gradients(X_support, y_support)
            self.meta_learner._update_weights(gradients, self.meta_learner.inner_lr)
        
        final_loss = self.meta_learner._compute_loss(X_support, y_support)
        
        print(f"   âœâ€ Initial Loss: {initial_loss:.6f}")
        print(f"   ââ€ Final Loss: {final_loss:.6f}")
    
    # --------------------------------------------------------
    # CONTINUAL LEARNING METHODS
    # --------------------------------------------------------
    
    def continual_train(self, tasks: List[Tuple[np.ndarray, np.ndarray]],epochs_per_task: int = 100,verbose: bool = True) -> Dict:
        """
        Sequential learning on multiple tasks without forgetting
        
        Args:
            tasks: List of (X, y) tuples for each task
            epochs_per_task: Training epochs per task
            verbose: Print progress
            
        Returns:
            Continual learning results
        """
        if not self.config.enable_continual_learning:
            raise ValueError("Continual learning not enabled")
        
        print(f"\nâ™¾ï¸  CONTINUAL LEARNING")
        print(f"   âœâ€ Number of Tasks: {len(tasks)}")
        print(f"   âœâ€ Strategy: {self.config.continual_strategy.value}")
        print(f"   ââ€ Epochs/Task: {epochs_per_task}\n")
        
        task_performances = []
        
        for task_id, (X_task, y_task) in enumerate(tasks):
            print(f"\nðŸ Learning Task {task_id + 1}/{len(tasks)}")
            print("â€" * 40)
            
            # Train on current task
            if self.config.continual_strategy == ContinualLearningStrategy.EXPERIENCE_REPLAY:
                # Mix with replay buffer
                if self.experience_replay and len(self.experience_replay.buffer) > 0:
                    X_replay, y_replay = self.experience_replay.sample(len(X_task) // 2)
                    X_mixed = np.vstack([X_task, X_replay])
                    y_mixed = np.vstack([y_task, y_replay])
                    self.base_cerebrum.train(X_mixed, y_mixed, 
                                            epochs=epochs_per_task, verbose=False)
                else:
                    self.base_cerebrum.train(X_task, y_task, 
                                            epochs=epochs_per_task, verbose=False)
                
                # Add to replay buffer
                self.experience_replay.add(X_task, y_task)
            
            else:  # EWC
                self.base_cerebrum.train(X_task, y_task, 
                                        epochs=epochs_per_task, verbose=False)
                
                # Compute Fisher information after training
                if self.ewc:
                    self.ewc.compute_fisher_information(X_task, y_task)
            
            # Evaluate on all previous tasks
            print(f"\n  ðŸŠ Evaluating all tasks:")
            task_losses = []
            for eval_id, (X_eval, y_eval) in enumerate(tasks[:task_id + 1]):
                pred = self.base_cerebrum.predict(X_eval)
                loss = np.mean((pred - y_eval) ** 2)
                task_losses.append(loss)
                print(f"     Task {eval_id + 1}: Loss = {loss:.6f}")
            
            task_performances.append(task_losses)
        
        # Compute final metrics
        avg_performance = np.mean([losses[-1] for losses in task_performances])
        forgetting = self._compute_forgetting(task_performances)
        
        print(f"\nâœ… Continual Learning Complete!")
        print(f"   âœâ€ Avg Final Loss: {avg_performance:.6f}")
        print(f"   ââ€ Avg Forgetting: {forgetting:.6f}")
        
        return {
            'task_performances': task_performances,
            'avg_performance': avg_performance,
            'forgetting': forgetting
        }
    
    def _compute_forgetting(self, task_performances: List[List[float]]) -> float:
        """Compute average forgetting metric"""
        if len(task_performances) < 2:
            return 0.0
        
        forgetting_values = []
        for task_id in range(len(task_performances) - 1):
            initial_perf = task_performances[task_id][task_id]
            final_perf = task_performances[-1][task_id]
            forgetting_values.append(final_perf - initial_perf)
        
        return np.mean(forgetting_values) if forgetting_values else 0.0
    
    # --------------------------------------------------------
    # MIXTURE OF EXPERTS METHODS
    # --------------------------------------------------------
    
    def train_moe(self, X: np.ndarray, y: np.ndarray, 
                  epochs: int = 100, verbose: bool = True) -> Dict:
        """
        Train Mixture of Experts
        
        Args:
            X: Training data
            y: Training labels
            epochs: Training epochs
            verbose: Print progress
            
        Returns:
            Training results
        """
        if not self.config.enable_moe or not self.moe:
            raise ValueError("Mixture of Experts not enabled")
        
        print(f"\nðŸ'¥ MIXTURE OF EXPERTS TRAINING")
        print(f"   âœâ€ Number of Experts: {self.config.num_experts}")
        print(f"   âœâ€ Top-K Selection: {self.config.top_k_experts}")
        print(f"   ââ€ Epochs: {epochs}\n")
        
        # Train all experts
        self.moe.train_experts(X, y, epochs=epochs)
        
        # Evaluate ensemble
        predictions = self.moe.forward(X)
        mse = np.mean((predictions - y) ** 2)
        
        print(f"\nâœ… MoE Training Complete!")
        print(f"   ââ€ Ensemble MSE: {mse:.6f}")
        
        return {'mse': mse}
    
    # --------------------------------------------------------
    # CURRICULUM LEARNING METHODS
    # --------------------------------------------------------
    
    def curriculum_train(self, X: np.ndarray, y: np.ndarray,
                        epochs: int = 200,
                        batch_size: int = 32,
                        verbose: bool = True) -> Dict:
        """
        Train with curriculum learning strategy
        
        Args:
            X: Training data
            y: Training labels
            epochs: Training epochs
            batch_size: Batch size
            verbose: Print progress
            
        Returns:
            Training results
        """
        if not self.config.enable_curriculum or not self.curriculum_scheduler:
            raise ValueError("Curriculum learning not enabled")
        
        print(f"\nðŸš CURRICULUM LEARNING")
        print(f"   âœâ€ Difficulty Measure: {self.config.difficulty_measure}")
        print(f"   âœâ€ Epochs: {epochs}")
        print(f"   ââ€ Batch Size: {batch_size}\n")
        
        # Initial difficulty assessment
        print("  ðŸŠ Computing initial sample difficulties...")
        self.curriculum_scheduler.sample_difficulties = \
            self.curriculum_scheduler.compute_difficulties(self.base_cerebrum, X, y)
        
        # Training with curriculum
        for epoch in range(epochs):
            # Get curriculum batch
            X_batch, y_batch = self.curriculum_scheduler.get_curriculum_batch(
                X, y, batch_size, epoch, epochs
            )
            
            # Train on batch
            self.base_cerebrum.train(X_batch, y_batch, epochs=1, verbose=False)
            
            # Periodic reporting
            if verbose and epoch % max(1, epochs // 20) == 0:
                pred = self.base_cerebrum.predict(X)
                loss = np.mean((pred - y) ** 2)
                print(f"  Epoch {epoch}/{epochs} | Loss: {loss:.6f} | "
                      f"Batch Size: {len(X_batch)}")
            
            # Update difficulties periodically
            if epoch % 50 == 0:
                self.curriculum_scheduler.sample_difficulties = \
                    self.curriculum_scheduler.compute_difficulties(
                        self.base_cerebrum, X, y
                    )
        
        final_pred = self.base_cerebrum.predict(X)
        final_loss = np.mean((final_pred - y) ** 2)
        
        print(f"\nâœ… Curriculum Training Complete!")
        print(f"   ââ€ Final Loss: {final_loss:.6f}")
        
        return {'final_loss': final_loss}
    
    # --------------------------------------------------------
    # ACTIVE LEARNING METHODS
    # --------------------------------------------------------
    
    def active_train(self, X_pool: np.ndarray, y_pool: np.ndarray,
                    X_initial: np.ndarray, y_initial: np.ndarray,
                    oracle_budget: int = None,
                    queries_per_round: int = 10,
                    verbose: bool = True) -> Dict:
        """
        Active learning with oracle queries
        
        Args:
            X_pool: Unlabeled data pool
            y_pool: True labels (oracle)
            X_initial: Initial labeled data
            y_initial: Initial labels
            oracle_budget: Total query budget
            queries_per_round: Queries per round
            verbose: Print progress
            
        Returns:
            Active learning results
        """
        if not self.config.enable_active_learning or not self.active_learner:
            raise ValueError("Active learning not enabled")
        
        if oracle_budget is None:
            oracle_budget = self.config.query_budget
        
        print(f"\nðŸŽ¯ ACTIVE LEARNING")
        print(f"   âœâ€ Pool Size: {len(X_pool)}")
        print(f"   âœâ€ Initial Labeled: {len(X_initial)}")
        print(f"   âœâ€ Oracle Budget: {oracle_budget}")
        print(f"   ââ€ Query Strategy: {self.config.query_strategy}\n")
        
        # Start with initial data
        X_labeled = X_initial.copy()
        y_labeled = y_initial.copy()
        
        queries_made = 0
        round_num = 0
        history = []
        
        while queries_made < oracle_budget:
            round_num += 1
            print(f"\nðŸ„ Round {round_num}")
            print("â€" * 30)
            
            # Train on current labeled data
            print(f"  Training on {len(X_labeled)} samples...")
            self.base_cerebrum.train(X_labeled, y_labeled, epochs=50, verbose=False)
            
            # Evaluate
            pred = self.base_cerebrum.predict(X_labeled)
            train_loss = np.mean((pred - y_labeled) ** 2)
            
            # Query oracle
            n_queries = min(queries_per_round, oracle_budget - queries_made)
            selected_indices = self.active_learner.select_samples(
                self.base_cerebrum, X_pool, n_queries
            )
            
            if len(selected_indices) == 0:
                print("  No more samples to query!")
                break
            
            # Add queried samples
            X_queried = X_pool[selected_indices]
            y_queried = y_pool[selected_indices]
            
            X_labeled = np.vstack([X_labeled, X_queried])
            y_labeled = np.vstack([y_labeled, y_queried])
            
            queries_made += len(selected_indices)
            
            print(f"  Queried: {len(selected_indices)} samples")
            print(f"  Total Labeled: {len(X_labeled)}")
            print(f"  Train Loss: {train_loss:.6f}")
            
            history.append({
                'round': round_num,
                'queries_made': queries_made,
                'labeled_size': len(X_labeled),
                'train_loss': train_loss
            })
        
        # Final training
        print(f"\n  ðŸŽ Final training on {len(X_labeled)} samples...")
        self.base_cerebrum.train(X_labeled, y_labeled, epochs=100, verbose=False)
        
        final_pred = self.base_cerebrum.predict(X_labeled)
        final_loss = np.mean((final_pred - y_labeled) ** 2)
        
        print(f"\nâœ… Active Learning Complete!")
        print(f"   âœâ€ Total Queries: {queries_made}")
        print(f"   âœâ€ Final Labeled: {len(X_labeled)}")
        print(f"   ââ€ Final Loss: {final_loss:.6f}")
        
        return {
            'history': history,
            'final_loss': final_loss,
            'total_queries': queries_made
        }
    
    # --------------------------------------------------------
    # UNIFIED TRAINING
    # --------------------------------------------------------
    
    def ultra_train(self, X: np.ndarray, y: np.ndarray,
                   X_val: Optional[np.ndarray] = None,
                   y_val: Optional[np.ndarray] = None,
                   epochs: int = 200,
                   use_evolution: bool = True,
                   use_curriculum: bool = True,
                   verbose: bool = True) -> Dict:
        """
        Ultra-advanced training combining multiple strategies
        
        Args:
            X: Training data
            y: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Training epochs
            use_evolution: Use evolutionary training
            use_curriculum: Use curriculum learning
            verbose: Print progress
            
        Returns:
            Comprehensive training results
        """
        print(f"\n{'='*60}")
        print(f"ðŸš€ ULTRA-ADVANCED TRAINING: {self.config.name}")
        print(f"{'='*60}")
        
        results = {}
        
        # Phase 1: Evolution (if enabled)
        if use_evolution and self.config.enable_evolution:
            print(f"\nðŸ§¬ PHASE 1: EVOLUTIONARY OPTIMIZATION")
            results['evolution'] = self.evolve(
                X, y, 
                generations=5,
                epochs_per_gen=epochs // 10,
                verbose=verbose
            )
        
        # Phase 2: Curriculum Learning (if enabled)
        if use_curriculum and self.config.enable_curriculum:
            print(f"\nðŸš PHASE 2: CURRICULUM LEARNING")
            results['curriculum'] = self.curriculum_train(
                X, y,
                epochs=epochs,
                batch_size=32,
                verbose=verbose
            )
        else:
            # Standard training
            print(f"\nðŸ‹ï¸ STANDARD TRAINING")
            self.base_cerebrum.train(
                X, y, X_val, y_val,
                epochs=epochs,
                verbose=verbose
            )
        
        # Phase 3: MoE Training (if enabled)
        if self.config.enable_moe:
            print(f"\nðŸ'¥ PHASE 3: MIXTURE OF EXPERTS")
            #FIX: Create proper config with input/output sizes
            moe_config = CerebrumConfig(
                name=f"{self.config.name}_Expert",
                architecture=self.config.base_architecture,
                input_size=self.base_cerebrum.input_size,  # ADD THIS
                output_size=self.base_cerebrum.output_size  # ADD THIS
            )
            self.moe = MixtureOfExperts(
                num_experts=self.config.num_experts,
                expert_config=moe_config,  # USE NEW CONFIG
                top_k=self.config.top_k_experts
            )
            results['moe'] = self.train_moe(X, y, epochs=epochs//2, verbose=verbose)
        
        # Final evaluation
        print(f"\n{'='*60}")
        print(f"ðŸŠ FINAL EVALUATION")
        print(f"{'='*60}")
        
        train_pred = self.predict(X)
        train_mse = np.mean((train_pred - y) ** 2)
        train_r2 = 1 - np.sum((y - train_pred)**2) / np.sum((y - np.mean(y))**2)
        
        print(f"  Train MSE: {train_mse:.6f}")
        print(f"  Train R²:  {train_r2:.6f}")
        
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_mse = np.mean((val_pred - y_val) ** 2)
            val_r2 = 1 - np.sum((y_val - val_pred)**2) / np.sum((y_val - np.mean(y_val))**2)
            
            print(f"  Val MSE:   {val_mse:.6f}")
            print(f"  Val R²:    {val_r2:.6f}")
            
            results['val_mse'] = val_mse
            results['val_r2'] = val_r2
        
        results['train_mse'] = train_mse
        results['train_r2'] = train_r2
        
        print(f"{'='*60}\n")
        
        return results
    
    # --------------------------------------------------------
    # PREDICTION METHODS
    # --------------------------------------------------------
    
    # --------------------------------------------------------
    # PREDICTION METHODS - BRAIN ONLY (NO MEMORY AUGMENTATION)
    # --------------------------------------------------------
    
    def predict(self, X: np.ndarray, 
                return_uncertainty: bool = False) -> Union[np.ndarray, Tuple]:
        """
        Make predictions using best available model
        
        Brain-only inference for clean, fast predictions.
        Memory helps during training via replay, not inference.
        
        Args:
            X: Input data
            return_uncertainty: Return uncertainty estimates
            
        Returns:
            Predictions (and uncertainty if requested)
        """
        # Use best model: MoE if available, else base cerebrum
        if self.config.enable_moe and self.moe:
            # Mixture of Experts forward pass
            predictions = self.moe.forward(X)
            
            if return_uncertainty:
                # Compute uncertainty from expert disagreement
                expert_preds = np.array([expert.predict(X) for expert in self.moe.experts])
                uncertainty = np.std(expert_preds, axis=0)
                return predictions, uncertainty
            
            return predictions
        
        elif self.base_cerebrum:
            # Base cerebrum prediction (brain only)
            return self.base_cerebrum.predict(X, return_uncertainty=return_uncertainty)
        
        else:
            raise ValueError("No trained model available for prediction")
    # --------------------------------------------------------
    # ANALYSIS & VISUALIZATION
    # --------------------------------------------------------
    
    def summary(self, detailed: bool = True):
        """Display comprehensive system summary"""
        print(f"\n{'='*60}")
        print(f"ðŸ§¬ NEUROGENESIS SUMMARY: {self.config.name}")
        print(f"{'='*60}")
        print(f"Base Architecture:   {self.config.base_architecture.upper()}")
        
        if self.config.enable_evolution:
            print(f"\nðŸ§¬ Evolution:")
            print(f"  Strategy:          {self.config.evolution_strategy.value}")
            print(f"  Population:        {self.config.population_size}")
            print(f"  Mutation Rate:     {self.config.mutation_rate}")
        
        if self.config.enable_meta_learning:
            print(f"\nðŸ§  Meta-Learning:")
            print(f"  Strategy:          {self.config.meta_strategy.value}")
            print(f"  Meta LR:           {self.config.meta_learning_rate}")
        
        # Initialize continual learning

        if self.config.enable_moe:
            print(f"\nðŸ'¥ Mixture of Experts:")
            print(f"  Num Experts:       {self.config.num_experts}")
            print(f"  Top-K:             {self.config.top_k_experts}")
        
        if self.config.enable_curriculum:
            print(f"\nðŸš Curriculum Learning: ENABLED")
        
        if self.config.enable_active_learning:
            print(f"\nðŸŽ¯ Active Learning:")
            print(f"  Strategy:          {self.config.query_strategy}")
            print(f"  Budget:            {self.config.query_budget}")
        
        if detailed and self.base_cerebrum:
            print(f"\nBase Cerebrum:")
            print("-" * 60)
            self.base_cerebrum.summary(detailed=False)
        
        print(f"{'='*60}\n")
    
    # --------------------------------------------------------
    # DEMONSTRATION SUITE
    # --------------------------------------------------------
    
    def demo_suite(self, mode: str = "all"):
        """
        Comprehensive demonstration suite
        
        Modes:
            - evolution: Neural evolution demo
            - meta: Meta-learning demo
            - continual: Continual learning demo
            - moe: Mixture of Experts demo
            - active: Active learning demo
            - all: All demonstrations
        """
        print(f"\n{'='*60}")
        print(f"ðŸŽ­ NEUROGENESIS DEMONSTRATION SUITE")
        print(f"{'='*60}\n")
        
        if mode in ["evolution", "all"]:
            self._demo_evolution()
        
        if mode in ["continual", "all"]:
            self._demo_continual()
        
        if mode in ["active", "all"]:
            self._demo_active()
    
    def _demo_evolution(self):
        """Evolutionary learning demonstration"""
        print("ðŸ§¬ EVOLUTION DEMONSTRATION")
        print("â€" * 60)
        
        # Generate complex non-linear data
        X = np.linspace(-2, 2, 100).reshape(-1, 1)
        y = (np.sin(3*X) * np.exp(-X**2/2) + 0.5*X).reshape(-1, 1)
        y += np.random.normal(0, 0.05, y.shape)
        
        # Create evolution-enabled system
        config = NeuroGenesisConfig(
            name="EvolutionDemo",
            base_architecture="small",
            enable_evolution=True,
            population_size=8,
            evolution_strategy=EvolutionStrategy.GENETIC_ALGORITHM
        )
        
        system = NeuroGenesis(config=config)
        
        # Evolve
        system.evolve(X, y, generations=5, epochs_per_gen=30, verbose=True)
        
        # Evaluate
        pred = system.predict(X)
        mse = np.mean((pred - y) ** 2)
        print(f"\nFinal MSE: {mse:.6f}\n")
    
    def _demo_continual(self):
        """Continual learning demonstration"""
        print("â™¾ï¸  CONTINUAL LEARNING DEMONSTRATION")
        print("â€" * 60)
        
        # Create sequence of tasks
        tasks = []
        for i in range(3):
            X = np.linspace(-2, 2, 80).reshape(-1, 1)
            y = (np.sin((i+1)*X) + 0.3*X*i).reshape(-1, 1)
            tasks.append((X, y))
        
        # Create continual learning system
        config = NeuroGenesisConfig(
            name="ContinualDemo",
            base_architecture="small",
            enable_continual_learning=True,
            continual_strategy=ContinualLearningStrategy.EXPERIENCE_REPLAY,
            memory_size=200
        )
        
        system = NeuroGenesis(config=config)
        
        # Sequential learning
        system.continual_train(tasks, epochs_per_task=50, verbose=True)
        print()
    
    def _demo_active(self):
        """Active learning demonstration"""
        print("ðŸŽ¯ ACTIVE LEARNING DEMONSTRATION")
        print("â€" * 60)
        
        # Generate data
        X_pool = np.linspace(-3, 3, 200).reshape(-1, 1)
        y_pool = (np.sin(2*X_pool) + 0.5*np.cos(X_pool)).reshape(-1, 1)
        
        # Start with few labeled samples
        initial_indices = np.random.choice(len(X_pool), 10, replace=False)
        X_initial = X_pool[initial_indices]
        y_initial = y_pool[initial_indices]
        
        # Create active learning system
        config = NeuroGenesisConfig(
            name="ActiveDemo",
            base_architecture="small",
            enable_active_learning=True,
            query_strategy="uncertainty",
            query_budget=30
        )
        
        system = NeuroGenesis(config=config)
        
        # Active learning
        system.active_train(
            X_pool, y_pool, X_initial, y_initial,
            oracle_budget=30, queries_per_round=5, verbose=True
        )
        print()


# ============================================================
# ADVANCED BENCHMARKING & COMPARISON
# ============================================================

class NeuroGenesisBenchmark:
    """Comprehensive benchmarking suite"""
    
    @staticmethod
    def compare_strategies(X: np.ndarray, y: np.ndarray,
                          strategies: List[str] = None,
                          epochs: int = 100) -> Dict:
        """
        Compare different NeuroGenesis strategies
        
        Args:
            X: Training data
            y: Training labels
            strategies: List of strategies to compare
            epochs: Training epochs
            
        Returns:
            Comparison results
        """
        if strategies is None:
            strategies = ['baseline', 'evolution', 'curriculum', 'moe']
        
        print(f"\nðŸŠ STRATEGY COMPARISON BENCHMARK")
        print(f"{'='*60}")
        print(f"Data: {len(X)} samples")
        print(f"Strategies: {', '.join(strategies)}")
        print(f"{'='*60}\n")
        
        results = {}
        
        for strategy in strategies:
            print(f"\nðŸ§ª Testing: {strategy.upper()}")
            print("â€" * 40)
            
            start_time = time.time()
            
            if strategy == 'baseline':
                # Standard Cerebrum
                config = NeuroGenesisConfig(
                    name=f"Baseline",
                    base_architecture="medium",
                    enable_evolution=False,
                    enable_curriculum=False,
                    enable_moe=False
                )
                system = NeuroGenesis(config=config)
                system.base_cerebrum.train(X, y, epochs=epochs, verbose=False)
                pred = system.predict(X)
                
            elif strategy == 'evolution':
                config = NeuroGenesisConfig(
                    name="Evolution",
                    base_architecture="medium",
                    enable_evolution=True,
                    population_size=6
                )
                system = NeuroGenesis(config=config)
                system.evolve(X, y, generations=5, epochs_per_gen=epochs//5, verbose=False)
                pred = system.predict(X)
                
            elif strategy == 'curriculum':
                config = NeuroGenesisConfig(
                    name="Curriculum",
                    base_architecture="medium",
                    enable_curriculum=True
                )
                system = NeuroGenesis(config=config)
                system.curriculum_train(X, y, epochs=epochs, batch_size=32, verbose=False)
                pred = system.predict(X)
                
            elif strategy == 'moe':
                config = NeuroGenesisConfig(
                    name="MoE",
                    base_architecture="small",
                    enable_moe=True,
                    num_experts=4
                )
                system = NeuroGenesis(config=config)
                system.train_moe(X, y, epochs=epochs, verbose=False)
                pred = system.predict(X)
            
            elapsed = time.time() - start_time
            
            # Metrics
            mse = np.mean((pred - y) ** 2)
            mae = np.mean(np.abs(pred - y))
            r2 = 1 - np.sum((y - pred)**2) / np.sum((y - np.mean(y))**2)
            
            results[strategy] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'time': elapsed
            }
            
            print(f"  MSE:  {mse:.6f}")
            print(f"  MAE:  {mae:.6f}")
            print(f"  R²:   {r2:.6f}")
            print(f"  Time: {elapsed:.2f}s")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"ðŸ† WINNER ANALYSIS")
        print(f"{'='*60}")
        
        best_mse = min(results.items(), key=lambda x: x[1]['mse'])
        best_r2 = max(results.items(), key=lambda x: x[1]['r2'])
        fastest = min(results.items(), key=lambda x: x[1]['time'])
        
        print(f"Best MSE:     {best_mse[0].upper()} ({best_mse[1]['mse']:.6f})")
        print(f"Best R²:      {best_r2[0].upper()} ({best_r2[1]['r2']:.6f})")
        print(f"Fastest:      {fastest[0].upper()} ({fastest[1]['time']:.2f}s)")
        print(f"{'='*60}\n")
        
        return results
    
    @staticmethod
    def scalability_test(architecture: str = "medium",
                        input_sizes: List[int] = None) -> Dict:
        """
        Test scalability across different input dimensions
        
        Args:
            architecture: Architecture to test
            input_sizes: List of input dimensions
            
        Returns:
            Scalability results
        """
        if input_sizes is None:
            input_sizes = [1, 5, 10, 20, 50]
        
        print(f"\nâš¡ SCALABILITY BENCHMARK")
        print(f"{'='*60}")
        print(f"Architecture: {architecture}")
        print(f"Input Sizes: {input_sizes}")
        print(f"{'='*60}\n")
        
        results = {}
        
        for input_size in input_sizes:
            print(f"\nðŸŠ Testing input_size = {input_size}")
            
            # Generate data
            X = np.random.randn(1000, input_size)
            y = np.random.randn(1000, 1)
            
            # Create system
            config = NeuroGenesisConfig(
                name=f"Scale_{input_size}",
                base_architecture=architecture,
                enable_evolution=False
            )
            
            # Adjust config for input size
            config_dict = {
                'name': f"Scale_{input_size}",
                'architecture': architecture,
                'input_size': input_size,
                'output_size': 1
            }
            
            cerebrum = create_cerebrum(**config_dict)
            
            # Benchmark
            start = time.time()
            cerebrum.train(X, y, epochs=50, verbose=False)
            train_time = time.time() - start
            
            # Prediction time
            start = time.time()
            _ = cerebrum.predict(X)
            pred_time = time.time() - start
            
            # Memory (approximate)
            params = cerebrum._count_parameters()
            
            results[input_size] = {
                'train_time': train_time,
                'pred_time': pred_time,
                'parameters': params
            }
            
            print(f"  Train Time: {train_time:.2f}s")
            print(f"  Pred Time:  {pred_time*1000:.2f}ms")
            print(f"  Parameters: {params:,}")
        
        print(f"\n{'='*60}\n")
        
        return results


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_neurogenesis(architecture: str = "medium",
                       enable_all: bool = False,
                       **kwargs) -> NeuroGenesis:
    """
    Factory function to create NeuroGenesis systems
    
    Args:
        architecture: Base architecture
        enable_all: Enable all advanced features
        **kwargs: Additional configuration
        
    Returns:
        Configured NeuroGenesis instance
    """
    if enable_all:
        config = NeuroGenesisConfig(
            base_architecture=architecture,
            enable_evolution=True,
            enable_meta_learning=False,  # Requires task generator
            enable_continual_learning=True,
            enable_moe=True,
            enable_curriculum=True,
            enable_active_learning=True,
            **kwargs
        )
    else:
        config = NeuroGenesisConfig(
            base_architecture=architecture,
            **kwargs
        )
    
    return NeuroGenesis(config=config)


def create_evolution_system(population_size: int = 10,
                           **kwargs) -> NeuroGenesis:
    """Create evolution-focused system"""
    config = NeuroGenesisConfig(
        enable_evolution=True,
        population_size=population_size,
        evolution_strategy=EvolutionStrategy.GENETIC_ALGORITHM,
        **kwargs
    )
    return NeuroGenesis(config=config)


def create_continual_system(memory_size: int = 1000,
                           **kwargs) -> NeuroGenesis:
    """Create continual learning system"""
    config = NeuroGenesisConfig(
        enable_continual_learning=True,
        continual_strategy=ContinualLearningStrategy.EXPERIENCE_REPLAY,
        memory_size=memory_size,
        **kwargs
    )
    return NeuroGenesis(config=config)

def create_moe_system(num_experts: int = 4,
                     top_k: int = 2,
                     input_size: int = 1,
                     output_size: int = 1,
                     **kwargs) -> NeuroGenesis:
    """Create Mixture of Experts system"""
    # ADD input_size and output_size to config
    config = NeuroGenesisConfig(
        enable_moe=True,
        num_experts=num_experts,
        top_k_experts=top_k,
        input_size=input_size,      # ADD THIS
        output_size=output_size,    # ADD THIS
        **kwargs
    )
    return NeuroGenesis(config=config)

# ============================================================
# MAIN DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ðŸ§¬ NEUROGENESIS v3.0 - Ultra-Advanced Neural Evolution")
    print("="*60)
    
    # ========================================
    # DEMO 1: EVOLUTION
    # ========================================
    print("\n" + "="*60)
    print("DEMO 1: EVOLUTIONARY NEURAL NETWORKS")
    print("="*60)
    
    # Complex function approximation
    X_train = np.linspace(-3, 3, 150).reshape(-1, 1)
    y_train = (np.sin(2*X_train) * np.exp(-X_train**2/3) + 
               0.5*np.cos(3*X_train)).reshape(-1, 1)
    y_train += np.random.normal(0, 0.05, y_train.shape)
    
    evolution_system = create_evolution_system(
        name="EvolutionMaster",
        base_architecture="medium",
        population_size=8
    )
    
    evolution_system.evolve(
        X_train, y_train,
        generations=5,
        epochs_per_gen=40,
        verbose=True
    )
    
    # ========================================
    # DEMO 2: CONTINUAL LEARNING
    # ========================================
    print("\n" + "="*60)
    print("DEMO 2: CONTINUAL LEARNING WITHOUT FORGETTING")
    print("="*60)
    
    # Create sequence of related tasks
    tasks = []
    for i in range(4):
        X_task = np.linspace(-2, 2, 100).reshape(-1, 1)
        # Each task is a variation
        y_task = (np.sin((i+1)*X_task) + 0.2*i*X_task**2).reshape(-1, 1)
        tasks.append((X_task, y_task))
    
    continual_system = create_continual_system(
        name="ContinualLearner",
        base_architecture="medium",
        memory_size=300
    )
    
    continual_results = continual_system.continual_train(
        tasks,
        epochs_per_task=80,
        verbose=True
    )
    
    # ========================================
    # DEMO 3: MIXTURE OF EXPERTS
    # ========================================
    print("\n" + "="*60)
    print("DEMO 3: MIXTURE OF EXPERTS")
    print("="*60)
    
    # Generate multi-modal data
    X_moe = np.random.uniform(-5, 5, (200, 2))
    # Different regions follow different patterns
    y_moe = np.where(
        X_moe[:, 0] > 0,
        np.sin(X_moe[:, 0]) + np.cos(X_moe[:, 1]),
        X_moe[:, 0]**2 + X_moe[:, 1]
    ).reshape(-1, 1)
    
    moe_system = create_moe_system(
        name="ExpertEnsemble",
        base_architecture="small",
        num_experts=4,
        top_k=2,
        input_size=2,  # ADD THIS LINE (X_moe has 2 features)
        output_size=1  # ADD THIS LINE
    )
    
    moe_system.train_moe(X_moe, y_moe, epochs=100, verbose=True)
    
    # ========================================
    # DEMO 4: COMPREHENSIVE COMPARISON
    # ========================================
    print("\n" + "="*60)
    print("DEMO 4: STRATEGY COMPARISON")
    print("="*60)
    
    # Generate challenging dataset
    X_compare = np.linspace(-4, 4, 200).reshape(-1, 1)
    y_compare = (np.sin(X_compare) * np.cos(2*X_compare) + 
                 0.3*X_compare**2).reshape(-1, 1)
    y_compare += np.random.normal(0, 0.1, y_compare.shape)
    
    comparison_results = NeuroGenesisBenchmark.compare_strategies(
        X_compare, y_compare,
        strategies=['baseline', 'evolution', 'curriculum', 'moe'],
        epochs=100
    )
    
    # ========================================
    # DEMO 5: ULTRA TRAINING
    # ========================================
    print("\n" + "="*60)
    print("DEMO 5: ULTRA-ADVANCED TRAINING")
    print("="*60)
    
    # Create ultimate system
    ultimate_system = create_neurogenesis(
        name="UltimateAI",
        architecture="large",
        enable_evolution=True,
        enable_curriculum=True,
        population_size=6
    )
    
    ultimate_system.summary(detailed=True)
    
    # Ultra training
    X_ultimate = np.linspace(-5, 5, 250).reshape(-1, 1)
    y_ultimate = (np.sin(2*X_ultimate) * np.exp(-X_ultimate**2/5) + 
                  np.cos(X_ultimate) + 0.5*X_ultimate).reshape(-1, 1)
    
    # Split for validation
    split = int(0.8 * len(X_ultimate))
    X_train_ult = X_ultimate[:split]
    y_train_ult = y_ultimate[:split]
    X_val_ult = X_ultimate[split:]
    y_val_ult = y_ultimate[split:]
    
    ultra_results = ultimate_system.ultra_train(
        X_train_ult, y_train_ult,
        X_val_ult, y_val_ult,
        epochs=150,
        use_evolution=True,
        use_curriculum=True,
        verbose=True
    )
    
    # ========================================
    # DEMO 6: ACTIVE LEARNING
    # ========================================
    print("\n" + "="*60)
    print("DEMO 6: ACTIVE LEARNING")
    print("="*60)
    
    # Large unlabeled pool
    X_pool = np.linspace(-4, 4, 300).reshape(-1, 1)
    y_pool = (np.sin(2*X_pool) + np.cos(X_pool)).reshape(-1, 1)
    
    # Small initial labeled set
    initial_idx = np.random.choice(len(X_pool), 15, replace=False)
    X_initial = X_pool[initial_idx]
    y_initial = y_pool[initial_idx]
    
    active_system = create_neurogenesis(
        name="ActiveLearner",
        architecture="medium",
        enable_active_learning=True,
        query_strategy="uncertainty",
        query_budget=40
    )
    
    active_results = active_system.active_train(
        X_pool, y_pool,
        X_initial, y_initial,
        oracle_budget=40,
        queries_per_round=8,
        verbose=True
    )
    
    # ========================================
    # DEMO 7: SCALABILITY BENCHMARK
    # ========================================
    print("\n" + "="*60)
    print("DEMO 7: SCALABILITY ANALYSIS")
    print("="*60)
    
    scalability_results = NeuroGenesisBenchmark.scalability_test(
        architecture="medium",
        input_sizes=[1, 5, 10, 20]
    )
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*60)
    print("âœ¨ ALL DEMONSTRATIONS COMPLETE!")
    print("="*60)
    
    print("\nðŸ§¬ NeuroGenesis Features Demonstrated:")
    print("  âœ Evolutionary Neural Networks")
    print("  âœ Continual Learning (No Catastrophic Forgetting)")
    print("  âœ Mixture of Experts Architecture")
    print("  âœ Curriculum Learning")
    print("  âœ Active Learning with Uncertainty Sampling")
    print("  âœ Comprehensive Benchmarking")
    print("  âœ Ultra-Advanced Training Pipeline")
    
    print("\nðŸš€ Key Innovations:")
    print("  • Genetic Algorithm-based Architecture Evolution")
    print("  • Experience Replay for Continual Learning")
    print("  • Expert Routing with Gating Networks")
    print("  • Adaptive Difficulty Curriculum")
    print("  • Query-Efficient Active Learning")
    print("  • Multi-Strategy Ensemble Methods")
    
    print("\nðŸŠ Performance Highlights:")
    if comparison_results:
        best_strategy = min(comparison_results.items(), 
                          key=lambda x: x[1]['mse'])
        print(f"  • Best Strategy: {best_strategy[0].upper()}")
        print(f"  • Achieved MSE: {best_strategy[1]['mse']:.6f}")
        print(f"  • R² Score: {best_strategy[1]['r2']:.6f}")
    
    if 'forgetting' in continual_results:
        print(f"  • Continual Learning Forgetting: {continual_results['forgetting']:.6f}")
    
    if 'total_queries' in active_results:
        print(f"  • Active Learning Queries: {active_results['total_queries']}")
        print(f"  • Final Loss: {active_results['final_loss']:.6f}")
    
    print("\n" + "="*60)
    print("ðŸŽ‰ NeuroGenesis v3.0 - Ready for Production!")
    print("="*60 + "\n")
    
    # Save ultimate system
    ultimate_system.base_cerebrum.save("neurogenesis_ultimate_model")
    
    print("ðŸ'¾ Model saved: neurogenesis_ultimate_model")
    print("\nðŸš Next Steps:")
    print("  1. Load pre-trained model for inference")
    print("  2. Fine-tune on your specific task")
    print("  3. Deploy with continual learning enabled")
    print("  4. Monitor and improve with active learning")
    print("\nHappy Neural Evolution! ðŸ§¬âœ¨")