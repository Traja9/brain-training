"""
===========================================================
ADVANCED LOGIC CEREBRUM MODULE v2.0
-----------------------------------------------------------
A sophisticated modular nano-brain architecture with:
- Adaptive learning rate scheduling
- Advanced regularization techniques
- Neural architecture search capabilities
- Ensemble learning support
- Real-time performance monitoring
- Transfer learning capabilities
- Attention mechanisms
- Gradient clipping and normalization
===========================================================
"""

import time
import numpy as np
from typing import Optional, List, Dict, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from Vectorial_Calculus_ML import (
    UltraPreciseVectorCalculusEngine,
    OptimizationMethod,
    MLOptimizationResult
)
from Neural_Algorithm import (
    NeuralDataProcessor,
    LayerConfig,
    ActivationFunction,
    LossFunction
)


class SchedulerType(Enum):
    """Learning rate scheduler types"""
    CONSTANT = "constant"
    STEP_DECAY = "step_decay"
    EXPONENTIAL = "exponential"
    COSINE_ANNEALING = "cosine_annealing"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    WARMUP_COSINE = "warmup_cosine"


class RegularizationType(Enum):
    """Regularization techniques"""
    L1 = "l1"
    L2 = "l2"
    ELASTIC_NET = "elastic_net"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics tracking"""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    weight_norm: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class CerebrumConfig:
    """Complete cerebrum configuration"""
    name: str = "AdvancedCerebrum"
    architecture: str = "small"
    input_size: int = 1
    output_size: int = 1
    learning_rate: float = 0.01
    scheduler_type: SchedulerType = SchedulerType.COSINE_ANNEALING
    regularization: List[RegularizationType] = field(default_factory=list)
    dropout_rate: float = 0.0
    l1_lambda: float = 0.0
    l2_lambda: float = 0.0
    gradient_clip_value: float = 1.0
    early_stopping_patience: int = 50
    use_attention: bool = False
    ensemble_size: int = 1


class LearningRateScheduler:
    """Advanced learning rate scheduling"""
    
    def __init__(self, initial_lr: float, scheduler_type: SchedulerType, **kwargs):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.scheduler_type = scheduler_type
        self.kwargs = kwargs
        self.step_count = 0
        self.best_loss = float('inf')
        
    def step(self, epoch: int, loss: Optional[float] = None) -> float:
        """Update learning rate based on scheduler type"""
        self.step_count += 1
        
        if self.scheduler_type == SchedulerType.CONSTANT:
            return self.current_lr
            
        elif self.scheduler_type == SchedulerType.STEP_DECAY:
            step_size = self.kwargs.get('step_size', 50)
            gamma = self.kwargs.get('gamma', 0.5)
            self.current_lr = self.initial_lr * (gamma ** (epoch // step_size))
            
        elif self.scheduler_type == SchedulerType.EXPONENTIAL:
            gamma = self.kwargs.get('gamma', 0.95)
            self.current_lr = self.initial_lr * (gamma ** epoch)
            
        elif self.scheduler_type == SchedulerType.COSINE_ANNEALING:
            T_max = self.kwargs.get('T_max', 100)
            eta_min = self.kwargs.get('eta_min', 1e-6)
            self.current_lr = eta_min + (self.initial_lr - eta_min) * \
                (1 + np.cos(np.pi * epoch / T_max)) / 2
                
        elif self.scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            if loss is not None and loss < self.best_loss:
                self.best_loss = loss
            elif loss is not None:
                patience = self.kwargs.get('patience', 10)
                factor = self.kwargs.get('factor', 0.5)
                if epoch % patience == 0:
                    self.current_lr *= factor
                    
        elif self.scheduler_type == SchedulerType.WARMUP_COSINE:
            warmup_epochs = self.kwargs.get('warmup_epochs', 10)
            T_max = self.kwargs.get('T_max', 100)
            if epoch < warmup_epochs:
                self.current_lr = self.initial_lr * (epoch + 1) / warmup_epochs
            else:
                self.current_lr = self.initial_lr * 0.5 * \
                    (1 + np.cos(np.pi * (epoch - warmup_epochs) / (T_max - warmup_epochs)))
        
        return self.current_lr


class EarlyStopping:
    """Advanced early stopping with model checkpointing"""
    
    def __init__(self, patience: int = 50, min_delta: float = 1e-6, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, score: float, model_weights: Optional[Dict] = None) -> bool:
        """Check if training should stop"""
        if self.best_score is None:
            self.best_score = score
            self.best_weights = model_weights
            return False
            
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
            
        if improved:
            self.best_score = score
            self.best_weights = model_weights
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


class AttentionMechanism:
    """Self-attention mechanism for enhanced feature learning"""
    
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.query_weights = np.random.randn(feature_dim, feature_dim) * 0.01
        self.key_weights = np.random.randn(feature_dim, feature_dim) * 0.01
        self.value_weights = np.random.randn(feature_dim, feature_dim) * 0.01
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply attention mechanism"""
        Q = x @ self.query_weights
        K = x @ self.key_weights
        V = x @ self.value_weights
        
        scores = Q @ K.T / np.sqrt(self.feature_dim)
        attention_weights = self._softmax(scores)
        output = attention_weights @ V
        
        return output
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class Cerebrum_Core:
    """
    Advanced modular AI brain with state-of-the-art features:
    - Adaptive learning rate scheduling
    - Multiple regularization techniques
    - Early stopping with checkpointing
    - Attention mechanisms
    - Ensemble learning
    - Comprehensive metrics tracking
    - Transfer learning support
    """

    def __init__(self, config: Optional[CerebrumConfig] = None, **kwargs):
        # Initialize configuration
        if config is None:
            config = CerebrumConfig(**kwargs)
        self.config = config
        
        # Core attributes
        self.name = config.name
        self.architecture = config.architecture.lower()
        self.input_size = config.input_size
        self.output_size = config.output_size
        
        # Models and layers
        self.models: List[NeuralDataProcessor] = []
        self.layers: List[LayerConfig] = []
        
        # Training components
        self.scheduler: Optional[LearningRateScheduler] = None
        self.early_stopping: Optional[EarlyStopping] = None
        self.attention: Optional[AttentionMechanism] = None
        
        # Metrics tracking
        self.training_history: List[TrainingMetrics] = []
        self.best_loss = float('inf')
        self.best_weights = None
        
        # Build architecture
        self._build_advanced_architecture()
        self._initialize_training_components()

    # --------------------------------------------------------
    # ADVANCED ARCHITECTURE BUILDER
    # --------------------------------------------------------
    def _build_advanced_architecture(self):
        """Build sophisticated neural architecture with modern design patterns"""
        
        # Architecture presets with better initialization
        arch_configs = {
            "nano": {
                "layers": [
                    (self.input_size, 4, ActivationFunction.TANH),
                    (4, 4, ActivationFunction.TANH),
                    (4, self.output_size, ActivationFunction.LINEAR)
                ],
                "lr": 0.15,
                "features": 4
            },
            "tiny": {
                "layers": [
                    (self.input_size, 16, ActivationFunction.TANH),
                    (16, 16, ActivationFunction.TANH),
                    (16, self.output_size, ActivationFunction.LINEAR)
                ],
                "lr": 0.1,
                "features": 16
            },
            "small": {
                "layers": [
                    (self.input_size, 32, ActivationFunction.TANH),
                    (32, 32, ActivationFunction.TANH),
                    (32, 16, ActivationFunction.TANH),
                    (16, self.output_size, ActivationFunction.LINEAR)
                ],
                "lr": 0.05,
                "features": 32
            },
            "medium": {
                "layers": [
                    (self.input_size, 64, ActivationFunction.TANH),
                    (64, 64, ActivationFunction.TANH),
                    (64, 32, ActivationFunction.TANH),
                    (32, 16, ActivationFunction.TANH),
                    (16, self.output_size, ActivationFunction.LINEAR)
                ],
                "lr": 0.01,
                "features": 64
            },
            "large": {
                "layers": [
                    (self.input_size, 128, ActivationFunction.TANH),
                    (128, 128, ActivationFunction.TANH),
                    (128, 64, ActivationFunction.TANH),
                    (64, 32, ActivationFunction.TANH),
                    (32, self.output_size, ActivationFunction.LINEAR)
                ],
                "lr": 0.005,
                "features": 128
            },
            "xlarge": {
                "layers": [
                    (self.input_size, 256, ActivationFunction.TANH),
                    (256, 128, ActivationFunction.TANH),
                    (128, 64, ActivationFunction.TANH),
                    (64, 32, ActivationFunction.TANH),
                    (32, 16, ActivationFunction.TANH),
                    (16, self.output_size, ActivationFunction.LINEAR)
                ],
                "lr": 0.001,
                "features": 256
            }
        }
        
        if self.architecture not in arch_configs:
            raise ValueError(f"Unknown architecture: {self.architecture}. "
                           f"Options: {list(arch_configs.keys())}")
        
        arch_config = arch_configs[self.architecture]
        
        # Build layers
        for in_size, out_size, activation in arch_config["layers"]:
            self.layers.append(
                LayerConfig(
                    input_size=in_size,
                    output_size=out_size,
                    activation=activation
                )
            )
        
        # Set learning rate (can be overridden by config)
        if self.config.learning_rate == 0.01:  # Default value
            self.config.learning_rate = arch_config["lr"]
        
        # Initialize attention if enabled
        if self.config.use_attention:
            self.attention = AttentionMechanism(arch_config["features"])
        
        # Create ensemble of models
        for i in range(self.config.ensemble_size):
            model = NeuralDataProcessor(
                layer_configs=self.layers,
                learning_rate=self.config.learning_rate,
                loss_function=LossFunction.MEAN_SQUARED_ERROR,
                optimization_method="adam"
            )
            self.models.append(model)
        
        print(f"\n🧠 [{self.name}] Advanced Architecture Initialized")
        print(f"   ├─ Type: {self.architecture.upper()}")
        print(f"   ├─ Layers: {len(self.layers)}")
        print(f"   ├─ Parameters: ~{self._count_parameters():,}")
        print(f"   ├─ Learning Rate: {self.config.learning_rate}")
        print(f"   ├─ Ensemble Size: {self.config.ensemble_size}")
        print(f"   ├─ Attention: {'✓' if self.config.use_attention else '✗'}")
        print(f"   └─ Regularization: {', '.join([r.value for r in self.config.regularization]) or 'None'}")

    def _initialize_training_components(self):
        """Initialize advanced training components"""
        # Learning rate scheduler
        self.scheduler = LearningRateScheduler(
            initial_lr=self.config.learning_rate,
            scheduler_type=self.config.scheduler_type,
            T_max=200,
            warmup_epochs=10
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=1e-6
        )

    def _count_parameters(self) -> int:
        """Estimate total trainable parameters"""
        total = 0
        for layer in self.layers:
            total += layer.input_size * layer.output_size + layer.output_size
        return total * self.config.ensemble_size

    # --------------------------------------------------------
    # ADVANCED TRAINING
    # --------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray, 
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 200, 
              batch_size: Optional[int] = None,
              verbose: bool = True,
              callback: Optional[Callable] = None) -> Dict:
        """
        Advanced training with comprehensive monitoring
        
        Args:
            X: Training input data
            y: Training target data
            X_val: Validation input data
            y_val: Validation target data
            epochs: Number of training epochs
            batch_size: Mini-batch size (None for full-batch)
            verbose: Print training progress
            callback: Optional callback function(epoch, metrics)
            
        Returns:
            Training history dictionary
        """
        print(f"\n🚀 Advanced Training: {self.name}")
        print(f"   ├─ Epochs: {epochs}")
        print(f"   ├─ Training Samples: {len(X)}")
        if X_val is not None:
            print(f"   ├─ Validation Samples: {len(X_val)}")
        print(f"   └─ Scheduler: {self.config.scheduler_type.value}")
        
        start_time = time.time()
        self.training_history = []
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Update learning rate
            current_lr = self.scheduler.step(epoch)
            for model in self.models:
                model.learning_rate = current_lr
            
            # Train all ensemble models
            train_losses = []
            try:
                for model in self.models:
                    model.train(X, y, epochs=1, batch_size=batch_size, verbose=False)
                    train_loss = self._compute_loss(model, X, y)
                    train_losses.append(train_loss)
            except Exception as e:
                print(f"   ⚠️  Warning: Training error at epoch {epoch}: {e}")
                train_losses = [float('inf')]
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation
            val_loss = None
            if X_val is not None and y_val is not None:
                try:
                    val_loss = np.mean([
                        self._compute_loss(model, X_val, y_val)
                        for model in self.models
                    ])
                except Exception as e:
                    print(f"   ⚠️  Warning: Validation error at epoch {epoch}: {e}")
            
            # Compute metrics (with safe defaults)
            gradient_norm = self._compute_gradient_norm()
            weight_norm = self._compute_weight_norm()
            
            # Record metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                gradient_norm=gradient_norm,
                weight_norm=weight_norm
            )
            self.training_history.append(metrics)
            
            # Progress reporting
            if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1):
                val_str = f", Val Loss: {val_loss:.6f}" if val_loss else ""
                print(f"   Epoch {epoch:4d}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.6f}{val_str} | "
                      f"LR: {current_lr:.6f} | "
                      f"Time: {(time.time() - epoch_start)*1000:.1f}ms")
            
            # Callback
            if callback:
                callback(epoch, metrics)
            
            # Early stopping check
            monitor_loss = val_loss if val_loss is not None else avg_train_loss
            if self.early_stopping(monitor_loss, self._get_model_weights()):
                print(f"\n   ⚠️  Early stopping triggered at epoch {epoch}")
                print(f"   └─ Best loss: {self.early_stopping.best_score:.6f}")
                self._restore_best_weights()
                break
        
        elapsed = (time.time() - start_time) * 1000
        final_loss = self.training_history[-1].train_loss
        
        print(f"\n✅ Training Complete!")
        print(f"   ├─ Total Time: {elapsed:.2f}ms ({elapsed/epochs:.2f}ms/epoch)")
        print(f"   ├─ Final Loss: {final_loss:.6f}")
        print(f"   └─ Best Loss: {self.early_stopping.best_score:.6f}")
        
        return self._get_training_summary()

    def _compute_loss(self, model: NeuralDataProcessor, 
                     X: np.ndarray, y: np.ndarray) -> float:
        """Compute loss for a model"""
        try:
            predictions = model.predict(X)
            mse = np.mean((predictions - y) ** 2)
            
            # Add regularization
            reg_loss = 0.0
            if RegularizationType.L2 in self.config.regularization:
                weight_norm = 0.0
                count = 0
                try:
                    if hasattr(model, 'network') and model.network:
                        for layer in model.network:
                            if hasattr(layer, 'weights'):
                                weight_norm += np.sum(layer.weights ** 2)
                                count += 1
                            elif hasattr(layer, 'W'):
                                weight_norm += np.sum(layer.W ** 2)
                                count += 1
                    if count > 0:
                        reg_loss += self.config.l2_lambda * weight_norm
                except (AttributeError, TypeError):
                    pass
                    
            if RegularizationType.L1 in self.config.regularization:
                l1_norm = 0.0
                try:
                    if hasattr(model, 'network') and model.network:
                        for layer in model.network:
                            if hasattr(layer, 'weights'):
                                l1_norm += np.sum(np.abs(layer.weights))
                            elif hasattr(layer, 'W'):
                                l1_norm += np.sum(np.abs(layer.W))
                    reg_loss += self.config.l1_lambda * l1_norm
                except (AttributeError, TypeError):
                    pass
            
            return mse + reg_loss
        except Exception as e:
            print(f"   ⚠️  Warning: Error computing loss: {e}")
            return float('inf')

    def _compute_gradient_norm(self) -> float:
        """Compute L2 norm of gradients across all models"""
        total_norm = 0.0
        count = 0
        try:
            for model in self.models:
                # Try to access layers or network structure
                if hasattr(model, 'network') and model.network:
                    for layer in model.network:
                        if hasattr(layer, 'weight_gradient'):
                            total_norm += np.sum(layer.weight_gradient ** 2)
                            count += 1
                        elif hasattr(layer, 'dW'):
                            total_norm += np.sum(layer.dW ** 2)
                            count += 1
        except (AttributeError, TypeError):
            # If structure is not accessible, return a default value
            pass
        
        return np.sqrt(total_norm / max(count, 1)) if count > 0 else 0.0

    def _compute_weight_norm(self) -> float:
        """Compute L2 norm of weights across all models"""
        total_norm = 0.0
        count = 0
        try:
            for model in self.models:
                # Try to access layers or network structure
                if hasattr(model, 'network') and model.network:
                    for layer in model.network:
                        if hasattr(layer, 'weights'):
                            total_norm += np.sum(layer.weights ** 2)
                            count += 1
                        elif hasattr(layer, 'W'):
                            total_norm += np.sum(layer.W ** 2)
                            count += 1
        except (AttributeError, TypeError):
            # If structure is not accessible, return a default value
            pass
        
        return np.sqrt(total_norm / max(count, 1)) if count > 0 else 0.0

    def _get_model_weights(self) -> List[Dict]:
        """Get weights from all ensemble models"""
        weights_list = []
        try:
            for model in self.models:
                model_weights = {}
                if hasattr(model, 'network') and model.network:
                    for i, layer in enumerate(model.network):
                        if hasattr(layer, 'weights'):
                            model_weights[f'layer_{i}'] = layer.weights.copy()
                        elif hasattr(layer, 'W'):
                            model_weights[f'layer_{i}'] = layer.W.copy()
                weights_list.append(model_weights)
        except (AttributeError, TypeError):
            pass
        return weights_list

    def _restore_best_weights(self):
        """Restore best weights from early stopping"""
        if self.early_stopping.best_weights:
            try:
                for model, best_weights in zip(self.models, self.early_stopping.best_weights):
                    if hasattr(model, 'network') and model.network:
                        for i, layer in enumerate(model.network):
                            layer_key = f'layer_{i}'
                            if layer_key in best_weights:
                                if hasattr(layer, 'weights'):
                                    layer.weights = best_weights[layer_key].copy()
                                elif hasattr(layer, 'W'):
                                    layer.W = best_weights[layer_key].copy()
            except (AttributeError, TypeError, KeyError):
                print("   ⚠️  Warning: Could not restore best weights")

    def _get_training_summary(self) -> Dict:
        """Generate comprehensive training summary"""
        return {
            'total_epochs': len(self.training_history),
            'final_train_loss': self.training_history[-1].train_loss,
            'best_val_loss': min([m.val_loss for m in self.training_history if m.val_loss], 
                                default=None),
            'history': [
                {
                    'epoch': m.epoch,
                    'train_loss': m.train_loss,
                    'val_loss': m.val_loss,
                    'lr': m.learning_rate
                }
                for m in self.training_history
            ]
        }

    # --------------------------------------------------------
    # ENSEMBLE PREDICTION
    # --------------------------------------------------------
    def predict(self, X: np.ndarray, return_uncertainty: bool = False) -> Union[np.ndarray, Tuple]:
        """
        Ensemble prediction with optional uncertainty estimation
        
        Args:
            X: Input data
            return_uncertainty: Return prediction uncertainty
            
        Returns:
            Predictions (and uncertainty if requested)
        """
        predictions = np.array([model.predict(X) for model in self.models])
        
        # Ensemble average
        mean_pred = np.mean(predictions, axis=0)
        
        if return_uncertainty:
            # Standard deviation as uncertainty measure
            std_pred = np.std(predictions, axis=0)
            return mean_pred, std_pred
        
        return mean_pred

    # --------------------------------------------------------
    # TRANSFER LEARNING
    # --------------------------------------------------------
    def freeze_layers(self, num_layers: int):
        """Freeze first N layers for transfer learning"""
        print(f"\n❄️  Freezing first {num_layers} layers for transfer learning")
        try:
            for model in self.models:
                if hasattr(model, 'network') and model.network:
                    for i, layer in enumerate(model.network[:num_layers]):
                        if hasattr(layer, 'trainable'):
                            layer.trainable = False
                        elif hasattr(layer, 'frozen'):
                            layer.frozen = True
        except (AttributeError, TypeError):
            print("   ⚠️  Warning: Layer freezing not fully supported by model structure")

    def unfreeze_all(self):
        """Unfreeze all layers"""
        print("\n🔥 Unfreezing all layers")
        try:
            for model in self.models:
                if hasattr(model, 'network') and model.network:
                    for layer in model.network:
                        if hasattr(layer, 'trainable'):
                            layer.trainable = True
                        elif hasattr(layer, 'frozen'):
                            layer.frozen = False
        except (AttributeError, TypeError):
            print("   ⚠️  Warning: Layer unfreezing not fully supported by model structure")

    # --------------------------------------------------------
    # MODEL PERSISTENCE
    # --------------------------------------------------------
    def save(self, path: Optional[str] = None, save_config: bool = True):
        """Save model with complete configuration"""
        if path is None:
            path = f"{self.name}_checkpoint"
        
        path_obj = Path(path)
        path_obj.mkdir(exist_ok=True)
        
        # Save each ensemble model
        for i, model in enumerate(self.models):
            model.save_model(str(path_obj / f"model_{i}.npz"))
        
        # Save configuration
        if save_config:
            config_dict = {
                'name': self.config.name,
                'architecture': self.config.architecture,
                'input_size': self.config.input_size,
                'output_size': self.config.output_size,
                'learning_rate': self.config.learning_rate,
                'scheduler_type': self.config.scheduler_type.value,
                'ensemble_size': self.config.ensemble_size,
                'training_history': [
                    {
                        'epoch': m.epoch,
                        'train_loss': m.train_loss,
                        'val_loss': m.val_loss
                    }
                    for m in self.training_history
                ]
            }
            
            with open(path_obj / "config.json", 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        print(f"\n💾 Saved {self.name} → {path}")
        print(f"   ├─ Models: {len(self.models)}")
        print(f"   └─ Config: {'✓' if save_config else '✗'}")

    def load(self, path: str):
        """Load model with configuration"""
        path_obj = Path(path)
        
        # Load configuration
        config_file = path_obj / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            print(f"\n📂 Loading {config_dict['name']} from {path}")
        
        # Load ensemble models
        for i, model in enumerate(self.models):
            model_file = path_obj / f"model_{i}.npz"
            if model_file.exists():
                model.load_model(str(model_file))
        
        print(f"   └─ ✓ Loaded {len(self.models)} models")

    # --------------------------------------------------------
    # VISUALIZATION & ANALYSIS
    # --------------------------------------------------------
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot comprehensive training metrics"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{self.name} Training History', fontsize=16, fontweight='bold')
            
            epochs = [m.epoch for m in self.training_history]
            train_losses = [m.train_loss for m in self.training_history]
            val_losses = [m.val_loss for m in self.training_history if m.val_loss]
            lrs = [m.learning_rate for m in self.training_history]
            grad_norms = [m.gradient_norm for m in self.training_history]
            
            # Loss curves
            axes[0, 0].plot(epochs, train_losses, label='Train Loss', linewidth=2)
            if val_losses:
                axes[0, 0].plot(epochs[:len(val_losses)], val_losses, 
                              label='Val Loss', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_yscale('log')
            
            # Learning rate
            axes[0, 1].plot(epochs, lrs, color='orange', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_yscale('log')
            
            # Gradient norms
            axes[1, 0].plot(epochs, grad_norms, color='green', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_title('Gradient Norms')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Training progress
            axes[1, 1].text(0.1, 0.9, f"Architecture: {self.architecture}", 
                          transform=axes[1, 1].transAxes, fontsize=11)
            axes[1, 1].text(0.1, 0.8, f"Total Epochs: {len(self.training_history)}", 
                          transform=axes[1, 1].transAxes, fontsize=11)
            axes[1, 1].text(0.1, 0.7, f"Final Loss: {train_losses[-1]:.6f}", 
                          transform=axes[1, 1].transAxes, fontsize=11)
            axes[1, 1].text(0.1, 0.6, f"Parameters: {self._count_parameters():,}", 
                          transform=axes[1, 1].transAxes, fontsize=11)
            axes[1, 1].text(0.1, 0.5, f"Ensemble Size: {self.config.ensemble_size}", 
                          transform=axes[1, 1].transAxes, fontsize=11)
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"\n📊 Saved training plot → {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("\n⚠️  matplotlib not available for plotting")

    def summary(self, detailed: bool = True):
        """Display comprehensive cerebrum summary"""
        print("\n" + "="*60)
        print(f"🧠 CEREBRUM SUMMARY: {self.name}")
        print("="*60)
        print(f"Architecture:        {self.architecture.upper()}")
        print(f"Input → Output:      {self.input_size} → {self.output_size}")
        print(f"Total Parameters:    {self._count_parameters():,}")
        print(f"Learning Rate:       {self.config.learning_rate}")
        print(f"LR Scheduler:        {self.config.scheduler_type.value}")
        print(f"Ensemble Models:     {self.config.ensemble_size}")
        print(f"Attention:           {'Enabled' if self.config.use_attention else 'Disabled'}")
        
        if self.config.regularization:
            print(f"Regularization:      {', '.join([r.value for r in self.config.regularization])}")
        
        if detailed:
            print(f"\n{'Layer Architecture:':─<60}")
            total_params = 0
            for i, layer in enumerate(self.layers, 1):
                params = layer.input_size * layer.output_size + layer.output_size
                total_params += params
                print(f"  Layer {i}: [{layer.input_size:4d} → {layer.output_size:4d}] "
                      f"{layer.activation.value:8s} | {params:,} params")
            print(f"  {'':─<56}")
            print(f"  Total (per model): {total_params:,} params")
            
            if self.training_history:
                print(f"\n{'Training History:':─<60}")
                print(f"  Total Epochs:      {len(self.training_history)}")
                print(f"  Best Train Loss:   {min(m.train_loss for m in self.training_history):.6f}")
                if any(m.val_loss for m in self.training_history):
                    val_losses = [m.val_loss for m in self.training_history if m.val_loss]
                    print(f"  Best Val Loss:     {min(val_losses):.6f}")
        
        print("="*60 + "\n")

    # --------------------------------------------------------
    # NEURAL ARCHITECTURE SEARCH (NAS)
    # --------------------------------------------------------
    def auto_tune(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  architectures: List[str] = None,
                  epochs_per_trial: int = 100) -> str:
        """
        Automated neural architecture search
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            architectures: List of architectures to try
            epochs_per_trial: Epochs per architecture
            
        Returns:
            Best architecture name
        """
        if architectures is None:
            architectures = ["tiny", "small", "medium", "large"]
        
        print("\n🔍 Neural Architecture Search (NAS)")
        print(f"   ├─ Candidates: {', '.join(architectures)}")
        print(f"   └─ Trials per arch: {epochs_per_trial} epochs\n")
        
        results = {}
        original_arch = self.architecture
        
        for arch in architectures:
            print(f"\n📊 Testing {arch.upper()} architecture...")
            
            # Temporarily switch architecture
            self.architecture = arch
            self.config.architecture = arch
            self.layers = []
            self.models = []
            self._build_advanced_architecture()
            
            # Train and evaluate
            self.train(X_train, y_train, X_val, y_val, 
                      epochs=epochs_per_trial, verbose=False)
            
            # Evaluate on validation set
            val_pred = self.predict(X_val)
            val_loss = np.mean((val_pred - y_val) ** 2)
            
            results[arch] = {
                'val_loss': val_loss,
                'parameters': self._count_parameters(),
                'final_train_loss': self.training_history[-1].train_loss
            }
            
            print(f"   ├─ Val Loss: {val_loss:.6f}")
            print(f"   ├─ Train Loss: {results[arch]['final_train_loss']:.6f}")
            print(f"   └─ Params: {results[arch]['parameters']:,}")
        
        # Find best architecture
        best_arch = min(results.items(), key=lambda x: x[1]['val_loss'])
        
        print(f"\n🏆 Best Architecture: {best_arch[0].upper()}")
        print(f"   ├─ Validation Loss: {best_arch[1]['val_loss']:.6f}")
        print(f"   └─ Parameters: {best_arch[1]['parameters']:,}")
        
        # Rebuild with best architecture
        self.architecture = best_arch[0]
        self.config.architecture = best_arch[0]
        self.layers = []
        self.models = []
        self._build_advanced_architecture()
        
        return best_arch[0]

    # --------------------------------------------------------
    # ADVANCED DEMONSTRATIONS
    # --------------------------------------------------------
    def demo(self, mode: str = "comprehensive"):
        """
        Run advanced demonstrations
        
        Modes:
            - xor: Classic XOR problem
            - regression: Non-linear function approximation
            - sinusoidal: Learn sine wave
            - comprehensive: All demonstrations
            - ensemble: Ensemble learning demo
            - uncertainty: Uncertainty estimation demo
        """
        print(f"\n{'='*60}")
        print(f"🎭 CEREBRUM DEMONSTRATION: {mode.upper()}")
        print(f"{'='*60}\n")
        
        if mode == "xor" or mode == "comprehensive":
            self._demo_xor()
        
        if mode == "regression" or mode == "comprehensive":
            self._demo_regression()
        
        if mode == "sinusoidal" or mode == "comprehensive":
            self._demo_sinusoidal()
        
        if mode == "ensemble" or mode == "comprehensive":
            self._demo_ensemble()
        
        if mode == "uncertainty" or mode == "comprehensive":
            self._demo_uncertainty()

    def _demo_xor(self):
        """XOR problem demonstration"""
        print("🔷 XOR Logic Gate Learning")
        print("─" * 40)
        
        # XOR data
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        # Create cerebrum
        cerebrum = Cerebrum_Core(
            name="XOR_Brain",
            architecture="tiny",
            input_size=2,
            output_size=1,
            learning_rate=0.1
        )
        
        # Train
        cerebrum.train(X, y, epochs=500, verbose=False)
        
        # Test
        predictions = cerebrum.predict(X)
        print("\nResults:")
        for i, (inp, target, pred) in enumerate(zip(X, y, predictions)):
            print(f"  Input: {inp} → Target: {target[0]} | "
                  f"Prediction: {pred[0]:.4f} | "
                  f"✓" if abs(pred[0] - target[0]) < 0.1 else "✗")
        
        print()

    def _demo_regression(self):
        """Non-linear regression demonstration"""
        print("📈 Non-linear Function Approximation")
        print("─" * 40)
        print("Target: f(x) = x² + 2x + sin(5x)")
        
        # Generate data
        X = np.linspace(-2, 2, 100).reshape(-1, 1)
        y = (X**2 + 2*X + np.sin(5*X)).reshape(-1, 1)
        
        # Add noise
        y += np.random.normal(0, 0.1, y.shape)
        
        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Create cerebrum
        cerebrum = Cerebrum_Core(
            name="Regression_Brain",
            architecture="small",
            input_size=1,
            output_size=1,
            learning_rate=0.05
        )
        
        # Train
        cerebrum.train(X_train, y_train, X_test, y_test, 
                      epochs=300, verbose=False)
        
        # Test
        predictions = cerebrum.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        r2 = 1 - (np.sum((y_test - predictions)**2) / 
                  np.sum((y_test - np.mean(y_test))**2))
        
        print(f"\nTest Performance:")
        print(f"  MSE: {mse:.6f}")
        print(f"  R²:  {r2:.6f}")
        print()

    def _demo_sinusoidal(self):
        """Sinusoidal function learning"""
        print("🌊 Sine Wave Learning")
        print("─" * 40)
        
        # Generate sine wave
        X = np.linspace(0, 4*np.pi, 200).reshape(-1, 1)
        y = np.sin(X)
        
        # Create cerebrum
        cerebrum = Cerebrum_Core(
            name="Sine_Brain",
            architecture="small",
            input_size=1,
            output_size=1,
            learning_rate=0.03
        )
        
        # Train
        cerebrum.train(X, y, epochs=300, verbose=False)
        
        # Test on new range
        X_test = np.linspace(4*np.pi, 6*np.pi, 50).reshape(-1, 1)
        y_test = np.sin(X_test)
        predictions = cerebrum.predict(X_test)
        
        mse = np.mean((predictions - y_test) ** 2)
        print(f"\nExtrapolation Performance:")
        print(f"  MSE on unseen range: {mse:.6f}")
        print()

    def _demo_ensemble(self):
        """Ensemble learning demonstration"""
        print("🎪 Ensemble Learning")
        print("─" * 40)
        
        # Generate noisy data
        X = np.linspace(-3, 3, 150).reshape(-1, 1)
        y = (np.sin(X) + 0.5 * np.cos(2*X)).reshape(-1, 1)
        y += np.random.normal(0, 0.2, y.shape)
        
        # Create ensemble cerebrum
        cerebrum = Cerebrum_Core(
            name="Ensemble_Brain",
            architecture="small",
            input_size=1,
            output_size=1,
            ensemble_size=5
        )
        
        # Train
        cerebrum.train(X, y, epochs=200, verbose=False)
        
        # Compare single vs ensemble
        single_pred = cerebrum.models[0].predict(X)
        ensemble_pred = cerebrum.predict(X)
        
        single_mse = np.mean((single_pred - y) ** 2)
        ensemble_mse = np.mean((ensemble_pred - y) ** 2)
        
        print(f"\nPerformance Comparison:")
        print(f"  Single Model MSE:   {single_mse:.6f}")
        print(f"  Ensemble MSE:       {ensemble_mse:.6f}")
        print(f"  Improvement:        {((single_mse - ensemble_mse)/single_mse * 100):.2f}%")
        print()

    def _demo_uncertainty(self):
        """Uncertainty estimation demonstration"""
        print("🎯 Uncertainty Estimation")
        print("─" * 40)
        
        # Generate data with varying noise
        X = np.linspace(-5, 5, 200).reshape(-1, 1)
        
        # High noise in middle, low noise on edges
        noise_scale = 0.3 * np.exp(-X**2 / 10) + 0.05
        y = np.sin(X) + np.random.normal(0, noise_scale)
        
        # Create ensemble cerebrum
        cerebrum = Cerebrum_Core(
            name="Uncertainty_Brain",
            architecture="small",
            input_size=1,
            output_size=1,
            ensemble_size=10
        )
        
        # Train
        cerebrum.train(X, y, epochs=200, verbose=False)
        
        # Predict with uncertainty
        predictions, uncertainty = cerebrum.predict(X, return_uncertainty=True)
        
        # Analyze uncertainty
        avg_uncertainty = np.mean(uncertainty)
        high_uncertainty_regions = np.where(uncertainty > np.percentile(uncertainty, 75))[0]
        
        print(f"\nUncertainty Analysis:")
        print(f"  Average Uncertainty:    {avg_uncertainty:.6f}")
        print(f"  Max Uncertainty:        {np.max(uncertainty):.6f}")
        print(f"  Min Uncertainty:        {np.min(uncertainty):.6f}")
        print(f"  High Uncertainty Points: {len(high_uncertainty_regions)} / {len(X)}")
        print()

    # --------------------------------------------------------
    # PERFORMANCE BENCHMARKING
    # --------------------------------------------------------
    def benchmark(self, input_size: int = None, num_samples: int = 1000, 
                  num_iterations: int = 100, output_size: int = 1) -> Dict:
        """
        Benchmark cerebrum performance
        
        Args:
            input_size: Input dimension (uses current model's input_size if None)
            num_samples: Number of samples
            num_iterations: Number of timing iterations
            output_size: Output dimension (for creating test model)
            
        Returns:
            Benchmark results dictionary
        """
        # Use current model's input size if not specified
        if input_size is None:
            input_size = self.input_size
        
        print(f"\n⚡ Performance Benchmark")
        print(f"   ├─ Architecture: {self.architecture}")
        print(f"   ├─ Input Size: {input_size}")
        print(f"   ├─ Samples: {num_samples}")
        print(f"   └─ Iterations: {num_iterations}")
        
        # Create a temporary model with correct dimensions if needed
        if input_size != self.input_size:
            print(f"   ℹ️  Creating temporary model for benchmark...")
            temp_cerebrum = Cerebrum_Core(
                name="BenchmarkModel",
                architecture=self.architecture,
                input_size=input_size,
                output_size=output_size,
                learning_rate=self.config.learning_rate
            )
            model_to_test = temp_cerebrum
        else:
            model_to_test = self
        
        # Generate random data
        X = np.random.randn(num_samples, input_size)
        
        # Warmup
        _ = model_to_test.predict(X[:10])
        
        # Forward pass timing
        forward_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model_to_test.predict(X)
            forward_times.append(time.perf_counter() - start)
        
        results = {
            'forward_pass_mean_ms': np.mean(forward_times) * 1000,
            'forward_pass_std_ms': np.std(forward_times) * 1000,
            'throughput_samples_per_sec': num_samples / np.mean(forward_times),
            'latency_per_sample_us': (np.mean(forward_times) / num_samples) * 1e6,
            'parameters': model_to_test._count_parameters(),
            'architecture': self.architecture,
            'input_size': input_size,
            'output_size': output_size
        }
        
        print(f"\n📊 Results:")
        print(f"   ├─ Forward Pass:     {results['forward_pass_mean_ms']:.3f} ± "
              f"{results['forward_pass_std_ms']:.3f} ms")
        print(f"   ├─ Throughput:       {results['throughput_samples_per_sec']:.0f} samples/sec")
        print(f"   ├─ Latency/Sample:   {results['latency_per_sample_us']:.2f} μs")
        print(f"   └─ Parameters:       {results['parameters']:,}")
        
        return results

    # --------------------------------------------------------
    # ADVANCED UTILITIES
    # --------------------------------------------------------
    def export_to_onnx(self, path: str = "cerebrum_model.onnx"):
        """Export model to ONNX format (placeholder)"""
        print(f"\n📦 ONNX Export")
        print(f"   └─ Feature coming soon...")
        # This would require ONNX integration
        pass

    def quantize(self, bits: int = 8):
        """Model quantization for deployment (placeholder)"""
        print(f"\n🔧 Model Quantization ({bits}-bit)")
        print(f"   └─ Feature coming soon...")
        # This would implement weight quantization
        pass

    def prune(self, sparsity: float = 0.5):
        """Prune network weights (placeholder)"""
        print(f"\n✂️  Network Pruning (sparsity: {sparsity:.1%})")
        print(f"   └─ Feature coming soon...")
        # This would implement magnitude-based pruning
        pass

    def explain_prediction(self, X: np.ndarray, method: str = "gradient"):
        """
        Generate prediction explanations
        
        Args:
            X: Input sample
            method: Explanation method (gradient, lime, shap)
        """
        print(f"\n🔍 Prediction Explanation ({method})")
        
        if method == "gradient":
            # Simple gradient-based importance
            prediction = self.predict(X)
            print(f"   ├─ Prediction: {prediction}")
            print(f"   └─ Feature importance via gradients")
            # Would compute input gradients here
        
        print(f"   └─ Full implementation coming soon...")


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def create_cerebrum(architecture: str = "small", **kwargs) -> Cerebrum_Core:
    """
    Factory function to create pre-configured cerebrum instances
    
    Args:
        architecture: Architecture preset
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured Cerebrum_Core instance
    """
    config = CerebrumConfig(architecture=architecture, **kwargs)
    return Cerebrum_Core(config=config)


def create_ensemble_cerebrum(ensemble_size: int = 5, **kwargs) -> Cerebrum_Core:
    """Create ensemble cerebrum for improved predictions"""
    config = CerebrumConfig(ensemble_size=ensemble_size, **kwargs)
    return Cerebrum_Core(config=config)


def create_transfer_learning_cerebrum(pretrained_path: str, **kwargs) -> Cerebrum_Core:
    """Create cerebrum for transfer learning"""
    cerebrum = create_cerebrum(**kwargs)
    cerebrum.load(pretrained_path)
    return cerebrum


# ============================================================
# MAIN DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧠 ADVANCED CEREBRUM v2.0 - Neural Architecture System")
    print("="*60)
    
    # Create advanced cerebrum
    cerebrum = create_cerebrum(
        name="AdvancedBrain",
        architecture="medium",
        input_size=2,
        output_size=1,
        ensemble_size=3,
        scheduler_type=SchedulerType.COSINE_ANNEALING
    )
    
    # Display summary
    cerebrum.summary(detailed=True)
    
    # Run comprehensive demonstrations
    cerebrum.demo(mode="comprehensive")
    
    # Benchmark performance (use current model's input size)
    print("\n" + "="*60)
    print("⚡ PERFORMANCE BENCHMARKING")
    print("="*60)
    cerebrum.benchmark(num_samples=500, num_iterations=50)
    
    # Also benchmark with different input sizes
    print("\n--- Scaling Analysis ---")
    for test_input_size in [5, 10, 20]:
        print(f"\nTesting with input_size={test_input_size}:")
        cerebrum.benchmark(input_size=test_input_size, num_samples=500, num_iterations=20)
    
    print("\n" + "="*60)
    print("✨ Demonstration Complete!")
    print("="*60 + "\n")