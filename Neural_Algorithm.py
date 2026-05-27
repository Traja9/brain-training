import numpy as np
import math
import time
from typing import List, Tuple, Optional, Dict, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

# Import precision modules if available
try:
    from Vectorial_Calculus_ML import (
        UltraPreciseVectorCalculusEngine,
        OptimizationMethod,
        MLOptimizationResult
    )
    CALCULUS_AVAILABLE = True
except ImportError:
    CALCULUS_AVAILABLE = False
    print("Warning: Vectorial Calculus module not available")


# =============================================================
# Pure-NumPy Neural Model wrapper
# Replaces PyTorchNeuralWrapper — identical interface, no torch.
# =============================================================
class NumpyNeuralWrapper:
    """
    Lightweight wrapper that mirrors the old PyTorchNeuralWrapper API.
    Stores layer configs + weight/bias arrays and exposes a forward()
    that applies the same linear → activation pipeline the PyTorch
    version did, but entirely in NumPy.
    """

    def __init__(self):
        self.PYTORCH_AVAILABLE = False   # kept so any legacy checks still work

    # ----------------------------------------------------------
    # Inner model — drop-in replacement for the nn.Module subclass
    # ----------------------------------------------------------
    class _NumpyNet:
        """Stateless feedforward net backed by NumPy arrays."""

        def __init__(self, layer_configs, weights, biases):
            self.layer_configs = layer_configs
            # Deep-copy so external mutation doesn't affect the model
            self.weights  = [w.copy() for w in weights]
            self.biases   = [b.copy() if b is not None else None for b in biases]

        # --- single-sample forward (1-D input) ---
        def _forward_single(self, x: np.ndarray) -> np.ndarray:
            current = x
            for i, config in enumerate(self.layer_configs):
                z = self.weights[i] @ current
                if self.biases[i] is not None:
                    z = z + self.biases[i]
                current = _apply_activation(z, config.activation)
            return current

        # --- public call: handles both 1-D and 2-D (batch) inputs ---
        def __call__(self, X: np.ndarray) -> np.ndarray:
            if X.ndim == 1:
                return self._forward_single(X)
            return np.array([self._forward_single(row) for row in X])

    # ----------------------------------------------------------
    def create_model(self, layer_configs, weights, biases):
        """
        Factory — same signature the old wrapper exposed.
        Returns a callable _NumpyNet instance.
        """
        return self._NumpyNet(layer_configs, weights, biases)


# ---------------------------------------------------------------------------
# Shared activation helper (used by the wrapper AND NeuralDataProcessor)
# ---------------------------------------------------------------------------
def _apply_activation(z: np.ndarray, activation) -> np.ndarray:
    """Stateless activation dispatch — works on any array shape."""
    if activation == ActivationFunction.SIGMOID:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    elif activation == ActivationFunction.TANH:
        return np.tanh(z)
    elif activation == ActivationFunction.RELU:
        return np.maximum(0, z)
    elif activation == ActivationFunction.LEAKY_RELU:
        return np.where(z > 0, z, 0.01 * z)
    elif activation == ActivationFunction.SOFTMAX:
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)
    else:  # LINEAR
        return z


# Global wrapper instance (mirrors the old _pytorch_wrapper drop-in)
_numpy_wrapper = NumpyNeuralWrapper()


# =============================================================
class ActivationFunction(Enum):
    """Neural activation functions"""
    SIGMOID    = "sigmoid"
    TANH       = "tanh"
    RELU       = "relu"
    LEAKY_RELU = "leaky_relu"
    SOFTMAX    = "softmax"
    LINEAR     = "linear"


class LossFunction(Enum):
    """Loss functions for optimization"""
    MEAN_SQUARED_ERROR      = "mse"
    CROSS_ENTROPY           = "cross_entropy"
    BINARY_CROSS_ENTROPY    = "binary_cross_entropy"
    HUBER                   = "huber"


@dataclass
class LayerConfig:
    """Configuration for a neural network layer"""
    input_size:     int
    output_size:    int
    activation:     ActivationFunction = ActivationFunction.SIGMOID
    use_bias:       bool  = True
    dropout_rate:   float = 0.0


@dataclass
class TrainingMetrics:
    """Metrics tracked during training"""
    epoch:                int          = 0
    loss_history:         List[float]  = field(default_factory=list)
    accuracy_history:     List[float]  = field(default_factory=list)
    gradient_norms:       List[float]  = field(default_factory=list)
    learning_rates:       List[float]  = field(default_factory=list)
    computation_time_ms:  float        = 0.0
    convergence_status:   str          = "training"


# =============================================================
class NeuralDataProcessor:
    """
    Neural network data processing algorithm using physics-inspired equations.
    Implements backpropagation with ultra-precise gradient computation.
    Fully self-contained in NumPy — no external deep-learning framework needed.
    """

    def __init__(self,
                 layer_configs: List[LayerConfig],
                 learning_rate: float = 0.01,
                 loss_function: LossFunction = LossFunction.MEAN_SQUARED_ERROR,
                 use_precision_calculus: bool = True,
                 optimization_method: str = "adam"):
        """
        Initialize neural data processor.

        Args:
            layer_configs:            List of layer configurations
            learning_rate:            Learning rate for optimisation
            loss_function:            Loss function to minimise
            use_precision_calculus:   Use ultra-precise gradient computation
            optimization_method:      Optimisation algorithm  (sgd | adam | momentum)
        """
        self.layer_configs        = layer_configs
        self.learning_rate        = learning_rate
        self.loss_function        = loss_function
        self.use_precision_calculus = use_precision_calculus and CALCULUS_AVAILABLE
        self.optimization_method  = optimization_method

        # Initialize weights and biases
        self.weights: List[np.ndarray] = []
        self.biases:  List[Optional[np.ndarray]] = []
        self._initialize_parameters()

        # Initialize precision calculus engine
        if self.use_precision_calculus:
            self.calculus_engine = UltraPreciseVectorCalculusEngine(
                precision_mode="maximum",
                derivative_step_size=1e-8,
                gradient_tolerance=1e-10
            )

        # Adam optimizer states
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases  = [np.zeros_like(b) for b in self.biases if b is not None]
        # keep None placeholders so indexing stays aligned
        self.m_biases  = [np.zeros_like(b) if b is not None else None for b in self.biases]
        self.v_biases  = [np.zeros_like(b) if b is not None else None for b in self.biases]
        self.adam_t    = 0

        # Training metrics
        self.metrics = TrainingMetrics()

        # Cache for forward pass
        self.layer_outputs: List[np.ndarray] = []
        self.layer_inputs:  List[np.ndarray] = []

        # Frozen-layer set (populated by load_pretrained_weights)
        self.frozen_layers: set = set()

    # ----------------------------------------------------------
    # to_numpy_model  —  replaces  to_pytorch()
    # ----------------------------------------------------------
    def to_numpy_model(self):
        """
        Create a standalone NumPy forward-only model from current weights.
        Equivalent to the old to_pytorch() — returns a callable that
        accepts a single sample (1-D) or a batch (2-D) and returns predictions.
        """
        return _numpy_wrapper.create_model(
            self.layer_configs, self.weights, self.biases
        )

    # ----------------------------------------------------------
    # from_numpy_model  —  replaces  from_pytorch()
    # ----------------------------------------------------------
    def from_numpy_model(self, numpy_model):
        """
        Import weights from a standalone NumPy model back into this processor.
        Mirrors the old from_pytorch() so the round-trip workflow is preserved.
        """
        for i in range(len(self.weights)):
            self.weights[i] = numpy_model.weights[i].copy()
            if self.biases[i] is not None and numpy_model.biases[i] is not None:
                self.biases[i] = numpy_model.biases[i].copy()

    # ----------------------------------------------------------
    # Parameter initialisation
    # ----------------------------------------------------------
    def _initialize_parameters(self, init_method: str = "auto"):
        """
        Initialize network parameters with advanced initialisation strategies.

        Args:
            init_method:
                - "auto"             : picks best method per activation
                - "xavier_uniform"   : Glorot uniform
                - "xavier_normal"    : Glorot normal
                - "he_uniform"       : He uniform  (ReLU)
                - "he_normal"        : He normal   (ReLU)
                - "lecun_normal"     : LeCun normal (SELU)
                - "orthogonal"       : Orthogonal  (deep nets / RNNs)
                - "zeros"            : All zeros
                - "ones"             : All ones × 0.01
        """
        for i, config in enumerate(self.layer_configs):
            # --- auto-select ---
            if init_method == "auto":
                if config.activation in (ActivationFunction.RELU,
                                         ActivationFunction.LEAKY_RELU):
                    method = "he_normal"
                elif config.activation in (ActivationFunction.TANH,
                                           ActivationFunction.SIGMOID):
                    method = "xavier_normal"
                else:                          # LINEAR, SOFTMAX
                    method = "xavier_normal"
            else:
                method = init_method

            n_in  = config.input_size
            n_out = config.output_size

            # --- compute weight matrix ---
            if method == "xavier_uniform":
                limit   = np.sqrt(6.0 / (n_in + n_out))
                weights = np.random.uniform(-limit, limit, (n_out, n_in))

            elif method == "xavier_normal":
                std     = np.sqrt(2.0 / (n_in + n_out))
                weights = np.random.randn(n_out, n_in) * std

            elif method == "he_uniform":
                limit   = np.sqrt(6.0 / n_in)
                weights = np.random.uniform(-limit, limit, (n_out, n_in))

            elif method == "he_normal":
                std     = np.sqrt(2.0 / n_in)
                weights = np.random.randn(n_out, n_in) * std

            elif method == "lecun_normal":
                std     = np.sqrt(1.0 / n_in)
                weights = np.random.randn(n_out, n_in) * std

            elif method == "orthogonal":
                a = np.random.randn(n_out, n_in)
                if n_out <= n_in:
                    q, _ = np.linalg.qr(a.T)
                    weights = q.T[:n_out, :]
                else:
                    q, _ = np.linalg.qr(a)
                    weights = q[:, :n_in]
                gain    = self._get_activation_gain(config.activation)
                weights = weights * gain

            elif method == "zeros":
                weights = np.zeros((n_out, n_in))
                print(f"Warning: Layer {i} initialized to zeros (may not learn)")

            elif method == "ones":
                weights = np.ones((n_out, n_in)) * 0.01

            else:
                raise ValueError(f"Unknown initialization method: {method}")

            self.weights.append(weights)

            # --- biases ---
            if config.use_bias:
                if config.activation in (ActivationFunction.RELU,
                                         ActivationFunction.LEAKY_RELU):
                    biases = np.ones(n_out) * 0.01   # small positive for ReLU
                else:
                    biases = np.zeros(n_out)
                self.biases.append(biases)
            else:
                self.biases.append(None)

        # Summary
        total_params = (sum(w.size for w in self.weights) +
                        sum(b.size for b in self.biases if b is not None))
        print(f"Initialized {len(self.weights)} layers with {total_params:,} "
              f"parameters using '{init_method}' method")

    # ----------------------------------------------------------
    def _get_activation_gain(self, activation: ActivationFunction) -> float:
        """Gain (scaling factor) for orthogonal init per activation."""
        gain_map = {
            ActivationFunction.LINEAR:     1.0,
            ActivationFunction.SIGMOID:    1.0,
            ActivationFunction.TANH:       5.0 / 3.0,
            ActivationFunction.RELU:       np.sqrt(2.0),
            ActivationFunction.LEAKY_RELU: np.sqrt(2.0 / (1 + 0.01**2)),
            ActivationFunction.SOFTMAX:    1.0,
        }
        return gain_map.get(activation, 1.0)

    # ----------------------------------------------------------
    # Pre-trained weight loading / saving
    # ----------------------------------------------------------
    def load_pretrained_weights(self, weights_dict: Dict[str, np.ndarray],
                                freeze_layers: Optional[List[int]] = None):
        """
        Load pre-trained weights into the network.

        Args:
            weights_dict:   keys like "layer_0_weights", "layer_0_bias"
            freeze_layers:  layer indices to freeze during training
        """
        for i in range(len(self.weights)):
            weight_key = f"layer_{i}_weights"
            bias_key   = f"layer_{i}_bias"

            if weight_key in weights_dict:
                loaded = weights_dict[weight_key]
                if loaded.shape == self.weights[i].shape:
                    self.weights[i] = loaded.copy()
                    print(f"Loaded weights for layer {i}: {loaded.shape}")
                else:
                    print(f"Warning: Shape mismatch for layer {i}. "
                          f"Expected {self.weights[i].shape}, got {loaded.shape}")

            if bias_key in weights_dict and self.biases[i] is not None:
                loaded = weights_dict[bias_key]
                if loaded.shape == self.biases[i].shape:
                    self.biases[i] = loaded.copy()
                    print(f"Loaded bias for layer {i}: {loaded.shape}")

        self.frozen_layers = set(freeze_layers) if freeze_layers else set()
        if freeze_layers:
            print(f"Frozen layers: {freeze_layers}")

    def save_weights(self, filepath: str):
        """Save model weights + metadata to .npz"""
        weights_dict = {}
        for i in range(len(self.weights)):
            weights_dict[f"layer_{i}_weights"] = self.weights[i]
            if self.biases[i] is not None:
                weights_dict[f"layer_{i}_bias"] = self.biases[i]

        weights_dict["metadata"] = {
            "layer_configs": [(c.input_size, c.output_size, c.activation.value)
                              for c in self.layer_configs],
            "total_parameters": (sum(w.size for w in self.weights) +
                                 sum(b.size for b in self.biases if b is not None))
        }
        np.savez(filepath, **weights_dict)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath: str):
        """Load model weights from .npz"""
        data = np.load(filepath, allow_pickle=True)
        weights_dict = {k: data[k] for k in data.files if k != "metadata"}
        self.load_pretrained_weights(weights_dict)
        print(f"Model weights loaded from {filepath}")

    # ----------------------------------------------------------
    # Activation functions  &  their derivatives
    # ----------------------------------------------------------
    def _activation(self, x: np.ndarray, func_type: ActivationFunction) -> np.ndarray:
        """
        Apply activation function.
        These are inspired by neuron firing dynamics.
        """
        if func_type == ActivationFunction.SIGMOID:
            # σ(x) = 1 / (1 + e^(-x))
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif func_type == ActivationFunction.TANH:
            # tanh(x) = (e^x − e^(−x)) / (e^x + e^(−x))
            return np.tanh(x)
        elif func_type == ActivationFunction.RELU:
            # ReLU(x) = max(0, x)
            return np.maximum(0, x)
        elif func_type == ActivationFunction.LEAKY_RELU:
            # Leaky ReLU(x) = x if x > 0 else 0.01·x
            return np.where(x > 0, x, 0.01 * x)
        elif func_type == ActivationFunction.SOFTMAX:
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)
        else:   # LINEAR
            return x

    def _activation_derivative(self, x: np.ndarray,
                               func_type: ActivationFunction) -> np.ndarray:
        """Compute derivative of activation function."""
        if func_type == ActivationFunction.SIGMOID:
            sig = self._activation(x, func_type)
            return sig * (1 - sig)
        elif func_type == ActivationFunction.TANH:
            return 1 - np.tanh(x) ** 2
        elif func_type == ActivationFunction.RELU:
            return np.where(x > 0, 1.0, 0.0)
        elif func_type == ActivationFunction.LEAKY_RELU:
            return np.where(x > 0, 1.0, 0.01)
        elif func_type == ActivationFunction.SOFTMAX:
            # Full Jacobian handled inside backprop; scalar placeholder here
            return np.ones_like(x)
        else:   # LINEAR
            return np.ones_like(x)

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through network.
        Analogous to signal propagation through a neural circuit.

            h^(l) = σ( W^(l) · h^(l-1) + b^(l) )
        """
        self.layer_inputs  = [X]
        self.layer_outputs = [X]

        current_input = X
        for i, config in enumerate(self.layer_configs):
            z = np.dot(self.weights[i], current_input)
            if self.biases[i] is not None:
                z += self.biases[i]
            self.layer_inputs.append(z)

            output = self._activation(z, config.activation)
            self.layer_outputs.append(output)
            current_input = output

        return current_input

    # ----------------------------------------------------------
    # Loss computation
    # ----------------------------------------------------------
    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute loss function.
        Represents energy / error in the system.
        """
        if self.loss_function == LossFunction.MEAN_SQUARED_ERROR:
            # L = ½ Σ(ŷ − y)²
            return 0.5 * np.mean((y_pred - y_true) ** 2)

        elif self.loss_function == LossFunction.CROSS_ENTROPY:
            # L = −Σ y · log(ŷ)
            eps = 1e-10
            return -np.sum(y_true * np.log(y_pred + eps))

        elif self.loss_function == LossFunction.BINARY_CROSS_ENTROPY:
            # L = −[ y·log(p) + (1−y)·log(1−p) ]
            eps = 1e-10
            return -np.mean(
                y_true * np.log(y_pred + eps) +
                (1 - y_true) * np.log(1 - y_pred + eps)
            )

        elif self.loss_function == LossFunction.HUBER:
            delta = 1.0
            err   = y_pred - y_true
            return np.mean(
                np.where(np.abs(err) <= delta,
                         0.5 * err ** 2,
                         delta * (np.abs(err) - 0.5 * delta))
            )

    # ----------------------------------------------------------
    # Backward pass  —  NumPy chain-rule backprop
    # ----------------------------------------------------------
    def backward(self, X: np.ndarray,
                 y_true: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backpropagation (chain rule / energy minimisation).

            ∂L/∂W^(l) = ∂L/∂h^(l) · ∂h^(l)/∂z^(l) · ∂z^(l)/∂W^(l)
        """
        y_pred = self.layer_outputs[-1]

        # Output-layer error (works for MSE, CE, BCE)
        if self.loss_function == LossFunction.MEAN_SQUARED_ERROR:
            delta = y_pred - y_true
        elif self.loss_function == LossFunction.CROSS_ENTROPY:
            delta = y_pred - y_true   # simplified for softmax output
        else:
            delta = y_pred - y_true

        weight_gradients = [None] * len(self.weights)
        bias_gradients   = [None] * len(self.biases)

        for i in reversed(range(len(self.layer_configs))):
            # ∂L/∂W = δ · h^(l-1)ᵀ
            weight_gradients[i] = np.outer(delta, self.layer_outputs[i])

            if self.biases[i] is not None:
                bias_gradients[i] = delta.copy()

            # Propagate error backwards
            if i > 0:
                # δ^(l-1) = Wᵀ · δ^(l) ⊙ σ′(z^(l-1))
                delta  = np.dot(self.weights[i].T, delta)
                delta *= self._activation_derivative(
                    self.layer_inputs[i],
                    self.layer_configs[i - 1].activation
                )

        return weight_gradients, bias_gradients

    # ----------------------------------------------------------
    # backward_vectorised  —  replaces  backward_pytorch()
    # Performs the exact same gradient computation but processes
    # the entire batch in one vectorised NumPy pass instead of
    # sample-by-sample.  Mirrors the old GPU-accelerated path.
    # ----------------------------------------------------------
    def backward_vectorised(self, X_batch: np.ndarray,
                            y_batch: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Vectorised batch backward pass — pure NumPy.
        Equivalent to the old backward_pytorch() that used autograd on a batch.

        Args:
            X_batch: (batch_size, n_features)
            y_batch: (batch_size, n_outputs)

        Returns:
            Averaged weight_gradients, bias_gradients over the batch.
        """
        batch_size = X_batch.shape[0]

        # --- vectorised forward: cache per-sample intermediates ---
        # all_inputs[layer][sample]  =  z  (pre-activation)
        # all_outputs[layer][sample] =  a  (post-activation)
        all_inputs  = []          # length = n_layers
        all_outputs = [X_batch]   # layer 0 output = raw input

        A = X_batch                           # (batch, features)
        for i, config in enumerate(self.layer_configs):
            # Z = A · Wᵀ  +  b        →  (batch, n_out)
            Z = A @ self.weights[i].T
            if self.biases[i] is not None:
                Z += self.biases[i][np.newaxis, :]
            all_inputs.append(Z)

            # activation — applied element-wise (softmax row-wise)
            if config.activation == ActivationFunction.SOFTMAX:
                exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
                A     = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
            else:
                A = self._activation(Z, config.activation)
            all_outputs.append(A)

        # --- output error delta  (batch, n_out) ---
        y_pred = all_outputs[-1]
        delta  = y_pred - y_batch              # works for MSE / CE / BCE

        # --- backprop through layers ---
        weight_gradients = [None] * len(self.weights)
        bias_gradients   = [None] * len(self.biases)

        for i in reversed(range(len(self.layer_configs))):
            # ∂L/∂W  =  (1/B) · δᵀ · A^(l-1)   →  (n_out, n_in)
            weight_gradients[i] = (delta.T @ all_outputs[i]) / batch_size

            if self.biases[i] is not None:
                bias_gradients[i] = delta.mean(axis=0)

            if i > 0:
                # propagate: δ^(l-1) = δ · W  ⊙  σ′(z^(l-1))
                delta = delta @ self.weights[i]
                delta *= self._activation_derivative(
                    all_inputs[i - 1],
                    self.layer_configs[i - 1].activation
                )

        return weight_gradients, bias_gradients

    # ----------------------------------------------------------
    # Optimiser updates
    # ----------------------------------------------------------
    def _update_parameters_sgd(self, weight_grads: List[np.ndarray],
                               bias_grads:  List[np.ndarray]):
        """Standard SGD update."""
        for i in range(len(self.weights)):
            if i in self.frozen_layers:
                continue
            self.weights[i] -= self.learning_rate * weight_grads[i]
            if self.biases[i] is not None:
                self.biases[i] -= self.learning_rate * bias_grads[i]

    def _update_parameters_adam(self, weight_grads: List[np.ndarray],
                                bias_grads:  List[np.ndarray]):
        """
        Adam optimiser  (Adaptive Moment Estimation).
        Physics-inspired: momentum + adaptive learning.
        """
        self.adam_t += 1
        beta1, beta2 = 0.9, 0.999
        epsilon      = 1e-8

        for i in range(len(self.weights)):
            if i in self.frozen_layers:
                continue

            # first moment (momentum)
            self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * weight_grads[i]
            # second moment (velocity)
            self.v_weights[i] = beta2 * self.v_weights[i] + (1 - beta2) * (weight_grads[i] ** 2)

            # bias-corrected estimates
            m_hat = self.m_weights[i] / (1 - beta1 ** self.adam_t)
            v_hat = self.v_weights[i] / (1 - beta2 ** self.adam_t)

            self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            # biases
            if self.biases[i] is not None:
                self.m_biases[i] = beta1 * self.m_biases[i] + (1 - beta1) * bias_grads[i]
                self.v_biases[i] = beta2 * self.v_biases[i] + (1 - beta2) * (bias_grads[i] ** 2)

                m_hat_b = self.m_biases[i] / (1 - beta1 ** self.adam_t)
                v_hat_b = self.v_biases[i] / (1 - beta2 ** self.adam_t)

                self.biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

    # ----------------------------------------------------------
    # Main training loop  (enhanced)
    # ----------------------------------------------------------
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 100,
              batch_size: Optional[int] = None,
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              verbose: bool = True,
              early_stopping: bool = True,
              patience: int = 20,
              min_delta: float = 1e-6,
              lr_schedule: str = "reduce_on_plateau",
              lr_patience: int = 10,
              lr_factor: float = 0.5,
              lr_min: float = 1e-7,
              gradient_clip: Optional[float] = 5.0,
              save_best: bool = True,
              checkpoint_path: str = "best_model.npz"):
        """
        Enhanced training with early stopping, LR scheduling, and checkpointing.

        Args:
            X_train:            Training inputs   (n_samples, n_features)
            y_train:            Training targets  (n_samples, n_outputs)
            epochs:             Max training epochs
            batch_size:         Mini-batch size (None = full batch)
            validation_data:    Optional (X_val, y_val) tuple
            verbose:            Print progress
            early_stopping:     Enable early stopping
            patience:           Epochs without improvement before stop
            min_delta:          Min change to count as improvement
            lr_schedule:        "reduce_on_plateau" | "cosine" | "step" | "none"
            lr_patience:        Epochs before LR reduction
            lr_factor:          LR reduction multiplier
            lr_min:             Minimum learning rate floor
            gradient_clip:      Max gradient norm (None = disabled)
            save_best:          Checkpoint best weights
            checkpoint_path:    File path for best-model checkpoint
        """
        start_time = time.perf_counter()
        n_samples  = X_train.shape[0]

        if batch_size is None:
            batch_size = n_samples

        # --- early-stopping state ---
        best_loss      = float('inf')
        best_val_loss  = float('inf')
        best_weights   = None
        best_biases    = None
        epochs_without_improvement = 0

        # --- LR-schedule state ---
        lr_epochs_without_improvement = 0
        initial_lr = self.learning_rate

        if verbose:
            print("=" * 70)
            print("NEURAL DATA PROCESSING - ENHANCED TRAINING")
            print("=" * 70)
            print(f"Samples: {n_samples}, Features: {X_train.shape[1]}")
            print(f"Architecture: {' → '.join(str(c.output_size) for c in self.layer_configs)}")
            print(f"Optimizer: {self.optimization_method}")
            print(f"Loss: {self.loss_function.value}")
            print(f"Learning Rate: {self.learning_rate} (schedule: {lr_schedule})")
            print(f"Early Stopping: {'Enabled' if early_stopping else 'Disabled'} (patience: {patience})")
            print(f"Gradient Clipping: {gradient_clip if gradient_clip else 'Disabled'}")
            print("-" * 70)

        try:
            for epoch in range(epochs):
                epoch_start = time.perf_counter()
                epoch_loss  = 0.0

                # shuffle
                indices    = np.random.permutation(n_samples)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]

                num_batches = 0
                for i in range(0, n_samples, batch_size):
                    batch_X = X_shuffled[i:i + batch_size]
                    batch_y = y_shuffled[i:i + batch_size]

                    batch_loss         = 0.0
                    weight_grads_sum   = [np.zeros_like(w) for w in self.weights]
                    bias_grads_sum     = [np.zeros_like(b) if b is not None else None
                                          for b in self.biases]

                    # per-sample forward + backward
                    for j in range(len(batch_X)):
                        y_pred = self.forward(batch_X[j])
                        loss   = self._compute_loss(y_pred, batch_y[j])
                        batch_loss += loss

                        w_grads, b_grads = self.backward(batch_X[j], batch_y[j])

                        for k in range(len(self.weights)):
                            weight_grads_sum[k] += w_grads[k]
                            if bias_grads_sum[k] is not None:
                                bias_grads_sum[k] += b_grads[k]

                    # average gradients
                    for k in range(len(self.weights)):
                        weight_grads_sum[k] /= len(batch_X)
                        if bias_grads_sum[k] is not None:
                            bias_grads_sum[k] /= len(batch_X)

                    # gradient clipping
                    if gradient_clip is not None:
                        grad_norm = np.sqrt(sum(np.sum(g ** 2) for g in weight_grads_sum))
                        if grad_norm > gradient_clip:
                            clip_factor = gradient_clip / (grad_norm + 1e-8)
                            for k in range(len(self.weights)):
                                weight_grads_sum[k] *= clip_factor
                                if bias_grads_sum[k] is not None:
                                    bias_grads_sum[k] *= clip_factor

                    # parameter update
                    if self.optimization_method == "adam":
                        self._update_parameters_adam(weight_grads_sum, bias_grads_sum)
                    else:
                        self._update_parameters_sgd(weight_grads_sum, bias_grads_sum)

                    epoch_loss  += batch_loss / len(batch_X)
                    num_batches += 1

                epoch_loss /= num_batches

                # gradient norm for monitoring (reuses last batch grads)
                grad_norm = np.sqrt(sum(np.sum(w ** 2) for w in weight_grads_sum))

                # --- validation ---
                val_loss     = None
                val_accuracy = None
                if validation_data is not None:
                    val_predictions = self.predict(validation_data[0])
                    val_loss = np.mean([
                        self._compute_loss(val_predictions[idx], validation_data[1][idx])
                        for idx in range(len(validation_data[0]))
                    ])
                    val_accuracy = self.evaluate(validation_data[0], validation_data[1])
                    self.metrics.accuracy_history.append(val_accuracy)

                # --- metrics ---
                self.metrics.loss_history.append(epoch_loss)
                self.metrics.gradient_norms.append(grad_norm)
                self.metrics.learning_rates.append(self.learning_rate)
                self.metrics.epoch = epoch + 1

                # --- LR scheduling ---
                monitor_loss = val_loss if val_loss is not None else epoch_loss

                if lr_schedule == "reduce_on_plateau":
                    if monitor_loss < best_val_loss - min_delta:
                        lr_epochs_without_improvement = 0
                        best_val_loss = monitor_loss
                    else:
                        lr_epochs_without_improvement += 1

                    if lr_epochs_without_improvement >= lr_patience:
                        old_lr = self.learning_rate
                        self.learning_rate = max(self.learning_rate * lr_factor, lr_min)
                        if verbose and self.learning_rate != old_lr:
                            print(f"  → Learning rate reduced: {old_lr:.6f} → {self.learning_rate:.6f}")
                        lr_epochs_without_improvement = 0

                elif lr_schedule == "cosine":
                    self.learning_rate = lr_min + (initial_lr - lr_min) * \
                        (1 + np.cos(np.pi * epoch / epochs)) / 2

                elif lr_schedule == "step":
                    if epoch > 0 and epoch % 50 == 0:
                        self.learning_rate = max(self.learning_rate * lr_factor, lr_min)

                # --- early stopping logic ---
                improved = False
                if monitor_loss < best_loss - min_delta:
                    best_loss = monitor_loss
                    epochs_without_improvement = 0
                    improved = True
                    if save_best:
                        best_weights = [w.copy() for w in self.weights]
                        best_biases  = [b.copy() if b is not None else None for b in self.biases]
                else:
                    epochs_without_improvement += 1

                # --- print progress ---
                if verbose and (epoch % 10 == 0 or epoch == epochs - 1 or improved):
                    elapsed = (time.perf_counter() - epoch_start) * 1000
                    status  = "✓" if improved else " "

                    info_str = f"{status} Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}"
                    if val_loss is not None:
                        info_str += f" | Val Loss: {val_loss:.6f}"
                    if val_accuracy is not None:
                        info_str += f" | Val Acc: {val_accuracy:.4f}"
                    info_str += (f" | Grad: {grad_norm:.4f} "
                                 f"| LR: {self.learning_rate:.6f} | {elapsed:.1f}ms")
                    print(info_str)

                # --- check early stop ---
                if early_stopping and epochs_without_improvement >= patience:
                    if verbose:
                        print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                        print(f"  No improvement for {patience} epochs (best loss: {best_loss:.6f})")
                    break

            # --- restore best weights ---
            if save_best and best_weights is not None:
                self.weights = best_weights
                self.biases  = best_biases
                if verbose:
                    print(f"\n✓ Restored best model weights (loss: {best_loss:.6f})")
                try:
                    self.save_weights(checkpoint_path)
                    if verbose:
                        print(f"✓ Best model saved to '{checkpoint_path}'")
                except Exception as e:
                    if verbose:
                        print(f"⚠ Could not save checkpoint: {e}")

            self.metrics.computation_time_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.convergence_status = (
                "completed" if epochs_without_improvement < patience else "early_stopped"
            )

            if verbose:
                print("-" * 70)
                print(f"Training complete! Total time: {self.metrics.computation_time_ms:.2f}ms")
                print(f"Final loss: {epoch_loss:.6f} | Best loss: {best_loss:.6f}")
                print(f"Status: {self.metrics.convergence_status}")
                print("=" * 70)

        except KeyboardInterrupt:
            if verbose:
                print("\n\n⚠ Training interrupted by user")
                print(f"Completed {epoch + 1}/{epochs} epochs")
            self.metrics.convergence_status = "interrupted"

    # ----------------------------------------------------------
    # train_two_phase  —  replaces  train_hybrid()
    # Old logic: Phase-1 = PyTorch fast bulk train, Phase-2 = NumPy fine-tune.
    # New logic: Phase-1 = vectorised batch NumPy train (same bulk speed
    #            advantage without a framework), Phase-2 = per-sample
    #            fine-tuning with full gradient precision.  Identical
    #            user-facing signature and behaviour.
    # ----------------------------------------------------------
    def train_two_phase(self, X_train: np.ndarray, y_train: np.ndarray,
                        epochs: int = 100,
                        phase1_epochs: int = 50,
                        batch_size: int = 32,
                        verbose: bool = True):
        """
        Two-phase training strategy (replaces train_hybrid).

        Phase 1 — vectorised batch training (fast, bulk updates).
        Phase 2 — per-sample fine-tuning   (precise gradients).

        Args:
            X_train:        Training inputs
            y_train:        Training targets
            epochs:         Total epochs across both phases
            phase1_epochs:  Epochs allocated to Phase 1
            batch_size:     Mini-batch size for Phase 1
            verbose:        Print progress
        """
        if verbose:
            print(f"Two-Phase Training: {phase1_epochs} vectorised epochs + "
                  f"{epochs - phase1_epochs} fine-tune epochs")

        # ---- Phase 1: fast vectorised updates ----
        n_samples = X_train.shape[0]
        if verbose:
            print("\nPhase 1: Vectorised batch training …")
            print("-" * 70)

        for epoch in range(phase1_epochs):
            indices    = np.random.permutation(n_samples)
            X_shuf     = X_train[indices]
            y_shuf     = y_train[indices]
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n_samples, batch_size):
                bX = X_shuf[start:start + batch_size]
                bY = y_shuf[start:start + batch_size]

                # vectorised backward gives averaged grads directly
                w_grads, b_grads = self.backward_vectorised(bX, bY)

                # gradient clipping
                grad_norm = np.sqrt(sum(np.sum(g ** 2) for g in w_grads))
                if grad_norm > 5.0:
                    clip = 5.0 / (grad_norm + 1e-8)
                    w_grads = [g * clip for g in w_grads]
                    b_grads = [g * clip if g is not None else None for g in b_grads]

                # update
                if self.optimization_method == "adam":
                    self._update_parameters_adam(w_grads, b_grads)
                else:
                    self._update_parameters_sgd(w_grads, b_grads)

                # loss (use last sample in batch as representative)
                _ = self.forward(bX[-1])
                epoch_loss += self._compute_loss(self.layer_outputs[-1], bY[-1])
                n_batches  += 1

            if verbose and (epoch % 10 == 0 or epoch == phase1_epochs - 1):
                print(f"  Phase-1 Epoch {epoch+1}/{phase1_epochs} | Loss ≈ {epoch_loss/n_batches:.6f}")

        # ---- Phase 2: precise per-sample fine-tuning ----
        remaining = epochs - phase1_epochs
        if remaining > 0:
            if verbose:
                print(f"\nPhase 2: Fine-tuning with per-sample precision ({remaining} epochs) …")
            self.train(X_train, y_train, epochs=remaining,
                       batch_size=batch_size, verbose=verbose)

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data (single sample or batch)."""
        if X.ndim == 1:
            return self.forward(X)
        return np.array([self.forward(X[i]) for i in range(X.shape[0])])

    # ----------------------------------------------------------
    # predict_batch_vectorised  —  replaces  predict_batch_pytorch()
    # Vectorised NumPy forward pass over the whole batch at once.
    # ----------------------------------------------------------
    def predict_batch_vectorised(self, X: np.ndarray) -> np.ndarray:
        """
        Fast batch prediction — vectorised NumPy.
        Equivalent to the old predict_batch_pytorch() that pushed
        the entire batch through the GPU in one shot.

        Args:
            X: (batch_size, n_features)

        Returns:
            predictions: (batch_size, n_outputs)
        """
        A = X                                          # (batch, features)
        for i, config in enumerate(self.layer_configs):
            Z = A @ self.weights[i].T
            if self.biases[i] is not None:
                Z += self.biases[i][np.newaxis, :]

            if config.activation == ActivationFunction.SOFTMAX:
                exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
                A     = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
            else:
                A = self._activation(Z, config.activation)
        return A

    # ----------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate accuracy on test data."""
        predictions = self.predict(X_test)

        if predictions.ndim > 1 and predictions.shape[1] > 1:
            # Multi-class classification
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
            return float(np.mean(pred_classes == true_classes))
        else:
            # Binary classification / regression
            return float(np.mean(np.abs(predictions - y_test) < 0.5))

    # ----------------------------------------------------------
    # benchmark  —  replaces  benchmark_pytorch_vs_numpy()
    # Benchmarks two NumPy strategies: per-sample loop vs vectorised.
    # ----------------------------------------------------------
    def benchmark(self, X_test: np.ndarray,
                  iterations: int = 100) -> Dict[str, float]:
        """
        Compare inference speed: per-sample loop vs vectorised batch.
        Replaces the old benchmark_pytorch_vs_numpy().

        Args:
            X_test:     (n_samples, n_features)
            iterations: Number of timing runs

        Returns:
            Dict with loop_ms, vectorised_ms, speedup keys.
        """
        results: Dict[str, float] = {}

        # --- per-sample loop (original predict path) ---
        start = time.perf_counter()
        for _ in range(iterations):
            _ = self.predict(X_test)
        results["loop_ms"] = (time.perf_counter() - start) * 1000 / iterations

        # --- vectorised batch (new fast path) ---
        start = time.perf_counter()
        for _ in range(iterations):
            _ = self.predict_batch_vectorised(X_test)
        results["vectorised_ms"] = (time.perf_counter() - start) * 1000 / iterations

        results["speedup"] = results["loop_ms"] / (results["vectorised_ms"] + 1e-12)

        return results

    # =============================================================
    # 📦 MODEL SAVE / LOAD UTILITIES  (robust .npz version)
    # =============================================================
    def save_model(self, path: str):
        """
        Save current weights & biases to a .npz file.
        Supports variable layer shapes via object arrays.
        """
        if not hasattr(self, "weights") or not hasattr(self, "biases"):
            raise RuntimeError("Model has no weights or biases to save — train it first.")

        weights_obj = np.array(self.weights, dtype=object)
        biases_obj  = np.array(self.biases,  dtype=object)

        np.savez(path, weights=weights_obj, biases=biases_obj, allow_pickle=True)
        print(f"[💾 NeuralDataProcessor] Model weights saved to {path}")

    def load_model(self, path: str):
        """
        Load weights & biases from a .npz file (supports object arrays).
        """
        try:
            data         = np.load(path, allow_pickle=True)
            self.weights = list(data["weights"])
            self.biases  = list(data["biases"])
            print(f"[📂 NeuralDataProcessor] Model weights loaded from {path}")
        except Exception as e:
            print(f"[⚠️  NeuralDataProcessor] Failed to load model: {e}")


# =============================================================
# Demo: core data processing  (unchanged logic)
# =============================================================
def demonstrate_data_processing():
    """Demonstrate neural data processing on various tasks."""

    print("\n" + "=" * 70)
    print("NEURAL DATA PROCESSING ALGORITHM DEMONSTRATION")
    print("=" * 70)

    # --- 1. XOR (non-linear classification) ---
    print("\n1. XOR PROBLEM (Non-linear Classification)")
    print("-" * 70)

    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])

    layers = [
        LayerConfig(input_size=2, output_size=4, activation=ActivationFunction.SIGMOID),
        LayerConfig(input_size=4, output_size=1, activation=ActivationFunction.SIGMOID),
    ]

    processor1 = NeuralDataProcessor(
        layer_configs=layers,
        learning_rate=0.5,
        loss_function=LossFunction.MEAN_SQUARED_ERROR,
        optimization_method="adam",
    )

    processor1.train(X_xor, y_xor, epochs=1000, verbose=False)

    print("XOR Truth Table:")
    print("Input → Predicted (Target)")
    for i in range(len(X_xor)):
        pred = processor1.predict(X_xor[i])
        print(f"{X_xor[i]} → {pred[0]:.4f} ({y_xor[i][0]})")

    # --- 2. Sine-wave regression ---
    print("\n\n2. REGRESSION - Sine Wave Approximation")
    print("-" * 70)

    X_sine = np.linspace(0, 2 * np.pi, 50).reshape(-1, 1)
    y_sine = np.sin(X_sine)

    layers2 = [
        LayerConfig(input_size=1,  output_size=32, activation=ActivationFunction.TANH),
        LayerConfig(input_size=32, output_size=32, activation=ActivationFunction.TANH),
        LayerConfig(input_size=32, output_size=16, activation=ActivationFunction.TANH),
        LayerConfig(input_size=16, output_size=1,  activation=ActivationFunction.LINEAR),
    ]

    processor2 = NeuralDataProcessor(
        layer_configs=layers2,
        learning_rate=0.01,
        loss_function=LossFunction.MEAN_SQUARED_ERROR,
        optimization_method="adam",
    )

    processor2.train(X_sine, y_sine, epochs=500, batch_size=10, verbose=True)

    test_points = np.array([[0], [np.pi / 2], [np.pi], [3 * np.pi / 2]])
    print("\nTest Predictions:")
    print("Input → Predicted (Actual)")
    for point in test_points:
        pred   = processor2.predict(point)
        actual = np.sin(point)
        print(f"{point[0]:.4f} → {pred[0]:.4f} ({actual[0]:.4f})")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


# =============================================================
# Demo: two-phase training + benchmark  (replaces the old
#        demonstrate_pytorch_integration)
# =============================================================
def demonstrate_two_phase_training():
    """Demonstrate two-phase training and vectorised benchmark."""

    print("\n" + "=" * 70)
    print("TWO-PHASE TRAINING & VECTORISED BENCHMARK DEMONSTRATION")
    print("=" * 70)

    # --- 1. XOR with two-phase training ---
    print("\n1. TWO-PHASE TRAINING - XOR Problem")
    print("-" * 70)

    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])

    layers = [
        LayerConfig(input_size=2, output_size=4, activation=ActivationFunction.SIGMOID),
        LayerConfig(input_size=4, output_size=1, activation=ActivationFunction.SIGMOID),
    ]

    processor = NeuralDataProcessor(
        layer_configs=layers,
        learning_rate=0.5,
        loss_function=LossFunction.MEAN_SQUARED_ERROR,
        optimization_method="adam",
    )

    processor.train_two_phase(
        X_xor, y_xor,
        epochs=100,
        phase1_epochs=70,
        verbose=True,
    )

    print("\nXOR Predictions:")
    for i in range(len(X_xor)):
        pred = processor.predict(X_xor[i])
        print(f"{X_xor[i]} → {pred[0]:.4f} (target: {y_xor[i][0]})")

    # --- 2. Benchmark: loop vs vectorised ---
    print("\n2. PERFORMANCE BENCHMARK")
    print("-" * 70)

    X_bench = np.random.randn(1000, 2)
    results = processor.benchmark(X_bench, iterations=50)

    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    print("\n" + "=" * 70)


# =============================================================
if __name__ == "__main__":
    demonstrate_data_processing()
    demonstrate_two_phase_training()