import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

# A simple dictionary-based container to hold the gradients during backpropagation.
class BackpropGrads(dict):
    pass

class NeuralNet:
    """
    Encapsulates the neural network logic for training and prediction.
    Fully supports multiple hidden layers by accepting a list of hidden sizes.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: Union[int, List[int]] = 32,  # single or multiple hidden layers
                 output_size: int = 4,
                 learning_rate: float = 0.01,
                 epochs: int = 150,
                 l2_lambda: float = 0.001,
                 batch_size: Optional[int] = None,
                 descent_type: str = 'batch',
                 momentum: float = 0.9,
                 lr_decay: float = 0.0,
                 use_batchnorm: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.5,
                 use_grad_clip: bool = False,
                 grad_clip_norm: float = 5.0,
                 early_stopping: bool = False,
                 patience: int = 10,
                 validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
        # Ensure hidden_size is a list
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        self.hidden_sizes: List[int] = hidden_size

        self.input_size: int = input_size
        self.output_size: int = output_size
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.l2_lambda: float = l2_lambda
        self.batch_size: Optional[int] = batch_size
        self.descent_type: str = descent_type
        self.momentum: float = momentum
        self.lr_decay: float = lr_decay

        self.use_batchnorm: bool = use_batchnorm
        self.use_dropout: bool = use_dropout
        self.dropout_rate: float = dropout_rate
        self.use_grad_clip: bool = use_grad_clip
        self.grad_clip_norm: float = grad_clip_norm

        self.early_stopping: bool = early_stopping
        self.patience: int = patience
        self.validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = validation_data

        self.best_weights: Dict[str, Any] = {}
        self.best_val_loss: float = np.inf
        self.no_improvement_count: int = 0

        self.num_hidden_layers: int = len(self.hidden_sizes)
        self.num_layers: int = self.num_hidden_layers + 1  # includes output layer

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        self.velocity_w: List[np.ndarray] = []
        self.velocity_b: List[np.ndarray] = []

        self.gamma: List[np.ndarray] = []
        self.beta: List[np.ndarray] = []
        self.running_mean: List[np.ndarray] = []
        self.running_var: List[np.ndarray] = []
        self.velocity_gamma: List[np.ndarray] = []
        self.velocity_beta: List[np.ndarray] = []
        self.bn_momentum: float = 0.9

        self._init_parameters()
        self._init_velocity()

    def _init_parameters(self) -> None:
        in_dim: int = self.input_size
        for out_dim in self.hidden_sizes:
            w: np.ndarray = (np.random.randn(in_dim, out_dim)
                 .astype(np.float32) * np.sqrt(2.0 / in_dim))
            b: np.ndarray = np.zeros((1, out_dim), dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)
            if self.use_batchnorm:
                self.gamma.append(np.ones((1, out_dim), dtype=np.float32))
                self.beta.append(np.zeros((1, out_dim), dtype=np.float32))
                self.running_mean.append(np.zeros((1, out_dim), dtype=np.float32))
                self.running_var.append(np.ones((1, out_dim), dtype=np.float32))
            in_dim = out_dim
        # Output layer initialization
        w_out: np.ndarray = (np.random.randn(in_dim, self.output_size)
                 .astype(np.float32) * np.sqrt(2.0 / in_dim))
        b_out: np.ndarray = np.zeros((1, self.output_size), dtype=np.float32)
        self.weights.append(w_out)
        self.biases.append(b_out)

    def _init_velocity(self) -> None:
        for w in self.weights:
            self.velocity_w.append(np.zeros_like(w))
        for b in self.biases:
            self.velocity_b.append(np.zeros_like(b))
        if self.use_batchnorm:
            for g in self.gamma:
                self.velocity_gamma.append(np.zeros_like(g))
            for b_ in self.beta:
                self.velocity_beta.append(np.zeros_like(b_))

    def reset(self) -> None:
        """Reset all parameters, velocities, and batch norm statistics."""
        self.weights.clear()
        self.biases.clear()
        self.velocity_w.clear()
        self.velocity_b.clear()
        self.gamma.clear()
        self.beta.clear()
        self.running_mean.clear()
        self.running_var.clear()
        self.velocity_gamma.clear()
        self.velocity_beta.clear()

        self._init_parameters()
        self._init_velocity()

        self.best_weights = {}
        self.best_val_loss = np.inf
        self.no_improvement_count = 0

    def fit(self, X: np.ndarray, y_integer: np.ndarray) -> None:
        """Train the neural network with the provided data and labels."""
        for epoch in range(self.epochs):
            current_lr: float = self.learning_rate * ((1.0 - self.lr_decay) ** epoch)
            indices: np.ndarray = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled: np.ndarray = X[indices]
            y_shuffled: np.ndarray = y_integer[indices]
            epoch_loss: float = 0.0
            num_batches: int = 0

            if self.descent_type == 'batch' or self.batch_size is None:
                _, cache, probs = self._forward_pass(X_shuffled, training=True)
                total_loss, _ = self._compute_loss(probs, y_shuffled)
                gradients: Dict[str, List[Optional[np.ndarray]]] = self._backward_pass(X_shuffled, y_shuffled, cache, probs)
                self._update_parameters(gradients, current_lr)
                epoch_loss = total_loss
                num_batches = 1
            else:
                for start_idx in range(0, X_shuffled.shape[0], self.batch_size):
                    end_idx: int = start_idx + self.batch_size
                    X_batch: np.ndarray = X_shuffled[start_idx:end_idx]
                    y_batch: np.ndarray = y_shuffled[start_idx:end_idx]
                    _, cache, probs = self._forward_pass(X_batch, training=True)
                    total_loss, _ = self._compute_loss(probs, y_batch)
                    gradients = self._backward_pass(X_batch, y_batch, cache, probs)
                    self._update_parameters(gradients, current_lr)
                    epoch_loss += total_loss
                    num_batches += 1
                epoch_loss /= num_batches

            accuracy: float = self._compute_accuracy(X, y_integer)
            print(f"Epoch {epoch + 1}/{self.epochs}\tLoss: {epoch_loss:.4f}\tAccuracy: {accuracy * 100:.2f}%")

            if self.early_stopping and self.validation_data is not None:
                X_val, y_val = self.validation_data
                val_probs: np.ndarray = self.predict_proba(X_val)
                val_loss, _ = self._compute_loss(val_probs, y_val)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.no_improvement_count = 0
                    self._save_current_weights()
                else:
                    self.no_improvement_count += 1
                    if self.no_improvement_count >= self.patience:
                        print(f"Early stopping at epoch {epoch + 1}.")
                        self._restore_best_weights()
                        break

        final_probs: np.ndarray = self.predict_proba(X)
        final_loss, _ = self._compute_loss(final_probs, y_integer)
        final_accuracy: float = self._compute_accuracy(X, y_integer)
        print("-" * 40)
        print(f"Final Training Loss: {final_loss:.4f}")
        print(f"Final Training Accuracy: {final_accuracy * 100:.2f}%")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the probability distribution over classes for input X."""
        _, _, probs = self._forward_pass(X, training=False)
        return probs

    def _forward_pass(self, X: np.ndarray, training: bool = True) -> Tuple[List[np.ndarray], Dict[str, Any], np.ndarray]:
        activations: List[np.ndarray] = []
        cache: Dict[str, Any] = {}
        layer_input: np.ndarray = X

        for i in range(self.num_hidden_layers):
            z: np.ndarray = np.dot(layer_input, self.weights[i]) + self.biases[i]
            if self.use_batchnorm:
                if training:
                    mu: np.ndarray = np.mean(z, axis=0, keepdims=True)
                    var: np.ndarray = np.var(z, axis=0, keepdims=True)
                    z_hat: np.ndarray = (z - mu) / np.sqrt(var + 1e-5)
                    z_bn: np.ndarray = self.gamma[i] * z_hat + self.beta[i]
                    self.running_mean[i] = self.bn_momentum * self.running_mean[i] + (1 - self.bn_momentum) * mu
                    self.running_var[i] = self.bn_momentum * self.running_var[i] + (1 - self.bn_momentum) * var
                else:
                    mu = self.running_mean[i]
                    var = self.running_var[i]
                    z_hat = (z - mu) / np.sqrt(var + 1e-5)
                    z_bn = self.gamma[i] * z_hat + self.beta[i]
                a_before_relu: np.ndarray = z_bn
            else:
                a_before_relu = z
                mu = np.empty((0,), dtype=np.float32)
                var = np.empty((0,), dtype=np.float32)
                z_hat = np.empty((0,), dtype=np.float32)

            a: np.ndarray = np.maximum(0, a_before_relu)
            if self.use_dropout and training:
                dropout_mask: np.ndarray = (np.random.rand(*a.shape) > self.dropout_rate).astype(a.dtype)
                a = a * dropout_mask
            else:
                dropout_mask = np.ones_like(a)
            activations.append(a)
            cache[f'layer_{i}'] = {
                'z': z,
                'z_hat': z_hat,
                'mu': mu,
                'var': var,
                'a_before_relu': a_before_relu,
                'activation': a,
                'dropout_mask': dropout_mask
            }
            layer_input = a

        z_out: np.ndarray = np.dot(layer_input, self.weights[-1]) + self.biases[-1]
        shifted_logits: np.ndarray = z_out - np.max(z_out, axis=1, keepdims=True)
        exp_scores: np.ndarray = np.exp(shifted_logits)
        probs: np.ndarray = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        activations.append(z_out)
        cache['output'] = {
            'z_out': z_out,
            'probs': probs
        }
        return activations, cache, probs

    def _compute_loss(self, probs: np.ndarray, y_integer: np.ndarray) -> Tuple[float, float]:
        correct_logprobs: np.ndarray = -np.log(probs[np.arange(probs.shape[0]), y_integer] + 1e-8)
        data_loss: float = float(np.mean(correct_logprobs))
        reg_loss: float = 0.0
        for w in self.weights:
            reg_loss += 0.5 * self.l2_lambda * np.sum(w * w)
        total_loss: float = data_loss + reg_loss
        return total_loss, data_loss

    def _backward_pass(self, X: np.ndarray, y_integer: np.ndarray, cache: Dict[str, Any], probs: np.ndarray) -> Dict[str, List[Optional[np.ndarray]]]:
        dW: List[Optional[np.ndarray]] = [np.zeros_like(w) for w in self.weights]
        db: List[Optional[np.ndarray]] = [np.zeros_like(b) for b in self.biases]
        dgamma: List[Optional[np.ndarray]] = [None] * self.num_hidden_layers
        dbeta: List[Optional[np.ndarray]] = [None] * self.num_hidden_layers

        N: int = X.shape[0]
        d_out: np.ndarray = probs.copy()
        d_out[np.arange(N), y_integer] -= 1
        d_out /= N

        if self.num_hidden_layers > 0:
            last_hidden_activation: np.ndarray = cache[f'layer_{self.num_hidden_layers - 1}']['activation']
        else:
            last_hidden_activation = X

        dW[-1] = np.dot(last_hidden_activation.T, d_out) + self.l2_lambda * self.weights[-1]
        db[-1] = np.sum(d_out, axis=0, keepdims=True)
        d_hidden: np.ndarray = np.dot(d_out, self.weights[-1].T)

        for i in reversed(range(self.num_hidden_layers)):
            cache_i: Dict[str, Any] = cache[f'layer_{i}']
            dropout_mask: np.ndarray = cache_i['dropout_mask']
            z: np.ndarray = cache_i['z']
            a_before_relu: np.ndarray = cache_i['a_before_relu']

            d_hidden *= dropout_mask
            relu_mask: np.ndarray = (a_before_relu > 0).astype(d_hidden.dtype)
            d_hidden *= relu_mask

            if self.use_batchnorm:
                z_hat: np.ndarray = cache_i['z_hat']
                mu: np.ndarray = cache_i['mu']
                var: np.ndarray = cache_i['var']
                eps: float = 1e-5

                d_z_hat: np.ndarray = d_hidden * self.gamma[i]
                d_var: np.ndarray = np.sum(d_z_hat * (z - mu) * -0.5 * (var + eps) ** -1.5,
                               axis=0, keepdims=True)
                d_mu: np.ndarray = (np.sum(d_z_hat * (-1.0 / np.sqrt(var + eps)), axis=0, keepdims=True) +
                        d_var * np.mean(-2.0 * (z - mu), axis=0, keepdims=True))
                dz: np.ndarray = (d_z_hat / np.sqrt(var + eps)) + (d_var * 2.0 * (z - mu) / N) + (d_mu / N)
                dgamma[i] = np.sum(d_hidden * z_hat, axis=0, keepdims=True)
                dbeta[i] = np.sum(d_hidden, axis=0, keepdims=True)
                d_hidden = dz
            else:
                dgamma[i] = None
                dbeta[i] = None

            if i == 0:
                layer_input: np.ndarray = X
            else:
                layer_input = cache[f'layer_{i-1}']['activation']

            dW[i] = np.dot(layer_input.T, d_hidden) + self.l2_lambda * self.weights[i]
            db[i] = np.sum(d_hidden, axis=0, keepdims=True)
            d_hidden = np.dot(d_hidden, self.weights[i].T)

        return {"dW": dW, "db": db, "dgamma": dgamma, "dbeta": dbeta}

    def _update_parameters(self, grads: Dict[str, List[Optional[np.ndarray]]], current_lr: float) -> None:
        for i, dw_ in enumerate(grads['dW']):
            if dw_ is not None:
                self.weights[i] -= current_lr * dw_
        for i, db_ in enumerate(grads['db']):
            if db_ is not None:
                self.biases[i] -= current_lr * db_
        if self.use_batchnorm:
            for i, dgamma_ in enumerate(grads['dgamma']):
                if dgamma_ is not None:
                    self.gamma[i] -= current_lr * dgamma_
            for i, dbeta_ in enumerate(grads['dbeta']):
                if dbeta_ is not None:
                    self.beta[i] -= current_lr * dbeta_

    def _compute_accuracy(self, X: np.ndarray, y_integer: np.ndarray) -> float:
        probs: np.ndarray = self.predict_proba(X)
        predictions: np.ndarray = np.argmax(probs, axis=1)
        return float(np.mean(predictions == y_integer))

    def _save_current_weights(self) -> None:
        self.best_weights = {
            'weights': [w.copy() for w in self.weights],
            'biases': [b.copy() for b in self.biases],
            'velocity_w': [v.copy() for v in self.velocity_w],
            'velocity_b': [v.copy() for v in self.velocity_b]
        }
        if self.use_batchnorm:
            self.best_weights['gamma'] = [g.copy() for g in self.gamma]
            self.best_weights['beta'] = [b.copy() for b in self.beta]
            self.best_weights['velocity_gamma'] = [vg.copy() for vg in self.velocity_gamma]
            self.best_weights['velocity_beta'] = [vb.copy() for vb in self.velocity_beta]

    def _restore_best_weights(self) -> None:
        if not self.best_weights:
            return
        for i in range(len(self.weights)):
            self.weights[i] = self.best_weights['weights'][i]
            self.biases[i] = self.best_weights['biases'][i]
            self.velocity_w[i] = self.best_weights['velocity_w'][i]
            self.velocity_b[i] = self.best_weights['velocity_b'][i]
        if self.use_batchnorm and 'gamma' in self.best_weights:
            for i in range(len(self.gamma)):
                self.gamma[i] = self.best_weights['gamma'][i]
                self.beta[i] = self.best_weights['beta'][i]
                self.velocity_gamma[i] = self.best_weights['velocity_gamma'][i]
                self.velocity_beta[i] = self.best_weights['velocity_beta'][i]

class Classifier:
    """
    A custom neural network classifier for Pacman's movement decisions.
    Wraps around the NeuralNet logic.
    """
    def __init__(self) -> None:
        self.nn: Optional[NeuralNet] = None
        self.hidden_layers: List[int] = [32]  # Default single hidden layer with 32 neurons
        self.output_size: int = 4
        self.learning_rate: float = 0.01
        self.epochs: int = 150
        self.l2_lambda: float = 0.001

        self.batch_size: int = 32
        self.descent_type: str = 'mini-batch'
        self.momentum: float = 0.9
        self.lr_decay: float = 0.01

        self.use_batchnorm: bool = True
        self.use_dropout: bool = True
        self.dropout_rate: float = 0.5
        self.use_grad_clip: bool = True
        self.grad_clip_norm: float = 5.0

        self.early_stopping: bool = False
        self.patience: int = 10

        self.enable_grid_search: bool = True

        print(f"Classifier initialised with hidden_layers={self.hidden_layers}, learning_rate={self.learning_rate}")

    def reset(self) -> None:
        """Reset the neural network if it has already been initialised."""
        print("\nResetting model weights, biases, and BN stats...")
        if self.nn is not None:
            self.nn.reset()

    def fit(self, data: Any, target: Any) -> None:
        """
        Train the neural network using the provided data and target labels.
        Splits the data into train (80%), validation (10%), and test (10%).
        """
        X: np.ndarray = np.array(data, dtype=np.float32)
        y_integer: np.ndarray = np.array(target, dtype=np.int32)
        total_samples: int = len(X)
        print(f"\nStarting training with {total_samples} samples...")

        train_end: int = int(total_samples * 0.8)
        val_end: int = int(total_samples * 0.9)
        X_train: np.ndarray = X[:train_end]
        X_val: np.ndarray = X[train_end:val_end]
        X_test: np.ndarray = X[val_end:]
        y_train: np.ndarray = y_integer[:train_end]
        y_val: np.ndarray = y_integer[train_end:val_end]
        y_test: np.ndarray = y_integer[val_end:]

        print("Dataset Metrics:")
        print(f"  - Total samples: {total_samples}")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Validation samples: {len(X_val)}")
        print(f"  - Test samples: {len(X_test)}")
        
        if self.enable_grid_search and X_val.shape[0] > 0:
            best_params: Dict[str, Any] = self._grid_search_hyperparams(X_train, y_train, X_val, y_val)
            self._init_nn(best_params, X_train.shape[1], (X_val, y_val))
        else:
            self._init_nn(None, X.shape[1], (X_val, y_val))
    
        print("\nTraining progress (with early stopping if enabled):")
        print("Epoch\tLoss\t\tTraining Accuracy")
        print("-" * 40)
        if self.nn is not None:
            self.nn.fit(X_train, y_train)

        print("\nEvaluating final model on test set...")
        if self.nn is not None and X_test.shape[0] > 0:
            test_probs: np.ndarray = self.nn.predict_proba(X_test)
            test_pred: np.ndarray = np.argmax(test_probs, axis=1)
            test_acc: float = float(np.mean(test_pred == y_test))
            print(f"Test Accuracy: {test_acc * 100:.2f}%")

    def predict(self, features: Any, legal: List[str]) -> int:
        """
        Predict the best action given the feature vector and legal moves.
        Returns the chosen action as an integer (0, 1, 2, or 3).
        """
        if self.nn is None:
            return int(np.random.randint(4))
        X: np.ndarray = np.array(features, dtype=np.float32).reshape(1, -1)
        probs: np.ndarray = self.nn.predict_proba(X)
        legal_actions: List[int] = []
        for move in legal:
            if move == 'North':
                legal_actions.append(0)
            elif move == 'East':
                legal_actions.append(1)
            elif move == 'South':
                legal_actions.append(2)
            elif move == 'West':
                legal_actions.append(3)
        if not legal_actions:
            return int(np.random.randint(4))
        legal_probs: np.ndarray = probs[0][legal_actions]
        best_index: int = int(np.argmax(legal_probs))
        return legal_actions[best_index]

    def _init_nn(self, params: Optional[Dict[str, Any]], input_size: int, val_data: Tuple[np.ndarray, np.ndarray]) -> None:
        if params is None:
            hidden_layers: List[int] = self.hidden_layers
            dropout_rate: float = self.dropout_rate
        else:
            hidden_layers = params['hidden_layers']
            dropout_rate = params['dropout_rate']

        print(f"Initialising NeuralNet with hidden_layers={hidden_layers}, dropout_rate={dropout_rate}")
        self.nn = NeuralNet(
            input_size=input_size,
            hidden_size=hidden_layers,
            output_size=self.output_size,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            l2_lambda=self.l2_lambda,
            batch_size=self.batch_size,
            descent_type=self.descent_type,
            momentum=self.momentum,
            lr_decay=self.lr_decay,
            use_batchnorm=self.use_batchnorm,
            use_dropout=self.use_dropout,
            dropout_rate=dropout_rate,
            use_grad_clip=self.use_grad_clip,
            grad_clip_norm=self.grad_clip_norm,
            early_stopping=self.early_stopping,
            patience=self.patience,
            validation_data=val_data
        )

    def _grid_search_hyperparams(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        layer_configs: List[List[int]] = [
            [4], [6], [8], [12], [16], [20], [24], [28], [32],
            [4, 4], [4, 6], [8, 4], [4, 8], [6, 8],
            [12, 4], [8, 8], [6, 12], [8, 12], [12, 8],
            [16, 6], [8, 16], [16, 8], [6, 24],
            [2, 2, 4], [2, 4, 4], [2, 4, 6], [2, 6, 4],
            [4, 4, 4], [4, 4, 6], [4, 6, 8], [4, 8, 8],
            [2, 8, 12], [6, 8, 12]
        ]
        dropout_rates: List[float] = [0.1, 0.3, 0.5, 0.7]
        best_val_acc: float = -1.0
        best_params: Dict[str, Any] = {'hidden_layers': self.hidden_layers, 'dropout_rate': self.dropout_rate}
        for layers in layer_configs:
            for dr in dropout_rates:
                temp_nn: NeuralNet = NeuralNet(
                    input_size=X_train.shape[1],
                    hidden_size=layers,
                    output_size=self.output_size,
                    learning_rate=self.learning_rate,
                    epochs=20,  # shorter training for quicker search
                    l2_lambda=self.l2_lambda,
                    batch_size=self.batch_size,
                    descent_type=self.descent_type,
                    momentum=self.momentum,
                    lr_decay=self.lr_decay,
                    use_batchnorm=self.use_batchnorm,
                    use_dropout=self.use_dropout,
                    dropout_rate=dr,
                    use_grad_clip=self.use_grad_clip,
                    grad_clip_norm=self.grad_clip_norm,
                    early_stopping=False
                )
                temp_nn.fit(X_train, y_train)
                probs: np.ndarray = temp_nn.predict_proba(X_val)
                predictions: np.ndarray = np.argmax(probs, axis=1)
                val_acc: float = float(np.mean(predictions == y_val))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = {'hidden_layers': layers, 'dropout_rate': dr}
        print(f"Best hyperparameters from grid search: {best_params}, val_acc={best_val_acc:.4f}")
        return best_params
