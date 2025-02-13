import numpy as np
from typing import List, Tuple, Dict, Optional, Union

class NeuralNet:
    """
    Encapsulates the neural network logic for training and prediction.
    Includes:
      - Dropout
      - Batch Normalisation
      - Momentum
      - Learning Rate Decay
      - Gradient Clipping
      - Early Stopping (when triggered by validation performance)

    Now supports specifying the hidden layer(s) as a list. For this coursework,
    we still only expect one hidden layer, so the list should be of length 1.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, List[int]] = 32,  # Can be int or list (of length 1)
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
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> None:
        """
        Parameters:
            input_size       (int): Number of input features.
            hidden_size (int or list): If int, we interpret it as a single hidden layer
                                       with that many neurons. If a list of length 1,
                                       each entry is the number of neurons in that layer.
            output_size      (int): Number of output classes.
            learning_rate    (float): Initial learning rate.
            epochs           (int): Maximum training epochs.
            l2_lambda        (float): L2 regularisation strength.
            batch_size       (int or None): Mini-batch size; None => full batch.
            descent_type     (str): 'batch' or 'mini-batch'.
            momentum         (float): Momentum coefficient.
            lr_decay         (float): Exponential decay for learning rate.
            use_batchnorm    (bool): Whether to use batch norm on the hidden layer.
            use_dropout      (bool): Whether to apply dropout on the hidden layer.
            dropout_rate     (float): Dropout probability.
            use_grad_clip    (bool): Enable gradient clipping if True.
            grad_clip_norm   (float): Max norm for gradient clipping.
            early_stopping   (bool): Whether to enable early stopping.
            patience         (int): Number of consecutive epochs allowed without improvement.
            validation_data  (tuple): (X_val, y_val) used for early stopping checks.
        """

        # If hidden_size is an integer, convert it to a single-element list.
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]

        # For this coursework, we still only expect exactly one hidden layer.
        if len(hidden_size) != 1:
            raise ValueError("For this coursework, hidden_size must be a single-element list or an integer.")

        # Internally, we'll keep the single hidden layer size as self.hidden_size.
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size[0]  # single hidden layer
        self.output_size: int = output_size

        # Training hyperparameters
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.l2_lambda: float = l2_lambda
        self.batch_size: Optional[int] = batch_size
        self.descent_type: str = descent_type
        self.momentum: float = momentum
        self.lr_decay: float = lr_decay

        # Additional features
        self.use_batchnorm: bool = use_batchnorm
        self.use_dropout: bool = use_dropout
        self.dropout_rate: float = dropout_rate
        self.use_grad_clip: bool = use_grad_clip
        self.grad_clip_norm: float = grad_clip_norm

        # Early Stopping
        self.early_stopping: bool = early_stopping
        self.patience: int = patience
        self.validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = validation_data

        # For tracking improvements
        self.best_weights: Dict[str, np.ndarray] = {}
        self.best_val_loss: float = np.inf
        self.no_improvement_count: int = 0

        # Initialise parameters and velocity
        self._init_parameters()
        self._init_velocity()

        # Batch norm parameters (for the single hidden layer)
        if self.use_batchnorm:
            self.gamma: np.ndarray = np.ones((1, self.hidden_size), dtype=np.float32)
            self.beta: np.ndarray = np.zeros((1, self.hidden_size), dtype=np.float32)
            self.running_mean: np.ndarray = np.zeros((1, self.hidden_size), dtype=np.float32)
            self.running_var: np.ndarray = np.ones((1, self.hidden_size), dtype=np.float32)
            self.bn_momentum: float = 0.9

    def _init_parameters(self) -> None:
        """Initialise weights and biases for one hidden layer."""
        self.weights1: np.ndarray = (
            np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        )
        self.weights2: np.ndarray = (
            np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        )
        self.bias1: np.ndarray = np.zeros((1, self.hidden_size))
        self.bias2: np.ndarray = np.zeros((1, self.output_size))

    def _init_velocity(self) -> None:
        """Initialise velocities for momentum-based updates."""
        self.velocity_w1: np.ndarray = np.zeros_like(self.weights1)
        self.velocity_b1: np.ndarray = np.zeros_like(self.bias1)
        self.velocity_w2: np.ndarray = np.zeros_like(self.weights2)
        self.velocity_b2: np.ndarray = np.zeros_like(self.bias2)
        if self.use_batchnorm:
            self.velocity_gamma: np.ndarray = np.zeros((1, self.hidden_size))
            self.velocity_beta: np.ndarray = np.zeros((1, self.hidden_size))

    def reset(self) -> None:
        """
        Reset all parameters, velocities, and batch norm running statistics.
        Also resets the counters for early stopping.
        """
        self._init_parameters()
        self._init_velocity()
        if self.use_batchnorm:
            self.running_mean = np.zeros((1, self.hidden_size), dtype=np.float32)
            self.running_var = np.ones((1, self.hidden_size), dtype=np.float32)
        self.best_weights = {}
        self.best_val_loss = np.inf
        self.no_improvement_count = 0

    def fit(self, X: np.ndarray, y_integer: np.ndarray) -> None:
        """
        Train the neural network using the provided data X and integer labels y_integer.
        Now prints loss and accuracy for each epoch and overall metrics at the end.
    
        Parameters:
            X         (np.ndarray): Training data of shape (N, input_size).
            y_integer (np.ndarray): Integer class labels of shape (N,).
        """
        for epoch in range(self.epochs):
            # Learning rate decay
            current_lr: float = self.learning_rate * (1.0 - self.lr_decay) ** epoch
    
            # Shuffle data
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y_integer[indices]
    
            # Accumulate loss for the epoch
            epoch_loss: float = 0.0
            num_batches: int = 0
    
            # Batch or mini-batch training
            if self.descent_type == 'batch' or self.batch_size is None:
                _, cache, probs = self._forward_pass(X_shuffled, training=True)
                total_loss, _ = self._compute_loss(probs, y_shuffled)
                dW1, db1, dW2, db2, dgamma, dbeta = self._backward_pass(
                    X_shuffled, y_shuffled, cache, probs
                )
                self._update_parameters(dW1, db1, dW2, db2, dgamma, dbeta, current_lr)
                epoch_loss = total_loss
                num_batches = 1
            else:
                for start_idx in range(0, X_shuffled.shape[0], self.batch_size):
                    end_idx = start_idx + self.batch_size
                    X_batch = X_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]
    
                    _, cache, probs = self._forward_pass(X_batch, training=True)
                    total_loss, _ = self._compute_loss(probs, y_batch)
                    dW1, db1, dW2, db2, dgamma, dbeta = self._backward_pass(
                        X_batch, y_batch, cache, probs
                    )
                    self._update_parameters(dW1, db1, dW2, db2, dgamma, dbeta, current_lr)
                    epoch_loss += total_loss
                    num_batches += 1
                epoch_loss /= num_batches
    
            # Compute training accuracy on the whole training set
            accuracy = self._compute_accuracy(X, y_integer)
            print(f"Epoch {epoch + 1}/{self.epochs}\tLoss: {epoch_loss:.4f}\tAccuracy: {accuracy * 100:.2f}%")
    
            # Early Stopping check
            if self.early_stopping and self.validation_data is not None:
                X_val, y_val = self.validation_data
                val_probs = self.predict_proba(X_val)
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
    
        # After training, report overall training metrics
        final_probs = self.predict_proba(X)
        final_loss, _ = self._compute_loss(final_probs, y_integer)
        final_accuracy = self._compute_accuracy(X, y_integer)
        print("-" * 40)
        print(f"Final Training Loss: {final_loss:.4f}")
        print(f"Final Training Accuracy: {final_accuracy * 100:.2f}%")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass to obtain probability distribution over the classes.

        Parameters:
            X (np.ndarray): Data of shape (N, input_size).

        Returns:
            np.ndarray: Probability distribution over classes, shape (N, output_size).
        """
        _, _, probs = self._forward_pass(X, training=False)
        return probs

    def _forward_pass(
        self,
        X: np.ndarray,
        training: bool = True
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Forward pass through the network.

        Parameters:
            X        (np.ndarray): Input data, shape (N, input_size).
            training      (bool): If True, apply batch norm updates and dropout.

        Returns:
            Tuple:
                - hidden_activation (np.ndarray): Output from hidden layer post-ReLU (N, hidden_size)
                - cache (dict): Contains intermediate values for backprop
                - probs (np.ndarray): Softmax probabilities (N, output_size)
        """
        z1: np.ndarray = np.dot(X, self.weights1) + self.bias1

        if self.use_batchnorm:
            if training:
                mu: np.ndarray = np.mean(z1, axis=0, keepdims=True)
                var: np.ndarray = np.var(z1, axis=0, keepdims=True)
                z1_hat: np.ndarray = (z1 - mu) / np.sqrt(var + 1e-5)
                z1_bn: np.ndarray = self.gamma * z1_hat + self.beta

                self.running_mean = self.bn_momentum * self.running_mean + (1 - self.bn_momentum) * mu
                self.running_var = self.bn_momentum * self.running_var + (1 - self.bn_momentum) * var
            else:
                z1_hat = (z1 - self.running_mean) / np.sqrt(self.running_var + 1e-5)
                z1_bn = self.gamma * z1_hat + self.beta

            a1_before_relu = z1_bn
        else:
            a1_before_relu = z1

        hidden_activation: np.ndarray = np.maximum(0, a1_before_relu)

        # Dropout on hidden activation
        if self.use_dropout and training:
            dropout_mask: np.ndarray = (
                (np.random.rand(*hidden_activation.shape) > self.dropout_rate)
                .astype(hidden_activation.dtype)
            )
            hidden_activation *= dropout_mask
        else:
            dropout_mask = np.ones_like(hidden_activation)

        z2: np.ndarray = np.dot(hidden_activation, self.weights2) + self.bias2

        # Softmax
        shifted_logits: np.ndarray = z2 - np.max(z2, axis=1, keepdims=True)
        exp_scores: np.ndarray = np.exp(shifted_logits)
        probs: np.ndarray = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        cache: Dict[str, np.ndarray] = {
            'X': X,
            'z1': z1,
            'hidden_activation': hidden_activation,
            'dropout_mask': dropout_mask,
            'z2': z2,
            'probs': probs
        }

        if self.use_batchnorm:
            cache['z1_hat'] = z1_hat
            cache['mu'] = mu if training else self.running_mean
            cache['var'] = var if training else self.running_var

        return hidden_activation, cache, probs

    def _compute_loss(
        self,
        probs: np.ndarray,
        y_integer: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute cross-entropy loss with an L2 regularisation term.

        Parameters:
            probs     (np.ndarray): Probability distributions (N, output_size).
            y_integer (np.ndarray): Ground-truth integer labels (N,).

        Returns:
            Tuple of:
             - total_loss (float): cross-entropy + regularisation
             - data_loss  (float): just cross-entropy
        """
        correct_logprobs: np.ndarray = -np.log(probs[np.arange(probs.shape[0]), y_integer] + 1e-8)
        data_loss: float = np.mean(correct_logprobs)
        reg_loss: float = 0.5 * self.l2_lambda * (np.sum(self.weights1**2) + np.sum(self.weights2**2))
        total_loss: float = data_loss + reg_loss
        return total_loss, data_loss

    def _backward_pass(
        self,
        X: np.ndarray,
        y_integer: np.ndarray,
        cache: Dict[str, np.ndarray],
        probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute gradients for weights and biases through backpropagation.

        Parameters:
            X         (np.ndarray): Input data of shape (N, input_size).
            y_integer (np.ndarray): Ground-truth integer labels (N,).
            cache     (dict): Intermediate values from the forward pass.
            probs     (np.ndarray): Softmax probabilities.

        Returns:
            A 6-tuple of gradients:
             (dW1, db1, dW2, db2, dgamma, dbeta)
            where dgamma and dbeta may be None if batchnorm is disabled.
        """
        N: int = X.shape[0]
        d_output: np.ndarray = probs.copy()
        d_output[np.arange(N), y_integer] -= 1
        d_output /= N

        hidden_activation: np.ndarray = cache['hidden_activation']
        dW2: np.ndarray = np.dot(hidden_activation.T, d_output) + self.l2_lambda * self.weights2
        db2: np.ndarray = np.sum(d_output, axis=0, keepdims=True)

        d_hidden: np.ndarray = np.dot(d_output, self.weights2.T)
        d_hidden *= cache['dropout_mask']  # Dropout

        # ReLU
        da1_before_relu: np.ndarray = (cache['z1'] > 0).astype(d_hidden.dtype) * d_hidden

        dgamma: Optional[np.ndarray] = None
        dbeta: Optional[np.ndarray] = None
        if self.use_batchnorm:
            z1_hat: np.ndarray = cache['z1_hat']
            mu: np.ndarray = cache['mu']
            var: np.ndarray = cache['var']
            eps: float = 1e-5

            d_z1_hat: np.ndarray = da1_before_relu * self.gamma
            d_var: np.ndarray = np.sum(
                d_z1_hat * (cache['z1'] - mu) * -0.5 * (var + eps) ** -1.5,
                axis=0,
                keepdims=True
            )
            d_mu: np.ndarray = (
                np.sum(d_z1_hat * -1.0 / np.sqrt(var + eps), axis=0, keepdims=True)
                + d_var * np.mean(-2.0 * (cache['z1'] - mu), axis=0, keepdims=True)
            )

            dz1: np.ndarray = (
                (d_z1_hat / np.sqrt(var + eps))
                + (d_var * 2.0 * (cache['z1'] - mu) / N)
                + (d_mu / N)
            )

            dgamma = np.sum(da1_before_relu * z1_hat, axis=0, keepdims=True)
            dbeta = np.sum(da1_before_relu, axis=0, keepdims=True)

            da1_before_relu = dz1

        dW1: np.ndarray = np.dot(X.T, da1_before_relu) + self.l2_lambda * self.weights1
        db1: np.ndarray = np.sum(da1_before_relu, axis=0, keepdims=True)

        return dW1, db1, dW2, db2, dgamma, dbeta

    def _update_parameters(
        self,
        dW1: np.ndarray,
        db1: np.ndarray,
        dW2: np.ndarray,
        db2: np.ndarray,
        dgamma: Optional[np.ndarray],
        dbeta: Optional[np.ndarray],
        current_lr: float
    ) -> None:
        """
        Update parameters using gradients. Includes optional gradient clipping and momentum.

        Parameters:
            dW1, db1, dW2, db2: Gradients for first and second layer weights and biases.
            dgamma, dbeta     : Gradients for batchnorm parameters (if applicable).
            current_lr        : Learning rate for this epoch.
        """

        # Gradient clipping
        if self.use_grad_clip:
            grad_list = [dW1, db1, dW2, db2]
            if self.use_batchnorm and dgamma is not None and dbeta is not None:
                grad_list += [dgamma, dbeta]
            total_norm: float = np.sqrt(sum(np.sum(g * g) for g in grad_list))
            if total_norm > self.grad_clip_norm:
                scale: float = self.grad_clip_norm / (total_norm + 1e-8)
                for g in grad_list:
                    g *= scale

        # Momentum updates
        self.velocity_w1 = self.momentum * self.velocity_w1 - current_lr * dW1
        self.weights1 += self.velocity_w1

        self.velocity_b1 = self.momentum * self.velocity_b1 - current_lr * db1
        self.bias1 += self.velocity_b1

        self.velocity_w2 = self.momentum * self.velocity_w2 - current_lr * dW2
        self.weights2 += self.velocity_w2

        self.velocity_b2 = self.momentum * self.velocity_b2 - current_lr * db2
        self.bias2 += self.velocity_b2

        if self.use_batchnorm and dgamma is not None and dbeta is not None:
            self.velocity_gamma = self.momentum * self.velocity_gamma - current_lr * dgamma
            self.gamma += self.velocity_gamma

            self.velocity_beta = self.momentum * self.velocity_beta - current_lr * dbeta
            self.beta += self.velocity_beta

    def _compute_accuracy(self, X: np.ndarray, y_integer: np.ndarray) -> float:
        """
        Compute classification accuracy of the model on data X with labels y_integer.

        Parameters:
            X         (np.ndarray): Data, shape (N, input_size).
            y_integer (np.ndarray): Ground-truth labels, shape (N,).

        Returns:
            float: Proportion of correct predictions.
        """
        probs = self.predict_proba(X)
        predictions = np.argmax(probs, axis=1)
        return np.mean(predictions == y_integer)

    # ----- Early Stopping helpers -----

    def _save_current_weights(self) -> None:
        """
        Save current network parameters to self.best_weights (as a backup).
        """
        self.best_weights = {
            'weights1': self.weights1.copy(),
            'bias1': self.bias1.copy(),
            'weights2': self.weights2.copy(),
            'bias2': self.bias2.copy(),
            'velocity_w1': self.velocity_w1.copy(),
            'velocity_b1': self.velocity_b1.copy(),
            'velocity_w2': self.velocity_w2.copy(),
            'velocity_b2': self.velocity_b2.copy()
        }
        if self.use_batchnorm:
            self.best_weights['gamma'] = self.gamma.copy()
            self.best_weights['beta'] = self.beta.copy()
            self.best_weights['velocity_gamma'] = self.velocity_gamma.copy()
            self.best_weights['velocity_beta'] = self.velocity_beta.copy()

    def _restore_best_weights(self) -> None:
        """
        Restore network parameters from self.best_weights.
        """
        if not self.best_weights:
            return
        self.weights1 = self.best_weights['weights1']
        self.bias1 = self.best_weights['bias1']
        self.weights2 = self.best_weights['weights2']
        self.bias2 = self.best_weights['bias2']
        self.velocity_w1 = self.best_weights['velocity_w1']
        self.velocity_b1 = self.best_weights['velocity_b1']
        self.velocity_w2 = self.best_weights['velocity_w2']
        self.velocity_b2 = self.best_weights['velocity_b2']

        if self.use_batchnorm:
            self.gamma = self.best_weights['gamma']
            self.beta = self.best_weights['beta']
            self.velocity_gamma = self.best_weights['velocity_gamma']
            self.velocity_beta = self.best_weights['velocity_beta']


class Classifier:
    """
    A custom neural network classifier for Pacman's movement decisions.
    Acts as a wrapper around the NeuralNet logic.
    """

    def __init__(self) -> None:
        """
        Initialise the classifier. The neural network will be created on the first call to fit.
        """
        self.nn: Optional[NeuralNet] = None
        # Now store the hidden layer in a list form to reflect the new approach:
        self.hidden_layers: List[int] = [32]   # Single hidden layer with 32 neurons
        self.output_size: int = 4             # 4 possible actions
        self.learning_rate: float = 0.01
        self.epochs: int = 150
        self.l2_lambda: float = 0.001         # L2 regularisation strength

        self.batch_size: int = 32
        self.descent_type: str = 'mini-batch'
        self.momentum: float = 0.9
        self.lr_decay: float = 0.01

        # Additional features
        self.use_batchnorm: bool = True
        self.use_dropout: bool = True
        self.dropout_rate: float = 0.5
        self.use_grad_clip: bool = True
        self.grad_clip_norm: float = 5.0

        # Early Stopping
        self.early_stopping: bool = False
        self.patience: int = 10

        # Hyperparameter search toggle
        self.enable_grid_search: bool = True

        print(f"Classifier initialised with hidden_layers={self.hidden_layers}, learning_rate={self.learning_rate}")

    def reset(self) -> None:
        """
        Reset the classifier's neural network if it has been initialised.
        """
        print("\nResetting model weights, biases, and BN stats...")
        if self.nn is not None:
            self.nn.reset()

    def fit(self, data: list, target: list) -> None:
        """
        Train the neural network using the provided data and targets.
    
        Parameters:
            data (list): Feature vectors from the game states.
            target (list): Corresponding action labels (0-3).
        """
        X = np.array(data, dtype=np.float32)
        y_integer = np.array(target, dtype=np.int32)
    
        total_samples = len(X)
        print(f"\nStarting training with {total_samples} samples...")
    
        # Simple train/validation split for early stopping + hyperparam search
        val_proportion: float = 0.2
        split_index: int = int(total_samples * (1.0 - val_proportion))
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y_integer[:split_index], y_integer[split_index:]
    
        print(f"Dataset Metrics:")
        print(f"  - Total samples: {total_samples}")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Validation samples: {len(X_val)}")
        
        # Optional hyperparameter tuning
        if self.enable_grid_search:
            best_params: Dict[str, int | float] = self._grid_search_hyperparams(X_train, y_train, X_val, y_val)
            self._init_nn(best_params, X_train.shape[1], (X_val, y_val))
        else:
            # Initialise with default hyperparameters
            self._init_nn(None, X.shape[1], (X_val, y_val))
    
        print("\nTraining progress (with early stopping if enabled):")
        print("Epoch\tLoss\t\tTraining Accuracy")
        print("-" * 40)
        if self.nn is not None:
            self.nn.fit(X_train, y_train)

    def predict(self, features: list, legal: list) -> int:
        """
        Predict the best action given current features and legal moves.

        Parameters:
            features (list): Feature vector describing the state.
            legal    (list): A list of legal moves, e.g. ['North', 'South'].

        Returns:
            int: The chosen action as an integer (0,1,2,3).
        """
        if self.nn is None:
            # Fallback: random if not fitted
            return np.random.randint(4)

        X = np.array(features, dtype=np.float32).reshape(1, -1)
        probs = self.nn.predict_proba(X)

        # Convert legal moves to numerical actions
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
            return np.random.randint(4)

        # Highest probability among legal actions
        legal_probs = probs[0][legal_actions]
        best_index = np.argmax(legal_probs)
        return legal_actions[best_index]

    def _init_nn(
        self,
        params: Optional[Dict[str, Union[int, float]]],
        input_size: int,
        val_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Initialise the neural network with given parameters or defaults.

        Parameters:
            params     (dict or None): {'hidden_size': int, 'dropout_rate': float} or None.
            input_size (int)         : Number of features in X.
            val_data   (tuple)       : (X_val, y_val) for early stopping if used.
        """
        if params is None:
            hidden_size: int = self.hidden_layers[0]
            dropout_rate: float = self.dropout_rate
        else:
            hidden_size = int(params['hidden_size'])
            dropout_rate = params['dropout_rate']

        print(f"Initialising NeuralNet with hidden_size={hidden_size}, dropout_rate={dropout_rate}")
        self.nn = NeuralNet(
            input_size=input_size,
            hidden_size=hidden_size,  # can be int or [int], but internally it’s handled
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

    def _grid_search_hyperparams(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Union[int, float]]:
        """
        A simple grid search to find the best combination of hidden_size & dropout_rate.
        You can expand this with more parameters if desired.

        Parameters:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val   (np.ndarray): Validation features.
            y_val   (np.ndarray): Validation labels.

        Returns:
            dict: A dictionary with keys 'hidden_size' and 'dropout_rate'.
        """
        param_grid: Dict[str, List[Union[int, float]]] = {
            'hidden_size': [16, 32, 64],  # explicitly integers
            'dropout_rate': [0.3, 0.5, 0.7],
        }

        best_val_acc: float = -1
        best_params: Dict[str, Union[int, float]] = {
            'hidden_size': self.hidden_layers[0],
            'dropout_rate': self.dropout_rate
        }

        for h in param_grid['hidden_size']:
            for dr in param_grid['dropout_rate']:
                # Create a temporary network
                temp_nn = NeuralNet(
                    input_size=X_train.shape[1],
                    hidden_size=[int(h)],  # convert to list of int
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
                    early_stopping=False  # skip early stopping for quick search
                )
                temp_nn.fit(X_train, y_train)

                # Evaluate on validation
                probs = temp_nn.predict_proba(X_val)
                predictions = np.argmax(probs, axis=1)
                val_acc = np.mean(predictions == y_val)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = {'hidden_size': h, 'dropout_rate': dr}

        print(f"Best hyperparameters from grid search: {best_params}, val_acc={best_val_acc:.4f}")
        return best_params
