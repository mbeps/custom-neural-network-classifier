import numpy as np

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
        input_size,
        hidden_size=32,  # Can be int or list of length 1
        output_size=4,
        learning_rate=0.01,
        epochs=150,
        l2_lambda=0.001,
        batch_size=None,
        descent_type='batch',
        momentum=0.9,
        lr_decay=0.0,
        use_batchnorm=False,
        use_dropout=False,
        dropout_rate=0.5,
        use_grad_clip=False,
        grad_clip_norm=5.0,
        # Early Stopping settings
        early_stopping=False,
        patience=10,
        validation_data=None
    ):
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
            batch_size       (int): Mini-batch size; None => full batch.
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
        self.input_size = input_size
        self.hidden_size = hidden_size[0]  # single hidden layer
        self.output_size = output_size

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size
        self.descent_type = descent_type
        self.momentum = momentum
        self.lr_decay = lr_decay

        # Additional features
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_grad_clip = use_grad_clip
        self.grad_clip_norm = grad_clip_norm

        # Early Stopping
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_data = validation_data

        # For tracking improvements
        self.best_weights = {}
        self.best_val_loss = np.inf
        self.no_improvement_count = 0

        # Initialisation
        self._init_parameters()
        self._init_velocity()

        # Batch norm parameters (for the single hidden layer)
        if self.use_batchnorm:
            self.gamma = np.ones((1, self.hidden_size), dtype=np.float32)
            self.beta = np.zeros((1, self.hidden_size), dtype=np.float32)
            self.running_mean = np.zeros((1, self.hidden_size), dtype=np.float32)
            self.running_var = np.ones((1, self.hidden_size), dtype=np.float32)
            self.bn_momentum = 0.9

    def _init_parameters(self):
        # Single hidden layer: weights1 => input->hidden, weights2 => hidden->output
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2. / self.hidden_size)
        self.bias1 = np.zeros((1, self.hidden_size))
        self.bias2 = np.zeros((1, self.output_size))

    def _init_velocity(self):
        self.velocity_w1 = np.zeros_like(self.weights1)
        self.velocity_b1 = np.zeros_like(self.bias1)
        self.velocity_w2 = np.zeros_like(self.weights2)
        self.velocity_b2 = np.zeros_like(self.bias2)
        if self.use_batchnorm:
            self.velocity_gamma = np.zeros((1, self.hidden_size))
            self.velocity_beta = np.zeros((1, self.hidden_size))

    def reset(self):
        self._init_parameters()
        self._init_velocity()
        if self.use_batchnorm:
            self.running_mean = np.zeros((1, self.hidden_size), dtype=np.float32)
            self.running_var = np.ones((1, self.hidden_size), dtype=np.float32)
        self.best_weights = {}
        self.best_val_loss = np.inf
        self.no_improvement_count = 0

    def fit(self, X, y_integer):
        for epoch in range(self.epochs):
            # Learning rate decay
            current_lr = self.learning_rate * (1.0 - self.lr_decay) ** epoch

            # Shuffle data
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y_integer[indices]

            # Batch or mini-batch
            if self.descent_type == 'batch' or self.batch_size is None:
                # Single batch
                _, cache, probs = self._forward_pass(X_shuffled, training=True)
                total_loss, _ = self._compute_loss(probs, y_shuffled)
                dW1, db1, dW2, db2, dgamma, dbeta = self._backward_pass(X_shuffled, y_shuffled, cache, probs)
                self._update_parameters(dW1, db1, dW2, db2, dgamma, dbeta, current_lr)
            else:
                # Mini-batches
                for start_idx in range(0, X_shuffled.shape[0], self.batch_size):
                    end_idx = start_idx + self.batch_size
                    X_batch = X_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]

                    _, cache, probs = self._forward_pass(X_batch, training=True)
                    total_loss, _ = self._compute_loss(probs, y_batch)
                    dW1, db1, dW2, db2, dgamma, dbeta = self._backward_pass(X_batch, y_batch, cache, probs)
                    self._update_parameters(dW1, db1, dW2, db2, dgamma, dbeta, current_lr)

            # Print every 10 epochs
            if epoch % 10 == 0:
                accuracy = self._compute_accuracy(X, y_integer)
                print(f"{epoch}\t{total_loss:.4f}\t\t{accuracy * 100:.2f}%")

            # Early Stopping check
            if self.early_stopping and self.validation_data is not None:
                X_val, y_val = self.validation_data
                val_probs = self.predict_proba(X_val)
                val_loss, _ = self._compute_loss(val_probs, y_val)

                if val_loss < self.best_val_loss:
                    # Improvement
                    self.best_val_loss = val_loss
                    self.no_improvement_count = 0
                    # Save current best weights
                    self._save_current_weights()
                else:
                    self.no_improvement_count += 1
                    # If patience exceeded, stop
                    if self.no_improvement_count >= self.patience:
                        print(f"Early stopping at epoch {epoch}.")
                        # Optionally restore best weights
                        self._restore_best_weights()
                        break

    def predict_proba(self, X):
        _, _, probs = self._forward_pass(X, training=False)
        return probs

    def _forward_pass(self, X, training=True):
        # First layer: input -> hidden
        z1 = np.dot(X, self.weights1) + self.bias1

        if self.use_batchnorm:
            if training:
                mu = np.mean(z1, axis=0, keepdims=True)
                var = np.var(z1, axis=0, keepdims=True)
                z1_hat = (z1 - mu) / np.sqrt(var + 1e-5)
                z1_bn = self.gamma * z1_hat + self.beta

                self.running_mean = self.bn_momentum * self.running_mean + (1 - self.bn_momentum) * mu
                self.running_var = self.bn_momentum * self.running_var + (1 - self.bn_momentum) * var
            else:
                z1_hat = (z1 - self.running_mean) / np.sqrt(self.running_var + 1e-5)
                z1_bn = self.gamma * z1_hat + self.beta

            a1_before_relu = z1_bn
        else:
            a1_before_relu = z1

        hidden_activation = np.maximum(0, a1_before_relu)

        # Dropout on hidden activation
        if self.use_dropout and training:
            dropout_mask = (np.random.rand(*hidden_activation.shape) > self.dropout_rate).astype(hidden_activation.dtype)
            hidden_activation *= dropout_mask
        else:
            dropout_mask = np.ones_like(hidden_activation)

        # Second layer: hidden -> output
        z2 = np.dot(hidden_activation, self.weights2) + self.bias2

        # Softmax
        shifted_logits = z2 - np.max(z2, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        cache = {
            'X': X,
            'z1': z1,
            'hidden_activation': hidden_activation,
            'dropout_mask': dropout_mask,
            'z2': z2,
            'probs': probs
        }

        if self.use_batchnorm:
            cache['z1_hat'] = z1_hat
            # Store for backprop
            cache['mu'] = mu if training else self.running_mean
            cache['var'] = var if training else self.running_var

        return hidden_activation, cache, probs

    def _compute_loss(self, probs, y_integer):
        correct_logprobs = -np.log(probs[np.arange(probs.shape[0]), y_integer] + 1e-8)
        data_loss = np.mean(correct_logprobs)
        reg_loss = 0.5 * self.l2_lambda * (np.sum(self.weights1**2) + np.sum(self.weights2**2))
        total_loss = data_loss + reg_loss
        return total_loss, data_loss

    def _backward_pass(self, X, y_integer, cache, probs):
        N = X.shape[0]
        d_output = probs.copy()
        d_output[np.arange(N), y_integer] -= 1
        d_output /= N

        hidden_activation = cache['hidden_activation']
        dW2 = np.dot(hidden_activation.T, d_output) + self.l2_lambda * self.weights2
        db2 = np.sum(d_output, axis=0, keepdims=True)
        d_hidden = np.dot(d_output, self.weights2.T)

        # Dropout
        d_hidden *= cache['dropout_mask']

        # ReLU
        da1_before_relu = (cache['z1'] > 0).astype(d_hidden.dtype) * d_hidden

        dgamma = None
        dbeta = None
        if self.use_batchnorm:
            z1_hat = cache['z1_hat']
            mu = cache['mu']
            var = cache['var']
            eps = 1e-5

            d_z1_hat = da1_before_relu * self.gamma
            d_var = np.sum(d_z1_hat * (cache['z1'] - mu) * -0.5 * (var + eps)**-1.5, axis=0, keepdims=True)
            d_mu = np.sum(d_z1_hat * -1.0 / np.sqrt(var + eps), axis=0, keepdims=True) \
                   + d_var * np.mean(-2.0*(cache['z1']-mu), axis=0, keepdims=True)

            dz1 = (d_z1_hat / np.sqrt(var + eps)) + (d_var * 2.0*(cache['z1']-mu)/N) + (d_mu/N)

            dgamma = np.sum(da1_before_relu * z1_hat, axis=0, keepdims=True)
            dbeta = np.sum(da1_before_relu, axis=0, keepdims=True)

            da1_before_relu = dz1

        dW1 = np.dot(X.T, da1_before_relu) + self.l2_lambda * self.weights1
        db1 = np.sum(da1_before_relu, axis=0, keepdims=True)

        return dW1, db1, dW2, db2, dgamma, dbeta

    def _update_parameters(self, dW1, db1, dW2, db2, dgamma, dbeta, current_lr):
        # Gradient clipping
        if self.use_grad_clip:
            grad_list = [dW1, db1, dW2, db2]
            if self.use_batchnorm and dgamma is not None and dbeta is not None:
                grad_list += [dgamma, dbeta]
            total_norm = np.sqrt(sum(np.sum(g*g) for g in grad_list))
            if total_norm > self.grad_clip_norm:
                scale = self.grad_clip_norm / (total_norm + 1e-8)
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

    def _compute_accuracy(self, X, y_integer):
        probs = self.predict_proba(X)
        predictions = np.argmax(probs, axis=1)
        return np.mean(predictions == y_integer)

    # ----- Early Stopping helpers -----

    def _save_current_weights(self):
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

    def _restore_best_weights(self):
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

    def __init__(self):
        """
        Initialise the classifier. The neural network will be created on the first call to fit.
        """
        self.nn = None
        # Now store the hidden layer in a list form to reflect the new approach:
        self.hidden_layers = [24]   # Single hidden layer with 32 neurons
        self.output_size = 4       # 4 possible actions
        self.learning_rate = 0.01
        self.epochs = 150
        self.l2_lambda = 0.001     # L2 regularisation strength

        self.batch_size = 32
        self.descent_type = 'mini-batch'
        self.momentum = 0.9
        self.lr_decay = 0.01

        # Additional features
        self.use_batchnorm = True
        self.use_dropout = True
        self.dropout_rate = 0.5
        self.use_grad_clip = True
        self.grad_clip_norm = 5.0

        # Early Stopping
        self.early_stopping = False
        self.patience = 10

        # Hyperparameter search toggle
        self.enable_grid_search = True

        print(f"Classifier initialised with hidden_layers={self.hidden_layers}, learning_rate={self.learning_rate}")

    def reset(self):
        """
        Reset the classifier's neural network if it has been initialised.
        """
        print("\nResetting model weights, biases, and BN stats...")
        if self.nn is not None:
            self.nn.reset()

    def fit(self, data, target):
        """
        Train the neural network using the provided data and targets.

        Parameters:
            data (list): Feature vectors from the game states.
            target (list): Corresponding action labels (0-3).
        """
        X = np.array(data, dtype=np.float32)
        y_integer = np.array(target, dtype=np.int32)

        print(f"\nStarting training with {len(X)} samples...")

        # Simple train/validation split for early stopping + hyperparam search
        val_proportion = 0.2
        split_index = int(len(X) * (1.0 - val_proportion))

        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y_integer[:split_index], y_integer[split_index:]

        # Optional hyperparameter tuning
        if self.enable_grid_search:
            best_params = self._grid_search_hyperparams(X_train, y_train, X_val, y_val)
            self._init_nn(best_params, X_train.shape[1], (X_val, y_val))
        else:
            # Initialise with default hyperparameters
            self._init_nn(None, X.shape[1], (X_val, y_val))

        print("\nTraining progress (with early stopping if enabled):")
        print("Epoch\tLoss\t\tTraining Accuracy")
        print("-" * 40)
        self.nn.fit(X_train, y_train)

    def predict(self, features, legal) -> int:
        """
        Predict the best action given current features and legal moves.
        """
        X = np.array(features, dtype=np.float32).reshape(1, -1)
        probs = self.nn.predict_proba(X)

        # Convert legal moves to numerical actions
        legal_actions = []
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

    def _init_nn(self, params, input_size, val_data):
        """
        Initialise the neural network with given parameters or defaults.
        """
        if params is None:
            hidden_size = self.hidden_layers[0]
            dropout_rate = self.dropout_rate
        else:
            hidden_size = params['hidden_size']
            dropout_rate = params['dropout_rate']

        print(f"Initialising NeuralNet with hidden_size={hidden_size}, dropout_rate={dropout_rate}")
        self.nn = NeuralNet(
            input_size=input_size,
            hidden_size=hidden_size,    # can be int or [int], but internally it’s handled
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

    def _grid_search_hyperparams(self, X_train, y_train, X_val, y_val):
        """
        A simple grid search to find the best combination of hidden_size & dropout_rate.
        You can expand this with more parameters.
        """
        param_grid = {
            'hidden_size': [16, 32, 64],  # these will be interpreted as single-layer sizes
            'dropout_rate': [0.3, 0.5, 0.7],
        }

        best_val_acc = -1
        best_params = {'hidden_size': self.hidden_layers[0], 'dropout_rate': self.dropout_rate}

        for h in param_grid['hidden_size']:
            for dr in param_grid['dropout_rate']:
                # Create a temporary network
                temp_nn = NeuralNet(
                    input_size=X_train.shape[1],
                    hidden_size=h,
                    output_size=self.output_size,
                    learning_rate=self.learning_rate,
                    epochs=20,  # shorter training
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
                    early_stopping=False,  # skip early stopping for quick search
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
