import numpy as np

class NeuralNet:
    """
    Encapsulates the neural network logic for training and prediction.
    Includes:
      - Batch Normalisation (optional)
      - Dropout (optional)
      - Momentum
      - Learning Rate Decay
      - Gradient Clipping (optional)
    """

    def __init__(
        self,
        input_size,
        hidden_size=32,
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
        grad_clip_norm=5.0
    ):
        """
        Parameters:
            input_size     (int): Number of input features.
            hidden_size    (int): Number of neurons in the hidden layer.
            output_size    (int): Number of output classes.
            learning_rate  (float): Initial learning rate.
            epochs         (int): Number of training epochs.
            l2_lambda      (float): L2 regularisation strength.
            batch_size     (int): Mini-batch size; if None, use full-batch.
            descent_type   (str): 'batch' or 'mini-batch'.
            momentum       (float): Momentum coefficient for updates.
            lr_decay       (float): Exponential decay rate for the learning rate.
            use_batchnorm  (bool): Whether to use batch normalisation in the hidden layer.
            use_dropout    (bool): Whether to apply dropout in the hidden layer.
            dropout_rate   (float): Dropout probability.
            use_grad_clip  (bool): Whether to apply gradient clipping.
            grad_clip_norm (float): Max norm for gradients.
        """
        # Core network structure
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size
        self.descent_type = descent_type
        self.momentum = momentum
        self.lr_decay = lr_decay

        # Optional features
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_grad_clip = use_grad_clip
        self.grad_clip_norm = grad_clip_norm

        # Initialise weights, biases, velocities (for momentum)
        self._init_parameters()
        self._init_velocity()

        # If using batch norm, initialise parameters for BN
        if self.use_batchnorm:
            self.gamma = np.ones((1, self.hidden_size), dtype=np.float32)
            self.beta = np.zeros((1, self.hidden_size), dtype=np.float32)

            # Running stats for inference
            self.running_mean = np.zeros((1, self.hidden_size), dtype=np.float32)
            self.running_var = np.ones((1, self.hidden_size), dtype=np.float32)

            # Momentum factor for updating running mean/var in BN
            self.bn_momentum = 0.9

    def _init_parameters(self):
        """Initialise weights and biases using He initialisation."""
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2. / self.hidden_size)
        self.bias1 = np.zeros((1, self.hidden_size))
        self.bias2 = np.zeros((1, self.output_size))

    def _init_velocity(self):
        """Initialise velocity terms for momentum-based updates (set to zero initially)."""
        self.velocity_w1 = np.zeros_like(self.weights1)
        self.velocity_b1 = np.zeros_like(self.bias1)
        self.velocity_w2 = np.zeros_like(self.weights2)
        self.velocity_b2 = np.zeros_like(self.bias2)

        # If using batch norm, we also track velocity for gamma and beta
        if self.use_batchnorm:
            self.velocity_gamma = np.zeros((1, self.hidden_size))
            self.velocity_beta = np.zeros((1, self.hidden_size))

    def reset(self):
        """Reset the network's weights, biases, velocities, and any BN stats."""
        self._init_parameters()
        self._init_velocity()
        if self.use_batchnorm:
            self.running_mean = np.zeros((1, self.hidden_size), dtype=np.float32)
            self.running_var = np.ones((1, self.hidden_size), dtype=np.float32)

    def fit(self, X, y_integer):
        """
        Train the network using the provided features and integer-encoded target labels.
        """
        for epoch in range(self.epochs):
            # Apply learning rate decay
            current_lr = self.learning_rate * (1.0 - self.lr_decay) ** epoch

            # Shuffle data
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y_integer[indices]

            # Training loop
            if self.descent_type == 'batch' or self.batch_size is None:
                # Full batch
                hidden_activation, cache, probs = self._forward_pass(X_shuffled, training=True)
                total_loss, _ = self._compute_loss(probs, y_shuffled)
                dW1, db1, dW2, db2, dgamma, dbeta = self._backward_pass(X_shuffled, y_shuffled, cache, probs)
                self._update_parameters(dW1, db1, dW2, db2, dgamma, dbeta, current_lr)
            else:
                # Mini-batches
                for start_idx in range(0, X_shuffled.shape[0], self.batch_size):
                    end_idx = start_idx + self.batch_size
                    X_batch = X_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]

                    hidden_activation, cache, probs = self._forward_pass(X_batch, training=True)
                    total_loss, _ = self._compute_loss(probs, y_batch)
                    dW1, db1, dW2, db2, dgamma, dbeta = self._backward_pass(X_batch, y_batch, cache, probs)
                    self._update_parameters(dW1, db1, dW2, db2, dgamma, dbeta, current_lr)

            # Print every 10 epochs
            if epoch % 10 == 0:
                accuracy = self._compute_accuracy(X, y_integer)
                print(f"{epoch}\t{total_loss:.4f}\t\t{accuracy * 100:.2f}%")

    def predict_proba(self, X):
        """
        Compute the network's output probabilities for the given features (inference).
        """
        _, _, probs = self._forward_pass(X, training=False)
        return probs

    def _forward_pass(self, X, training=True):
        """
        Forward pass through the network:
          1. Linear -> BN (optional) -> ReLU -> Dropout (optional)
          2. Linear -> Softmax
        Returns:
            hidden_activation: Post-activation outputs of the hidden layer (used for backprop).
            cache: dictionary containing intermediate values for backprop.
            probs: final output probabilities.
        """
        # Linear transform for hidden layer
        z1 = np.dot(X, self.weights1) + self.bias1

        # Batch Normalisation on z1, if enabled
        if self.use_batchnorm:
            if training:
                # Compute batch stats
                mu = np.mean(z1, axis=0, keepdims=True)
                var = np.var(z1, axis=0, keepdims=True)

                # Normalise
                z1_hat = (z1 - mu) / np.sqrt(var + 1e-5)

                # Scale and shift
                z1_bn = self.gamma * z1_hat + self.beta

                # Update running stats
                self.running_mean = self.bn_momentum * self.running_mean + (1 - self.bn_momentum) * mu
                self.running_var = self.bn_momentum * self.running_var + (1 - self.bn_momentum) * var
            else:
                # Use running mean and var at inference
                z1_hat = (z1 - self.running_mean) / np.sqrt(self.running_var + 1e-5)
                z1_bn = self.gamma * z1_hat + self.beta

            # We'll feed z1_hat for backprop
            # Because ReLU is next, we feed z1_bn into ReLU
            a1_before_relu = z1_bn
        else:
            # No batch norm
            a1_before_relu = z1

        # ReLU activation
        hidden_activation = np.maximum(0, a1_before_relu)

        # Dropout, if enabled
        # (Only apply dropout during training)
        if self.use_dropout and training:
            dropout_mask = (np.random.rand(*hidden_activation.shape) > self.dropout_rate).astype(hidden_activation.dtype)
            hidden_activation *= dropout_mask
        else:
            dropout_mask = np.ones_like(hidden_activation)

        # Output layer
        z2 = np.dot(hidden_activation, self.weights2) + self.bias2

        # Softmax with numerical stability
        shifted_logits = z2 - np.max(z2, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Cache for backprop
        cache = {
            'X': X,
            'z1': z1,
            'z2': z2,
            'a1_before_relu': a1_before_relu,
            'hidden_activation': hidden_activation,
            'dropout_mask': dropout_mask
        }

        if self.use_batchnorm:
            cache['z1_hat'] = z1_hat  # Normalised input
            cache['mu'] = mu if training else self.running_mean
            cache['var'] = var if training else self.running_var

        return hidden_activation, cache, probs

    def _compute_loss(self, probs, y_integer):
        """
        Compute the cross-entropy loss with L2 regularisation.
        """
        correct_logprobs = -np.log(probs[np.arange(probs.shape[0]), y_integer] + 1e-8)
        data_loss = np.mean(correct_logprobs)
        reg_loss = 0.5 * self.l2_lambda * (np.sum(self.weights1**2) + np.sum(self.weights2**2))
        total_loss = data_loss + reg_loss
        return total_loss, data_loss

    def _backward_pass(self, X, y_integer, cache, probs):
        """
        Backprop through the network:
          - Softmax derivative
          - Hidden layer (with optional BN, ReLU, Dropout)
          - Compute parameter gradients
        Returns:
            Gradients for weights1, bias1, weights2, bias2, gamma, beta
        """
        N = X.shape[0]

        # Softmax derivative
        d_output = probs.copy()
        d_output[np.arange(N), y_integer] -= 1
        d_output /= N

        # Grad w.r.t. weights2, bias2
        hidden_activation = cache['hidden_activation']
        dW2 = np.dot(hidden_activation.T, d_output) + self.l2_lambda * self.weights2
        db2 = np.sum(d_output, axis=0, keepdims=True)

        # Backprop into hidden_activation
        d_hidden = np.dot(d_output, self.weights2.T)

        # Dropout
        d_hidden *= cache['dropout_mask']

        # ReLU derivative
        da1_before_relu = (cache['a1_before_relu'] > 0).astype(d_hidden.dtype) * d_hidden

        # If using batch norm, backprop through BN
        dgamma = np.zeros_like(self.gamma) if self.use_batchnorm else None
        dbeta = np.zeros_like(self.beta) if self.use_batchnorm else None

        if self.use_batchnorm:
            z1_hat = cache['z1_hat']
            mu = cache['mu']
            var = cache['var']
            eps = 1e-5

            # Derivatives for BN
            d_z1_hat = da1_before_relu * self.gamma
            d_var = np.sum(d_z1_hat * (cache['z1'] - mu) * -0.5 * (var + eps)**-1.5, axis=0, keepdims=True)
            d_mu = np.sum(d_z1_hat * -1.0 / np.sqrt(var + eps), axis=0, keepdims=True) + d_var * np.mean(-2.0 * (cache['z1'] - mu), axis=0, keepdims=True)

            # Grad w.r.t. z1
            dz1 = (d_z1_hat / np.sqrt(var + eps)) + (d_var * 2.0 * (cache['z1'] - mu) / N) + (d_mu / N)

            # Grad w.r.t. gamma, beta
            dgamma = np.sum(da1_before_relu * z1_hat, axis=0, keepdims=True)
            dbeta = np.sum(da1_before_relu, axis=0, keepdims=True)

            # Now replace da1_before_relu with dz1 for next step
            da1_before_relu = dz1

        # Finally, backprop into the first layer's weights/bias
        dW1 = np.dot(X.T, da1_before_relu) + self.l2_lambda * self.weights1
        db1 = np.sum(da1_before_relu, axis=0, keepdims=True)

        return dW1, db1, dW2, db2, dgamma, dbeta

    def _update_parameters(self, dW1, db1, dW2, db2, dgamma, dbeta, current_lr):
        """
        Update the parameters using momentum and optional gradient clipping.
        """
        # Optionally clip gradients
        if self.use_grad_clip:
            # Collect all gradient arrays in a list
            grad_list = [dW1, db1, dW2, db2]
            if self.use_batchnorm:
                grad_list += [dgamma, dbeta]

            # Compute global norm
            total_norm = 0
            for g in grad_list:
                total_norm += np.sum(g * g)
            total_norm = np.sqrt(total_norm)

            if total_norm > self.grad_clip_norm:
                scale_factor = self.grad_clip_norm / (total_norm + 1e-6)
                for g in grad_list:
                    g *= scale_factor

        # Momentum updates for W1, b1
        self.velocity_w1 = self.momentum * self.velocity_w1 - current_lr * dW1
        self.velocity_b1 = self.momentum * self.velocity_b1 - current_lr * db1
        self.weights1 += self.velocity_w1
        self.bias1 += self.velocity_b1

        # Momentum updates for W2, b2
        self.velocity_w2 = self.momentum * self.velocity_w2 - current_lr * dW2
        self.velocity_b2 = self.momentum * self.velocity_b2 - current_lr * db2
        self.weights2 += self.velocity_w2
        self.bias2 += self.velocity_b2

        # If using batchnorm, update gamma, beta with momentum
        if self.use_batchnorm:
            self.velocity_gamma = self.momentum * self.velocity_gamma - current_lr * dgamma
            self.velocity_beta = self.momentum * self.velocity_beta - current_lr * dbeta
            self.gamma += self.velocity_gamma
            self.beta += self.velocity_beta

    def _compute_accuracy(self, X, y_integer):
        """Compute the accuracy on the provided dataset."""
        probs = self.predict_proba(X)
        predictions = np.argmax(probs, axis=1)
        return np.mean(predictions == y_integer)


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
        self.hidden_size = 32   # Number of neurons in the hidden layer
        self.output_size = 4    # 4 possible actions
        self.learning_rate = 0.01
        self.epochs = 150
        self.l2_lambda = 0.001  # L2 regularisation strength

        # Mini-batch settings
        self.batch_size = 32
        self.descent_type = 'mini-batch'

        # Momentum, LR decay
        self.momentum = 0.9
        self.lr_decay = 0.01

        # Optional features
        self.use_batchnorm = True
        self.use_dropout = True
        self.dropout_rate = 0.5
        self.use_grad_clip = True
        self.grad_clip_norm = 5.0

        print(f"Classifier initialised with {self.hidden_size} hidden units and learning rate {self.learning_rate}")

    def reset(self):
        """
        Reset the classifier's neural network if it has been initialised.
        """
        print("\nResetting model weights, biases, and BN stats...")
        if self.nn is not None:
            self.nn.reset()
            print("Model reset complete.")

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

        # Initialise the neural network on first fit
        if self.nn is None:
            input_size = X.shape[1]
            print(f"First fit - initialising input size to {input_size}")
            self.nn = NeuralNet(
                input_size=input_size,
                hidden_size=self.hidden_size,
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
                dropout_rate=self.dropout_rate,
                use_grad_clip=self.use_grad_clip,
                grad_clip_norm=self.grad_clip_norm
            )

        print("\nTraining progress:")
        print("Epoch\tLoss\t\tTraining Accuracy")
        print("-" * 40)

        self.nn.fit(X, y_integer)

    def predict(self, features, legal) -> int:
        """
        Predict the best action given current features and legal moves.

        Parameters:
            features (list): Binary feature vector from the game state.
            legal (list): Legal directions.

        Returns:
            int: Predicted action (0-3).
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

        # If no legal actions, choose a random fallback
        if not legal_actions:
            return np.random.randint(4)

        # Select the legal action with the highest probability
        legal_probs = probs[0][legal_actions]
        best_index = np.argmax(legal_probs)
        return legal_actions[best_index]
