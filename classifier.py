import numpy as np

class NeuralNet:
    """
    Encapsulates the neural network logic for training and prediction.
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
        lr_decay=0.0
    ):
        """
        Parameters:
            input_size   (int): Number of input features.
            hidden_size  (int): Number of neurons in the hidden layer.
            output_size  (int): Number of output classes.
            learning_rate(float): Initial learning rate for parameter updates.
            epochs       (int): Number of training iterations.
            l2_lambda    (float): L2 regularisation strength.
            batch_size   (int): Size of each mini-batch. If None, batch gradient descent is used.
            descent_type (str): 'batch' or 'mini-batch'.
            momentum     (float): Momentum coefficient for accelerating gradient descent.
            lr_decay     (float): Exponential decay rate for the learning rate per epoch.
        """
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

        # Initialise weights, biases, and velocities
        self._init_parameters()
        self._init_velocity()

    def _init_parameters(self):
        """
        Initialise weights and biases.
        """
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2. / self.hidden_size)
        self.bias1 = np.zeros((1, self.hidden_size))
        self.bias2 = np.zeros((1, self.output_size))

    def _init_velocity(self):
        """
        Initialise velocity terms for momentum-based updates (set to zero initially).
        """
        self.velocity_w1 = np.zeros_like(self.weights1)
        self.velocity_b1 = np.zeros_like(self.bias1)
        self.velocity_w2 = np.zeros_like(self.weights2)
        self.velocity_b2 = np.zeros_like(self.bias2)

    def reset(self):
        """
        Reset the network's weights, biases, and velocities.
        """
        self._init_parameters()
        self._init_velocity()

    def fit(self, X, y_integer):
        """
        Train the network using the provided features and integer-encoded target labels.

        Parameters:
            X (np.ndarray): Feature vectors.
            y_integer (np.ndarray): Target labels (0-3).
        """
        for epoch in range(self.epochs):
            # Decay the learning rate if lr_decay > 0
            current_lr = self.learning_rate * (1.0 - self.lr_decay) ** epoch

            # Shuffle the dataset each epoch
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y_integer[indices]

            # Decide on batch or mini-batch training
            if self.descent_type == 'batch' or self.batch_size is None:
                # One single batch for the entire dataset
                hidden_layer, hidden_activation, output_layer, probs = self._forward_pass(X_shuffled)
                total_loss, data_loss = self._compute_loss(probs, y_shuffled)
                dW1, db1, dW2, db2 = self._backward_pass(X_shuffled, y_shuffled, 
                                                        hidden_layer, hidden_activation, probs)
                self._update_parameters(dW1, db1, dW2, db2, current_lr)

            elif self.descent_type == 'mini-batch':
                # Split data into mini-batches
                for start_idx in range(0, X_shuffled.shape[0], self.batch_size):
                    end_idx = start_idx + self.batch_size
                    X_batch = X_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]

                    hidden_layer, hidden_activation, output_layer, probs = self._forward_pass(X_batch)
                    total_loss, data_loss = self._compute_loss(probs, y_batch)
                    dW1, db1, dW2, db2 = self._backward_pass(X_batch, y_batch,
                                                            hidden_layer, hidden_activation, probs)
                    self._update_parameters(dW1, db1, dW2, db2, current_lr)

            # Print training metrics every 10 epochs
            if epoch % 10 == 0:
                accuracy = self._compute_accuracy(X, y_integer)
                print(f"{epoch}\t{total_loss:.4f}\t\t{accuracy * 100:.2f}%")

    def predict_proba(self, X):
        """
        Compute the network's output probabilities for the given features.

        Parameters:
            X (np.ndarray): Feature vectors.

        Returns:
            probs (np.ndarray): Softmax probabilities.
        """
        _, _, _, probs = self._forward_pass(X)
        return probs

    def _forward_pass(self, X):
        """
        Compute a forward pass through the network.

        Returns:
            hidden_layer: Pre-activation values in the hidden layer.
            hidden_activation: ReLU activations.
            output_layer: Pre-softmax outputs.
            probs: Softmax probabilities.
        """
        hidden_layer = np.dot(X, self.weights1) + self.bias1
        hidden_activation = np.maximum(0, hidden_layer)
        output_layer = np.dot(hidden_activation, self.weights2) + self.bias2

        # Softmax with numerical stability
        shifted_logits = output_layer - np.max(output_layer, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return hidden_layer, hidden_activation, output_layer, probs

    def _compute_loss(self, probs, y_integer):
        """
        Compute the cross-entropy loss with L2 regularisation.

        Returns:
            total_loss (float): Sum of data loss and regularisation loss.
            data_loss (float): Cross-entropy loss.
        """
        correct_logprobs = -np.log(probs[range(probs.shape[0]), y_integer] + 1e-8)
        data_loss = np.mean(correct_logprobs)
        reg_loss = 0.5 * self.l2_lambda * (np.sum(self.weights1 ** 2) + np.sum(self.weights2 ** 2))
        total_loss = data_loss + reg_loss
        return total_loss, data_loss

    def _backward_pass(self, X, y_integer, hidden_layer, hidden_activation, probs):
        """
        Compute gradients via backpropagation.

        Returns:
            dW1, db1, dW2, db2: Gradients for weights and biases.
        """
        d_output = probs.copy()
        d_output[range(X.shape[0]), y_integer] -= 1
        d_output /= X.shape[0]

        dW2 = np.dot(hidden_activation.T, d_output) + self.l2_lambda * self.weights2
        db2 = np.sum(d_output, axis=0, keepdims=True)

        d_hidden = np.dot(d_output, self.weights2.T)
        d_hidden[hidden_layer <= 0] = 0

        dW1 = np.dot(X.T, d_hidden) + self.l2_lambda * self.weights1
        db1 = np.sum(d_hidden, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def _update_parameters(self, dW1, db1, dW2, db2, current_lr):
        """
        Update network parameters using momentum-based updates
        with the (optionally) decayed learning rate.
        """
        # Update velocity
        self.velocity_w1 = self.momentum * self.velocity_w1 - current_lr * dW1
        self.velocity_b1 = self.momentum * self.velocity_b1 - current_lr * db1
        self.velocity_w2 = self.momentum * self.velocity_w2 - current_lr * dW2
        self.velocity_b2 = self.momentum * self.velocity_b2 - current_lr * db2

        # Update weights and biases
        self.weights1 += self.velocity_w1
        self.bias1 += self.velocity_b1
        self.weights2 += self.velocity_w2
        self.bias2 += self.velocity_b2

    def _compute_accuracy(self, X, y_integer):
        """
        Compute the accuracy on the provided dataset.

        Returns:
            accuracy (float): Proportion of correct predictions.
        """
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
        self.output_size = 4    # Corresponding to 4 possible actions
        self.learning_rate = 0.01
        self.epochs = 150
        self.l2_lambda = 0.001  # L2 regularisation strength

        # Mini-batch settings
        self.batch_size = 32
        self.descent_type = 'mini-batch'

        # Momentum and decay hyperparameters
        self.momentum = 0.9
        self.lr_decay = 0.01

        print(f"Classifier initialised with {self.hidden_size} hidden units and learning rate {self.learning_rate}")

    def reset(self):
        """
        Reset the classifier's neural network if it has been initialised.
        """
        print("\nResetting model weights and biases...")
        if self.nn is not None:
            self.nn.reset()
            print(f"Model reset complete. Input size: {self.nn.input_size}, Hidden size: {self.hidden_size}, Output size: {self.output_size}")

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

        # Initialise the neural network on the first fit
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
                lr_decay=self.lr_decay
            )

        print("\nTraining progress:")
        print("Epoch\tLoss\t\tTraining Accuracy")
        print("-" * 40)

        # Train the neural network
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

        if not legal_actions:
            return np.random.randint(4)  # Fallback if no legal actions

        # Select the legal action with the highest probability
        legal_probs = probs[0][legal_actions]
        best_index = np.argmax(legal_probs)
        return legal_actions[best_index]
