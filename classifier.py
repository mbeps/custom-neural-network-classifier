import numpy as np

class Classifier:
    """
    A custom neural network classifier for Pacman's movement decisions.
    Implements a two-layer neural network trained with backpropagation.
    """

    def __init__(self):
        """Initialise the classifier with default parameters."""
        self.input_size = None  # Determined during first fit call
        self.hidden_size = 32   # Number of neurons in the hidden layer
        self.output_size = 4    # Corresponding to 4 possible actions
        self.weights1 = None    # Input to hidden layer weights
        self.weights2 = None    # Hidden to output layer weights
        self.bias1 = None       # Hidden layer bias
        self.bias2 = None       # Output layer bias
        self.learning_rate = 0.01
        self.epochs = 150
        self.l2_lambda = 0.001  # L2 regularisation strength
        print(f"Classifier initialised with {self.hidden_size} hidden units and learning rate {self.learning_rate}")

    def reset(self):
        """Re-initialise the model's weights and biases."""
        print("\nResetting model weights and biases...")
        if self.input_size is not None:
            # He initialisation for ReLU activation
            self.weights1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
            self.weights2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2. / self.hidden_size)
            self.bias1 = np.zeros((1, self.hidden_size))
            self.bias2 = np.zeros((1, self.output_size))
            print(f"Model reset complete. Input size: {self.input_size}, Hidden size: {self.hidden_size}, Output size: {self.output_size}")

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

        # Initialise model parameters on first fit
        if self.input_size is None:
            self.input_size = X.shape[1]
            print(f"First fit - initialising input size to {self.input_size}")
            self.reset()

        print("\nTraining progress:")
        print("Epoch\tLoss\t\tTraining Accuracy")
        print("-" * 40)

        # Convert targets to one-hot encoding
        y_onehot = np.eye(self.output_size)[y_integer]

        # Training loop
        for epoch in range(self.epochs):
            # Shuffle data each epoch
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_integer_shuffled = y_integer[indices]  # Ensure labels match data order
            y_shuffled = y_onehot[indices]

            # Forward pass
            hidden_layer = np.dot(X_shuffled, self.weights1) + self.bias1
            hidden_activation = np.maximum(0, hidden_layer)  # ReLU

            output_layer = np.dot(hidden_activation, self.weights2) + self.bias2
            # Softmax with numerical stability
            shifted_logits = output_layer - np.max(output_layer, axis=1, keepdims=True)
            exp_scores = np.exp(shifted_logits)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Compute loss (cross-entropy with L2 regularisation)
            correct_logprobs = -np.log(probs[range(X.shape[0]), y_integer_shuffled] + 1e-8)
            data_loss = np.sum(correct_logprobs) / X.shape[0]
            reg_loss = 0.5 * self.l2_lambda * (np.sum(self.weights1**2) + np.sum(self.weights2**2))
            total_loss = data_loss + reg_loss

            # Backward pass
            d_output = probs.copy()
            d_output[range(X.shape[0]), y_integer_shuffled] -= 1
            d_output /= X.shape[0]

            # Compute gradients
            dW2 = np.dot(hidden_activation.T, d_output) + self.l2_lambda * self.weights2
            db2 = np.sum(d_output, axis=0, keepdims=True)
            d_hidden = np.dot(d_output, self.weights2.T)
            d_hidden[hidden_layer <= 0] = 0  # ReLU derivative

            dW1 = np.dot(X_shuffled.T, d_hidden) + self.l2_lambda * self.weights1
            db1 = np.sum(d_hidden, axis=0, keepdims=True)

            # Update parameters
            self.weights2 -= self.learning_rate * dW2
            self.bias2 -= self.learning_rate * db2
            self.weights1 -= self.learning_rate * dW1
            self.bias1 -= self.learning_rate * db1

            # Check model learning: calculate training accuracy every 10 epochs
            if epoch % 10 == 0:
                # Forward pass on the entire training set
                hidden_all = np.dot(X, self.weights1) + self.bias1
                hidden_all_activation = np.maximum(0, hidden_all)
                output_all = np.dot(hidden_all_activation, self.weights2) + self.bias2
                shifted_logits_all = output_all - np.max(output_all, axis=1, keepdims=True)
                exp_scores_all = np.exp(shifted_logits_all)
                probs_all = exp_scores_all / np.sum(exp_scores_all, axis=1, keepdims=True)
                predictions = np.argmax(probs_all, axis=1)
                accuracy = np.mean(predictions == y_integer)

                print(f"{epoch}\t{total_loss:.4f}\t\t{accuracy*100:.2f}%")

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

        # Forward pass
        hidden_layer = np.dot(X, self.weights1) + self.bias1
        hidden_activation = np.maximum(0, hidden_layer)
        output_layer = np.dot(hidden_activation, self.weights2) + self.bias2

        # Softmax
        shifted_logits = output_layer - np.max(output_layer)
        exp_scores = np.exp(shifted_logits)
        probs = exp_scores / np.sum(exp_scores)

        # Convert legal moves to numerical actions
        legal_actions = []
        for move in legal:
            if move == 'North': legal_actions.append(0)
            elif move == 'East': legal_actions.append(1)
            elif move == 'South': legal_actions.append(2)
            elif move == 'West': legal_actions.append(3)

        if not legal_actions:
            return np.random.randint(4)  # Fallback if no legal actions

        # Select action with highest probability among legal options
        legal_probs = probs[0][legal_actions]
        best_index = np.argmax(legal_probs)
        return legal_actions[best_index]
