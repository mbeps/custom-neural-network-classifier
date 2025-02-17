# Pacman Movement Classifier

This project implements a custom neural network classifier to control Pacman's movements. The classifier decides Pacman's next move based on the game state, which includes information about nearby walls, food, and ghosts. The implementation focuses on machine learning principles rather than optimal game performance.

The solution features a sophisticated neural network with batch normalisation, dropout regularisation, and momentum-based optimisation. While simpler approaches like k-NN or Naive Bayes could work, this implementation demonstrates advanced classification techniques with proper software engineering practices.

# Technical Features

## Core Classification Architecture
- Feed-forward neural network with configurable hidden layers
- ReLU activation functions for hidden layers
- Softmax output layer for 4-class movement prediction
- He initialisation for weights

## Optimisation Techniques
- Mini-batch gradient descent
- Momentum-based updates with velocity tracking
- Inverse scaling learning rate decay
- L2 regularisation
- Gradient clipping with norm thresholding

## Regularisation Methods
- Batch Normalisation
  - Running mean/variance tracking
  - Learnable scale (gamma) and shift (beta) parameters
  - Training/inference mode handling
- Dropout
  - Inverted dropout scaling
  - Configurable dropout rate

## Training Management
- Early stopping with validation monitoring
- Automatic hyperparameter optimisation via grid search
- Train/validation/test split (80/10/10)
- Best model checkpointing

## Implementation Features
- Type hints throughout codebase
- Numerical stability safeguards
- Comprehensive error handling
- Memory efficient operations
- Legal move validation and filtering
- Fallback strategies for edge cases

# Requirements
- Conda 
- Poetry with Python 3.10 or above

# Usage

## Using Conda
Since this project uses standard Python packages available in Anaconda, no additional installation is needed.

1. Run Pacman with the classifier:
```bash
python pacman.py --pacman ClassifierAgent
```

## Using Poetry
1. Install dependencies:
```bash
poetry install
```

2. Run Pacman with the classifier:
```bash
poetry run python pacman.py --pacman ClassifierAgent
```

Additional Commands:
```bash
# Play manually using keyboard
python pacman.py

# Run with a random agent
python pacman.py --pacman RandomAgent

# Generate training data while playing
python pacman.py --p TraceAgent
```

# Sources
- https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/
- https://optimization.cbe.cornell.edu/index.php?title=Momentum
- https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/
- https://arxiv.org/abs/1908.01878
- https://towardsdatascience.com/early-stopping-a-cool-strategy-to-regularize-neural-networks-bfdeca6d722e/
- https://www.datacamp.com/tutorial/loss-function-in-machine-learning
- https://www.datacamp.com/tutorial/batch-normalization-tensorflow
- https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9/
- https://builtin.com/machine-learning/relu-activation-function
- https://www.analyticsvidhya.com/blog/2021/04/introduction-to-softmax-for-neural-network/
- https://arxiv.org/abs/1502.01852
