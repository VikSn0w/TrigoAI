# TrigoAI
A series of AI experiments about trigonometric functions
This project has borned with the purpose of better understading how neural networks work by starting with an easy concept.
The first code is about the Sine function. 

## SineApproximator
### Version 1
The base dataset is a sample pool of 10000 values between 0 and 2π.

What I've tried so far to upgrade the NN:
- Use as activation function ReLU and Leaky ReLU, but noticed a great decrease in performance
- Implemented a backdrop logic but a great decrease in performance (further studies are necessary)

### Version 2 

The two versions of the SineApproximator neural network have significant differences in their architecture, training approach, and complexity. 
#### Improvements:
#### 1. Network Architecture
* The network has five hidden layers, each with 100 neurons.
* Each hidden layer is followed by a batch normalization layer.
* The activation function used is ReLU.
* The model structure is more complex: input -> hidden layer 1 -> batch norm -> hidden layer 2 -> batch norm -> ... -> hidden layer 5 -> batch norm -> output.

#### 2. Initialization of Weights
Applies custom weight initialization using kaiming_uniform_ for weights and initializes biases to zero. This is more suitable for networks using ReLU.

#### 3. Training Details
* Uses the AdamW optimizer with a learning rate of 0.001 and weight decay of 1e-5.
* Implements learning rate scheduling using ReduceLROnPlateau, which reduces the learning rate when the loss plateaus.
* Applies gradient clipping with a max norm of 1.0 to prevent exploding gradients.

#### 4. Complexity
* More complex, likely more capable of capturing the intricacies of the sine function due to deeper layers and regularization techniques (batch normalization, weight decay, gradient clipping).
* The network is also more resistant to overfitting and has mechanisms to adjust the learning rate dynamically.

#### 5. Model Training and Performance Monitoring
* The training process is the same, but with more epochs (10,000 vs. 5,000). 
* Learning rate adjustments and gradient clipping are added to handle the more complex network better.

#### 6. Dataset
Uses 100,000 samples for training, which provides more data for learning the sine function.
