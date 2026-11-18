# Neural Networks in 100 Lines of Pure Python

A from-scratch implementation of a small neural network library using only NumPy. The notebook builds the core pieces of a deep learning framework (parameters, layers, activations, a sequential container, a loss function, an optimizer and a training loop) and shows backpropagation working end to end.

Based on the blog post [Deep Learning in 100 Lines](https://eisenjulian.github.io/deep-learning-in-100-lines/) and the [PyTorch `nn` module tutorial](https://pytorch.org/tutorials/beginner/nn_tutorial.html).

## Project Structure

```
.
├── Neural_Net_in_Python.ipynb
├── image/
│   ├── model_ERM.png
│   └── model.png
└── README.md
```

The notebook displays two diagrams from the `image/` folder (`model_ERM.png` and `model.png`). Keep that folder next to the notebook, or those cells will not render.

## Requirements

- Python 3.6+
- numpy
- matplotlib
- seaborn
- jupyter

```bash
pip install numpy matplotlib seaborn jupyter
```

## Core Idea

During the forward pass, each layer returns two things: its output and a `backward` function. That function is a closure that remembers what it needs from the forward pass. It can only be called after the loss is known, so it computes the gradient later and updates the layer's weight gradients along the way.

Training has three steps:

1. **Forward:** evaluate the model layer by layer.
2. **Backward:** call the saved `backward` functions in reverse order to accumulate gradients.
3. **Update:** let the optimizer change the weights, only at the end, since weights can be reused across layers.

## Components

| Class / function | Purpose |
|------------------|---------|
| `Parameter` | Holds a tensor and its gradient (initialised to zeros) |
| `Layer` | Base class with `forward`, `build_param` and `update` |
| `SGDOptimizer` | Mini-batch stochastic gradient descent; updates the tensor, then zeroes the gradient |
| `Linear` | Fully connected layer, `X @ W + b`, with Xavier-style initialisation (`1/sqrt(n)`) |
| `ReLu` / `ReLu2` | ReLU activation (lambda version and named-function version) |
| `Sigmoid` | Sigmoid activation |
| `Sequential` | Chains layers and runs their `backward` functions in reverse |
| `mse_loss` | Mean squared error, returning the loss and its gradient |
| `Learner` | Training loop with `fit_batch` and `fit(X, Y, epochs, bs)`, similar to scikit-learn or Keras |

## Experiments

### 1. Linear regression

A single `Linear(10, 1)` layer is trained on synthetic data (`Y = X @ W + B`, 100 samples, 10 features) for 10 epochs with batch size 10 and learning rate 0.05.

- The loss falls from about 21.28 in epoch 1 to about 0.0002 in epoch 10.
- The learned parameters match the true ones: weight error ≈ 0.0034 and bias error ≈ 0.0008.

### 2. Non-linear target

The target is `Y = X[:, 0] * X[:, 1]` with 1,000 samples and 2 features, trained for 50 epochs. Two models are compared:

- **1 layer:** `Linear(2, 1)`, learning rate 0.01
- **2 layer:** `Linear(2, 10)` → `Sigmoid` → `Linear(10, 1)`, learning rate 0.3

The final plot shows the loss per epoch for both models. A single linear layer cannot fit the product of two inputs, while the hidden layer with a sigmoid can.

## Usage

```bash
jupyter notebook Neural_Net_in_Python.ipynb
```

Run the cells in order. The cells marked `#@param` (sliders in Colab) hold the dataset and training settings. In plain Jupyter they work as ordinary variables you can edit.

Random seed `2019` is used for reproducibility.

## Notes

- The plotting cells use the `seaborn-talk` style. In matplotlib 3.6 and later it was renamed, so change it to `seaborn-v0_8-talk`.
- `Linear` initialises weights with `np.random.randn`, so results depend on the global NumPy seed (set with `np.random.seed(SEED)` in the experiment cells).
- Targets `Y` must be 2D, shape `(n, 1)`. A 1D array causes a broadcasting error, so reshape with `np.reshape(Y, (-1, 1))`.
