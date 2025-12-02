import numpy as np
from activations import ReLU, Sigmoid, Tanh, Softmax  # import activations

class Dense:
    def __init__(self, input_size, output_size, activation=None):
        """
        Fully connected (dense) layer.
        input_size: number of input features
        output_size: number of neurons in this layer
        activation: an activation class from activations.py
        """
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, X):
        """
        Forward pass: compute output of this layer
        X: input data (batch_size, input_size)
        Returns: activated output
        """
        self.X = X                  # store input for backprop
        self.Z = X @ self.W + self.b  # linear step
        if self.activation:
            self.A = self.activation.forward(self.Z)  # apply activation
            return self.A
        return self.Z

    def backward(self, dA):
        """
        Backward pass: compute gradients w.r.t weights and input
        dA: gradient of loss w.r.t output of this layer
        Returns: gradient w.r.t input (to propagate to previous layer)
        """
        if self.activation:
            dZ = self.activation.backward(dA, self.Z)
        else:
            dZ = dA

        # Gradients w.r.t weights and biases
        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)

        # Gradient w.r.t input for previous layer
        dX = dZ @ self.W.T
        return dX
