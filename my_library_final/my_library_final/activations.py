import numpy as np

class ReLU:
    def forward(self, Z):
        return np.maximum(0, Z)

    def backward(self, dA, Z):
        return dA * (Z > 0)

class Sigmoid:
    def forward(self, Z):
        return 1 / (1 + np.exp(-Z))

    def backward(self, dA, Z):
        A = self.forward(Z)
        return dA * A * (1 - A)

class Tanh:
    def forward(self, Z):
        return np.tanh(Z)

    def backward(self, dA, Z):
        A = self.forward(Z)
        return dA * (1 - A**2)

class Softmax:
    def forward(self, Z):
        exps = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def backward(self, dA, Z):
        # For cross-entropy + softmax, gradient handled in loss
        return dA
