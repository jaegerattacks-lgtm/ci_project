import numpy as np

class MSE:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true)**2)

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.size

class CrossEntropy:
    def forward(self, y_pred, y_true):
        # y_true: one-hot encoded
        self.y_pred = np.clip(y_pred, 1e-12, 1-1e-12)
        self.y_true = y_true
        return -np.mean(np.sum(y_true * np.log(self.y_pred), axis=1))

    def backward(self):
        return -(self.y_true / self.y_pred) / self.y_true.shape[0]