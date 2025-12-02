class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layer):
        layer.W -= self.lr * layer.dW
        layer.b -= self.lr * layer.db