class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dA):
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
        return dA

    def predict(self, X):
        return self.forward(X)
