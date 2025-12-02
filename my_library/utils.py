import numpy as np

def flatten(X):
    return X.reshape(X.shape[0], -1)

def normalize(X):
    return X / 255.0

def one_hot(y, num_classes):
    return np.eye(num_classes)[y]
