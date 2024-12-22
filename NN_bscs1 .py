import numpy as np
from data import X
import keras
import torch
import torch.nn as nn

class LayerDense:
    def __init__(self,n_inputs, n_neurons):
        self.weights =0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
        self.output = None

    def forward(self,inputs):
        self.output= np.dot(inputs,self.weights) + self.biases

class ActivationReLU:
    def __init__(self):
        self.output = None

    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class ActivationSoftmax:
    def __init__(self):
        self.output=None

    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities =  exp_values/ np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


dense1 = LayerDense(4,3)
activation1 = ActivationReLU()

dense2 = LayerDense(3,4)
activation2 = ActivationSoftmax()

dense1.forward(X[3])
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output)
