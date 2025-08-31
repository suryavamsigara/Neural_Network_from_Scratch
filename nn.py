import math
import random
from engine import Value

class Neuron:
    def __init__(self, nin):
        # Initialize weights and bias with random values between -1 and 1

        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        """
        Forward pass: w*x + b
        x: input values
        returns: tanh activation of the weighted sum
        """
        act = sum((wi * xi for wi, xi in zip(self.weights, x)), [self.bias])
        out = act.tanh()
        return out
    
    def parameters(self):
        """Return all trainable parameters (weights and bias)"""
        return self.weights + [self.bias]
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        """Forward pass through all neurons in the layer"""

        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        params = []
        
        for neuron in self.neurons:
            p = neuron.parameters()
            params.extend(p)

        return params

class MLP:
    def __init__(self, nin, nouts):
        """
        Create network architecture:
        [input_size, hidden_layer1_size, hidden_layer2_size,..., output_size]
        """
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        """Forward pass through the entire multi layer perceptron"""

        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        """Returns all parameters from all layers in the network"""

        params = []
        for layer in self.layers:
            params.extend(layer.parameters())

        return params
