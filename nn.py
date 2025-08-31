import numpy as np
import math
import random
from engine import Value

class Neuron:
    def __init__(self, n_inputs):
        """
        A single artificial neuron

        Args:
            n_inputs: Number of input features to this neuron
        """

        self.weights = np.array([Value(random.uniform(-1, 1)) for _ in range(n_inputs)])
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        """
        Forward pass: w*x + b followed by tanh activation
        Args:
            x: Input values (1D or 2D numpy array)
        Returns:
            tanh activation of the weighted sum
        """
        if isinstance(x, np.ndarray) and x.ndim == 1:
            vals = [xi if isinstance(xi, Value) else Value(float(xi)) for xi in x]
            out = sum((wi * xi for wi, xi in zip(self.weights, vals)), self.bias)
            return out.tanh()
        elif isinstance(x, np.ndarray) and x.ndim == 2:
            outputs = []
            for row in x:
                vals = [xi if isinstance(xi, Value) else Value(float(xi)) for xi in row]
                out = sum((wi * xi for wi, xi in zip(self.weights, vals)), self.bias)
                outputs.append(out.tanh())
            return np.array(outputs, dtype=object)
        else:
            raise ValueError("Input must be 1D or 2D numpy array")
    
    def parameters(self):
        """Return all trainable parameters (weights and bias) for this neuron"""
        return list(self.weights) + [self.bias]
    
class Layer:
    def __init__(self, n_inputs, n_neurons):
        """
        A layer containing multiple neurons

        Args:
            n_imputs: Number of inputs to each neuron in this layer
            n_neurons: Number of neurons in this layer
        """
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def __call__(self, x):
        """
        Forward pass through all neurons in the layer
        
        Args:
            x: Input to the layer
        Returns:
            Output from all neurons in the layer
        """

        outs = [neuron(x) for neuron in self.neurons]
        if isinstance(x, np.ndarray) and x.ndim == 1:
            return np.array(outs, dtype=object)
        else:
            return np.array(outs).T
    
    def parameters(self):
        """Returns all parameters from all neurons in this layer"""
        params = []
        
        for neuron in self.neurons:
            p = neuron.parameters()
            params.extend(p)

        return params

class MLP:
    def __init__(self, n_inputs, n_outs):
        """
        Create network architecture:
        [input_size, hidden_layer1_size, hidden_layer2_size,..., output_size]
        Args:
            n_inputs: Number of input features
            n_outs: List of neuron counts for each layer [hidden1, hidden2,..., output]
        """
        sizes = [n_inputs] + n_outs
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(n_outs))]

    def __call__(self, x):
        """
        Forward pass through the entire network
        Args:
            x: Input data
        Returns:
            Network output after passing through all layers
        """

        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        """Returns all parameters from all layers in the network"""

        params = []
        for layer in self.layers:
            params.extend(layer.parameters())

        return params
