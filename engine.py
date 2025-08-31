import numpy as np
import math
import random

class Value:
    def __init__(self, data, _children=(), operation=''):
        self.data = data
        self.grad = 0.0
        self._children = set(_children)
        self.operation = operation
        self._backward = lambda : None

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out_backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "supports int, float"
        out = Value(self.data ** other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        out = self * other ** -1
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += out.grad * (1 - t**2)
        out._backward = _backward
        return out
    
    def sigmoid(self):
        x = self.data
        s = (1 / (1 + math.exp(-x)))
        out = Value(s, (self, ), 'sigmoid')

        def _backward():
            self.grad += out.grad * (s * (1 - s))
        out._backward = _backward
        return out
    
    def backward(self):
        """
        computation_order contains nodes in forward pass order
        Starts from output: self.grad = 1
        Calls `_backward()` for each node in reverse computation order
        """
        computation_order = []
        visited = set()

        def build_computational_order(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_computational_order(child)
                computation_order.append(node)
        
        build_computational_order(self)

        self.grad = 1.0

        for node in reversed(computation_order):
            node._backward()
