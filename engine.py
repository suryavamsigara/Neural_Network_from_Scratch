import numpy as np
import math
import random

class Value:
    def __init__(self, data, _children=(), operation=''):
        self.data = data
        self._children = set(_children)
        self.operation = operation

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '+')
        return out

    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "supports int, float"
        out = Value(self.data ** other, (self, ), f'**{other}')
        return out
    
    def __truediv__(self, other):
        out = self * other ** -1
        return out
    
    def __exp__(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        return out