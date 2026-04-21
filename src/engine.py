import math
import numpy as np
from ml_dtypes import bfloat16


def cast_low(x):
    """Cast to bfloat16 and back to float64 for memory efficiency."""
    return x.astype(bfloat16).astype(np.float64)


class Value:
    """
    Scalar-valued autograd engine.
    Tracks operations and supports backward pass via reverse-mode autodiff.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self.grad = 0.0
        self._backwards = None
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backwards():
            self.grad += out.grad
            other.grad += out.grad

        out._backwards = _backwards
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backwards():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backwards = _backwards
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def __pow__(self, other):
        assert isinstance(other, int)
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backwards():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backwards = _backwards
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backwards():
            self.grad += out.data * out.grad

        out._backwards = _backwards
        return out

    def tanh(self):
        x = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(x, (self,), 'tanh')

        def _backwards():
            self.grad += (1 - x ** 2) * out.grad

        out._backwards = _backwards
        return out

    def ReLU(self):
        x = self.data if self.data > 0 else 0
        out = Value(x, (self,), "ReLU")

        def _backwards():
            self.grad += out.grad if self.data > 0 else 0

        out._backwards = _backwards
        return out

    def sigmoid(self):
        x = 1 / (1 + math.exp(-self.data))
        out = Value(x, (self,), "sigmoid")

        def _backwards():
            self.grad += x * (1 - x) * out.grad

        out._backwards = _backwards
        return out

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            if v._backwards:
                v._backwards()


class Tensor:
    """
    Batch-capable tensor autograd engine backed by NumPy.
    Supports broadcasting, matmul, and common activation functions.
    """

    def __init__(self, data, _children=(), _op='', requires_grad=False):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self._prev = set(_children)
        self._op = _op
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+',
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = out.grad
                while g.ndim > self.data.ndim:
                    g = g.sum(axis=0)
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        g = g.sum(axis=i, keepdims=True)
                self.grad += g
            if other.requires_grad:
                g = out.grad
                while g.ndim > other.data.ndim:
                    g = g.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        g = g.sum(axis=i, keepdims=True)
                other.grad += g

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * (-1.0)

    def __sub__(self, other):
        return self + (-other)

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), '@',
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*',
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                g = self.data * out.grad
                while g.ndim > other.data.ndim:
                    g = g.sum(axis=0)
                other.grad += g

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __rmatmul__(self, other):
        return self @ other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data ** other, (self,), '**', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        t = np.exp(self.data)
        out = Tensor(t, (self,), 'exp', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += t * out.grad

        out._backward = _backward
        return out

    def log(self):
        t = np.log(self.data + 1e-15)
        out = Tensor(t, (self,), 'log', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (1 / (self.data + 1e-15)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def ReLU(self):
        mask = (self.data > 0)
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += mask * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        t = 1 / (1 + np.exp(-self.data))
        out = Tensor(t, (self,), 'sigmoid', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += t * (1 - t) * out.grad

        out._backward = _backward
        return out

    def softmax(self):
        e = np.exp(self.data - self.data.max(axis=1, keepdims=True))
        t = e / e.sum(axis=1, keepdims=True)
        out = Tensor(t, (self,), 'softmax', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                dx = t * (out.grad - (out.grad * t).sum(axis=1, keepdims=True))
                self.grad += dx

        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(np.mean(self.data), (self,), 'mean', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (np.ones_like(self.data) / self.data.size) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data)
        for n in reversed(topo):
            n._backward()
