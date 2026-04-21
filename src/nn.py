import numpy as np
from engine import Tensor, cast_low


class Linear:
    """Fully connected linear layer with He-initialized weights"""

    def __init__(self, nin, nout):
        self.weight = Tensor(np.random.randn(nin, nout) * np.sqrt(2.0 / nin), requires_grad=True)
        self.bias = Tensor(np.zeros((1, nout)), requires_grad=True)

    def __call__(self, x):
        return x @ self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]


class MLP:
    """Standard multi-layer perceptron with ReLU activations and softmax output"""

    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.linears = [Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for i, layer in enumerate(self.linears):
            x = layer(x)
            x = x.softmax() if i == len(self.linears) - 1 else x.ReLU()
        return x

    def parameters(self):
        return [p for layer in self.linears for p in layer.parameters()]


class ResidualBlock:
    """
    Two-layer residual block with bfloat16 casting between layers.
    Applies: ReLU(lin2(ReLU(lin1(x))) + x)
    """

    def __init__(self, dim):
        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, dim)

    def __call__(self, x):
        skip = x
        x.data = cast_low(x.data)
        h = self.lin1(x).ReLU()
        h.data = cast_low(h.data)
        h = self.lin2(h)
        return (h + skip).ReLU()

    def parameters(self):
        return self.lin1.parameters() + self.lin2.parameters()


class ResidualMLP:
    """
    Residual MLP with an input projection, stacked residual blocks,
    and an output projection with softmax.
    """

    def __init__(self, nin, hidden_dim, n_blocks, nout):
        self.input_projection = Linear(nin, hidden_dim)
        self.blocks = [ResidualBlock(hidden_dim) for _ in range(n_blocks)]
        self.output_projection = Linear(hidden_dim, nout)

    def __call__(self, x):
        x = self.input_projection(x).ReLU()
        for block in self.blocks:
            x = block(x)
        return self.output_projection(x).softmax()

    def parameters(self):
        params = self.input_projection.parameters()
        for block in self.blocks:
            params += block.parameters()
        params += self.output_projection.parameters()
        return params
