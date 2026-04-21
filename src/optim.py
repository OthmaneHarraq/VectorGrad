import numpy as np
from engine import Tensor


def cross_entropy(y_pred, y_true, num_classes):
    """
    Cross-entropy loss between predicted probabilities and true class indices.

    Args:
        y_pred: Tensor of shape (batch, num_classes)
        y_true: NumPy array of integer class indices with shape (batch,)
        num_classes: Total number of classes

    Returns Scalar Tensor representing the mean cross-entropy loss.
    """
    Y_onehot = Tensor(np.eye(num_classes)[y_true])
    return -(Y_onehot * y_pred.log()).mean() * num_classes


class Adam:
    """
    AdamW optimizer with decoupled L2 weight decay.

    Args:
        params: List of Tensor parameters to optimize
        lr: Learning rate (default: 0.001)
        beta1: First moment decay (default: 0.9)
        beta2: Second moment decay (default: 0.999)
        eps: Numerical stability constant (default: 1e-8)
        weight_decay: L2 regularization coefficient (default: 0.001)
    """

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.001):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {id(p): np.zeros_like(p.data) for p in params}
        self.v = {id(p): np.zeros_like(p.data) for p in params}

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            pid = id(p)
            self.m[pid] = self.beta1 * self.m[pid] + (1 - self.beta1) * p.grad
            self.v[pid] = self.beta2 * self.v[pid] + (1 - self.beta2) * p.grad ** 2
            m_hat = self.m[pid] / (1 - self.beta1 ** self.t)
            v_hat = self.v[pid] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p.data)

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)
