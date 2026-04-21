# Neural Network from Scratch — ASL Sign Language MNIST

A from-scratch implementation of a Residual MLP in pure Python and NumPy — no PyTorch, no TensorFlow, no autograd libraries. Every component is hand-written: the autograd engine, tensor operations, residual blocks, and the AdamW optimizer.

Trained and evaluated on the [ASL Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) dataset (24-class classification, 34,627 samples).

**Best test accuracy: ~98.5%**

---

## What's implemented from scratch

| Component | File |
|---|---|
| Scalar autograd engine | `src/engine.py` → `Value` |
| Batched tensor autograd | `src/engine.py` → `Tensor` |
| Linear layer, MLP, Residual Block | `src/nn.py` |
| AdamW optimizer | `src/optim.py` |
| Cross-entropy loss | `src/optim.py` |
| Training loop & evaluation | `src/train.py` |

---

## Project Structure

```
├── notebook.ipynb        # Full walkthrough with outputs and plots
├── src/
│   ├── engine.py         # Value (scalar) and Tensor (batched) autograd
│   ├── nn.py             # Linear, MLP, ResidualBlock, ResidualMLP
│   ├── optim.py          # Adam optimizer and cross-entropy loss
│   └── train.py          # Data loading, training loop, evaluation
├── data/
│   ├── sign_mnist_train.csv
│   └── sign_mnist_test.csv
└── README.md
```

---

## Architecture

```
Input (784)
    → Linear(784 → 128) + ReLU        # input projection
    → ResidualBlock(128) × 3           # residual stack
    → Linear(128 → 25) + Softmax       # output projection
```

Each `ResidualBlock` applies:
```
output = ReLU(lin2(ReLU(lin1(x))) + x)
```
with bfloat16 casting between layers for reduced numerical noise.

---

## Results

| Learning Rate | Batch Size | Test Accuracy |
|---|---|---|
| 0.0005 | 128 | **98.52%** |
| 0.001  | 128 | 98.21% |
| 0.003  | 128 | 97.19% |
| 0.001  | 64  | 92.80% |
| 0.003  | 16  | 32.99% ❌ |

Best stable configuration: `lr=0.0005, batch_size=128`

---

## Setup

```bash
pip install numpy pandas scikit-learn ml-dtypes tqdm matplotlib
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) and place the CSVs in a `data/` folder, then:

```bash
python src/train.py
```

Or open `notebook.ipynb` for the full walkthrough.

---

## Dataset

The ASL Sign Language MNIST contains 34,627 grayscale 28×28 images of hand gestures representing 24 letters of the American Sign Language alphabet (J and Z are excluded as they require motion).
