# Neural Network from Scratch — ASL Sign Language MNIST

An implementation of a Residual MLP from-scratch in pure Python and NumPy without the use of ML Libraries, such as PyTorch and TensorFlow. Every component is hand-written: the autograd engine, tensor operations, residual blocks, and the AdamW optimizer.

Trained and evaluated on the [ASL Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) dataset (24-class classification, 34,627 samples).

**Best test accuracy: ~99.95%**

<img width="1514" height="463" alt="image" src="https://github.com/user-attachments/assets/331c7f07-f7f2-4e56-a52d-f1b14d8fc5d8" />

## What's implemented from scratch

- Scalar autograd engine: `src/engine.py` → `Value` 
- Batched tensor autograd: `src/engine.py` → `Tensor`
- Linear layer, MLP, Residual Block: `src/nn.py`
- AdamW optimizer: `src/optim.py`
- Cross-entropy loss: `src/optim.py`
- Training loop & evaluation: `src/train.py`

## Project Structure

├── notebook.ipynb        
├── src/
│   ├── engine.py         
│   ├── nn.py             
│   ├── optim.py          
│   └── train.py            
└── README.md

## Architecture

Input (784)
    → Linear(784 → 128) + ReLU        
    → ResidualBlock(128) × 3        
    → Linear(128 → 25) + Softmax  

Each `ResidualBlock` applies:

output = ReLU(lin2(ReLU(lin1(x))) + x)

with bfloat16 casting between layers for reduced numerical noise.

## Results

| Learning Rate | Batch Size | Test Accuracy |
|---|---|---|
| 0.0005 | 128 | **98.52%** |
| 0.001  | 128 | 98.21% |
| 0.003  | 128 | 97.19% |
| 0.001  | 64  | 92.80% |
| 0.003  | 16  | 32.99% ❌ |

Best stable configuration: `lr=0.0005, batch_size=128`

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
