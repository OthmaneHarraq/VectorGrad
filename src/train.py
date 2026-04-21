import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from engine import Tensor
from nn import ResidualMLP
from optim import Adam, cross_entropy


# ── Configuration ─────────────────────────────────────────────────────────────

NUM_CLASSES  = 25      # ASL labels 0–24 (J=9 and Z=25 excluded)
HIDDEN_DIM   = 128
N_BLOCKS     = 3
EPOCHS       = 10

LEARNING_RATES = [0.0005, 0.001, 0.003]
BATCH_SIZES    = [16, 64, 128]

TRAIN_CSV = "data/sign_mnist_train.csv"
TEST_CSV  = "data/sign_mnist_test.csv"


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_data(train_path=TRAIN_CSV, test_path=TEST_CSV):
    """
    Load and preprocess the ASL Sign Language MNIST dataset.

    Returns train/val/test splits as NumPy arrays.
    """
    print("Loading Sign Language MNIST...")
    df = pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)], ignore_index=True)

    X = df.drop("label", axis=1).values.astype(np.float64) / 255.0
    y = df["label"].values.astype(int)

    print(f"  Total samples : {X.shape[0]}")
    print(f"  Features      : {X.shape[1]}")
    print(f"  Classes       : {sorted(set(y))}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=42, test_size=0.3)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, random_state=42, test_size=0.5)

    print(f"\n  Train : {X_train.shape}, Val : {X_val.shape}, Test : {X_test.shape}\n")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Training & Evaluation ──────────────────────────────────────────────────────

def train(X_train, X_val, y_train, y_val, learning_rate, batch_size, epochs=EPOCHS):
    """Train a ResidualMLP and return the model and loss/accuracy history."""
    print(f"Training  lr={learning_rate}  batch_size={batch_size}")

    model     = ResidualMLP(nin=784, hidden_dim=HIDDEN_DIM, n_blocks=N_BLOCKS, nout=NUM_CLASSES)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

    train_losses, val_accuracies = [], []

    for epoch in range(epochs):
        total_loss, num_batches = 0.0, 0

        for i in tqdm(range(0, len(X_train), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch = Tensor(X_train[i:i + batch_size])
            y_batch = y_train[i:i + batch_size]

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss   = cross_entropy(y_pred, y_batch, NUM_CLASSES)
            loss.backward()
            optimizer.step()

            total_loss  += loss.data
            num_batches += 1

        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)

        # Validation
        correct = 0
        for i in range(0, len(X_val), batch_size):
            X_batch = Tensor(X_val[i:i + batch_size])
            preds   = np.argmax(model(X_batch).data, axis=1)
            correct += (preds == y_val[i:i + batch_size]).sum()

        val_acc = correct / len(X_val)
        val_accuracies.append(val_acc)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}  val_acc={val_acc*100:.2f}%")

    return model, train_losses, val_accuracies


def evaluate(model, X_test, y_test, batch_size):
    """Evaluate model on the held-out test set."""
    correct = 0
    for i in range(0, len(X_test), batch_size):
        X_batch = Tensor(X_test[i:i + batch_size])
        preds   = np.argmax(model(X_batch).data, axis=1)
        correct += (preds == y_test[i:i + batch_size]).sum()
    acc = correct / len(X_test)
    print(f"  Test accuracy: {acc*100:.2f}%")
    return acc


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(results, learning_rates, batch_sizes):
    """Plot validation accuracy curves grouped by learning rate."""
    fig, axes = plt.subplots(1, len(learning_rates), figsize=(15, 4))
    for ax, lr in zip(axes, learning_rates):
        ax.set_title(f"LR = {lr}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Accuracy")
        for r in results:
            if r["lr"] == lr:
                ax.plot(r["val_accs"], label=f"batch={r['batch']}")
        ax.legend()
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    results      = []
    best_acc     = 0
    best_lr      = None
    best_batch   = None

    for lr in LEARNING_RATES:
        for batch in BATCH_SIZES:
            model, losses, accs = train(X_train, X_val, y_train, y_val, lr, batch)
            acc = evaluate(model, X_test, y_test, batch)

            results.append({"lr": lr, "batch": batch, "losses": losses, "val_accs": accs, "test_acc": acc})

            if acc > best_acc:
                best_acc, best_lr, best_batch = acc, lr, batch

    print(f"\nBest test accuracy: {best_acc*100:.2f}%  (lr={best_lr}, batch={best_batch})")
    plot_results(results, LEARNING_RATES, BATCH_SIZES)
