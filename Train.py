#!/usr/bin/env python3
"""Train a multilayer perceptron from scratch (NumPy only).
 
Usage:
    python train.py --layer 24 24 --epochs 70 --batch_size 8 --learning_rate 0.01
    python train.py --optimizer adam --learning_rate 0.001 --patience 10 --name adam
 
Bonus features: optimizers (sgd, momentum, nesterov, rmsprop, adam), early
stopping with best-weights restore, several metrics (accuracy, precision,
recall, F1), and a metrics history saved to a JSON file (see compare.py).
 
This file also holds the network and helpers, which predict.py imports.
"""
import argparse
 
import numpy as np
 
EPS = 1e-15
 
 
# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def read_csv(path):
    """Return (labels, features). Expects: id, diagnosis (M/B), 30 features."""
    import csv
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    if rows[0][1].strip() not in ("M", "B"):  # skip header if present
        rows = rows[1:]
    labels = np.array([r[1].strip() for r in rows])
    x = np.array([[float(v) for v in r[2:]] for r in rows])
    return labels, x
 
 
def one_hot(labels):
    """B -> [1, 0], M -> [0, 1]"""
    y = np.zeros((len(labels), 2))
    y[labels == "B", 0] = 1
    y[labels == "M", 1] = 1
    return y
 
 
# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
 
 
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
 
 
# name -> (activation(z), derivative expressed from the activation output a)
ACTIVATIONS = {
    "sigmoid": (sigmoid, lambda a: a * (1 - a)),
    "tanh": (np.tanh, lambda a: 1 - a ** 2),
    "relu": (lambda z: np.maximum(0, z), lambda a: (a > 0).astype(float)),
}
 
 
def init_weights(fan_in, fan_out, method, rng):
    if method == "heUniform":
        lim = np.sqrt(6.0 / fan_in)
    elif method == "xavierUniform":
        lim = np.sqrt(6.0 / (fan_in + fan_out))
    else:
        raise ValueError(f"unknown initializer {method}")
    return rng.uniform(-lim, lim, size=(fan_in, fan_out))
 
 
class MLP:
    def __init__(self, sizes, activation="sigmoid", init="heUniform", seed=None):
        rng = np.random.default_rng(seed)
        self.sizes, self.activation = sizes, activation
        self.W = [init_weights(a, b, init, rng) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros((1, b)) for b in sizes[1:]]
 
    def forward(self, x):
        """Feedforward: returns the activations of every layer."""
        act = ACTIVATIONS[self.activation][0]
        a = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a[-1] @ W + b
            a.append(softmax(z) if i == len(self.W) - 1 else act(z))
        return a
 
    def predict_proba(self, x):
        return self.forward(x)[-1]
 
    def backward(self, a, y):
        """Backpropagation: returns gradients (dW, db) for every layer."""
        d_act = ACTIVATIONS[self.activation][1]
        delta = (a[-1] - y) / y.shape[0]  # softmax + cross-entropy shortcut
        dW, db = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            dW[i] = a[i].T @ delta
            db[i] = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.W[i].T) * d_act(a[i])
        return dW, db
    
    def params(self):
        """Return a list of all parameters (weights and biases).
        All trainable arrays (updated in place by the optimizer)"""
        return self.W + self.b
 
    def step(self, dW, db, optimizer):
        """Update weights and biases with the given gradients and optimizer."""
        optimizer.update(self.params(), dW + db)
 
# --------------------------------------------------------------------------- #
# Optimizers (bonus)
# --------------------------------------------------------------------------- #
class Optimizer:
    """sgd is plain gradient descent: p <- p - lr * grad."""
    def __init__(self, name, lr, mu=0.9, beta1=0.9, beta2=0.999, rho=0.9, eps=1e-8):
        self.name, self.lr = name, lr
        self.mu, self.beta1, self.beta2, self.rho, self.eps = mu, beta1, beta2, rho, eps
        self.t = 0
        self.states = None  # for momentum, rmsprop, adam

    def update(self, params, grads):
        if self.name == "sgd":
            for p, g in zip(params, grads):
                p -= self.lr * g
        else:
            if self.states is None:
                self.states = [np.zeros_like(p) for p in params]  # momentum / rmsprop / adam
                if self.name == "adam":
                    self.m = [np.zeros_like(p) for p in params]
                    self.v = [np.zeros_like(p) for p in params]
            self.t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                if self.name == "momentum":
                    self.states[i] = self.mu * self.states[i] - self.lr * g
                    p += self.states[i]
                elif self.name == "nesterov":
                    v_prev = self.states[i].copy()
                    self.states[i] = self.mu * self.states[i] - self.lr * g
                    p += -self.mu * v_prev + (1 + self.mu) * self.states[i]
                elif self.name == "rmsprop":
                    self.states[i] = self.rho * self.states[i] + (1 - self.rho) * g ** 2
                    p -= (self.lr / (np.sqrt(self.states[i]) + self.eps)) * g
                elif self.name == "adam":
                    # update biased first moment estimate
                    self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                    # update biased second raw moment estimate
                    self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
                    # compute bias-corrected first moment estimate
                    m_hat = self.m[i] / (1 - (self.beta1 ** self.t))
                    # compute bias-corrected second raw moment estimate
                    v_hat = self.v[i] / (1 - (self.beta2 ** self.t))
                    p -= (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat

OPTIMIZERS = ["sgd", "momentum", "nesterov", "rmsprop", "adam"]

# --------------------------------------------------------------------------- #
# Losses / metrics
# --------------------------------------------------------------------------- #
def categorical_crossentropy(p, y):
    return float(-np.mean(np.sum(y * np.log(np.clip(p, EPS, 1)), axis=1)))
 
 
def binary_crossentropy(p_malignant, y_malignant):
    p = np.clip(p_malignant, EPS, 1 - EPS)
    return float(-np.mean(y_malignant * np.log(p) + (1 - y_malignant) * np.log(1 - p)))
 
 
def accuracy(p, y):
    return float(np.mean(p.argmax(axis=1) == y.argmax(axis=1)))
 
def classification_metrics(p, y):
    """Accuracy, precision, recall and F1 with M (malignant) as positive class."""
    pred, true = p[:, 1] > 0.5, y[:, 1] == 1
    tp = int(np.sum(pred & true))
    tn = int(np.sum(~pred & ~true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / max(len(y), 1), "precision": precision,
            "recall": recall, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

LOSSES = {
    "categoricalCrossentropy": categorical_crossentropy,
    "binaryCrossentropy": lambda p, y: binary_crossentropy(p[:, 1], y[:, 1]),
}
 
 
# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def plot_curves(hist):
    import matplotlib
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(hist["loss"], label="training loss")
    ax1.plot(hist["val_loss"], label="validation loss")
    ax1.set(title="Loss", xlabel="epoch", ylabel="loss")
    ax2.plot(hist["acc"], label="training acc")
    ax2.plot(hist["val_acc"], label="validation acc")
    ax2.set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    ax3.plot(hist["val_precision"], label="precision")
    ax3.plot(hist["val_recall"], label="recall")
    ax3.plot(hist["val_f1"], label="F1 score")
    ax3.set(title="Metrics", xlabel="epoch", ylabel="value", ylim=(0, 1))
    for ax in (ax1, ax2, ax3):
        ax.legend()
    fig.tight_layout()
    fig.savefig("learning_curves.png", dpi=120)
    print("> learning curves saved to learning_curves.png")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data_training.csv")
    ap.add_argument("--valid", default="data_validation.csv")
    ap.add_argument("--layer", type=int, nargs="+", default=[24, 24],
                    help="hidden layer sizes (two by default)")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--loss", choices=LOSSES, default="categoricalCrossentropy")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=0.01)
    ap.add_argument("--activation", choices=ACTIVATIONS, default="sigmoid")
    ap.add_argument("--init", choices=["heUniform", "xavierUniform"], default="heUniform")
    ap.add_argument("--optimizer", choices=OPTIMIZERS, default="sgd")
    ap.add_argument("--patience", type=int, default=0, help="early stopping (0 = off)")
    ap.add_argument("--history", default=None, help="save metrics history to .json file")
    ap.add_argument("--name", default=None, help="run lable used by compare.py")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save", default="./saved_model.npy")
    args = ap.parse_args()
 
    lab_t, x_t = read_csv(args.train)
    lab_v, x_v = read_csv(args.valid)
    y_t, y_v = one_hot(lab_t), one_hot(lab_v)
 
    # preprocessing: standardise with training-set statistics only
    mean, std = x_t.mean(axis=0), x_t.std(axis=0)
    std[std == 0] = 1.0
    x_t, x_v = (x_t - mean) / std, (x_v - mean) / std
 
    print(f"x_train shape : {x_t.shape}")
    print(f"x_valid shape : {x_v.shape}")
 
    sizes = [x_t.shape[1]] + args.layer + [2]
    net = MLP(sizes, args.activation, args.init, args.seed)
    loss_fn = LOSSES[args.loss]
    rng = np.random.default_rng(args.seed)

    keys = ["loss", "val_loss", "acc", "val_acc", "val_precision", "val_recall", "val_f1"]
    hist = {"k: [] for k in keys}
    best, wait, best_weights = np.inf, 0, None
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x_t))
        for s in range(0, len(order), args.batch_size):
            idx = order[s:s + args.batch_size]
            acts = net.forward(x_t[idx])
            dW, db = net.backward(acts, y_t[idx])
            net.step(dW, db, args.learning_rate)
 
        p_t, p_v = net.predict_proba(x_t), net.predict_proba(x_v)
        hist["loss"].append(loss_fn(p_t, y_t))
        hist["val_loss"].append(loss_fn(p_v, y_v))
        hist["acc"].append(accuracy(p_t, y_t))
        hist["val_acc"].append(mv["accuracy"])
        hist["val_precision"].append(mv["precision"])
        hist["val_recall"].append(mv["recall"])
        hist["val_f1"].append(mv["f1"])
        print(f"epoch {epoch:02d}/{args.epochs} - loss: {hist['loss'][-1]:.4f} "
              f"- val_loss: {hist['val_loss'][-1]:.4f} - acc: {hist['acc'][-1]:.4f} "
              f"- val_acc: {hist['val_acc'][-1]:.4f} - val_precision: {hist['val_precision'][-1]:.4f} "
              f"- val_recall: {hist['val_recall'][-1]:.4f} - val_f1: {hist['val_f1'][-1]:.4f}")
 
        if args.patience:
            if hist["val_loss"][-1] < best - 1e-6:
                best, wait = hist["val_loss"][-1], 0
                best_weights = ([W.copy() for W in net.W], [b.copy() for b in net.b])
            else:
                wait += 1
                if wait >= args.patience:
                    print(f"early stopping at epoch {epoch}(restoring best weights)")
                    break
    if best_weights is not None:
        net.W, net.b = best_weights
 
    model = {"sizes": sizes, "activation": args.activation, "W": net.W, "b": net.b,
             "mean": mean, "std": std}
    print(f"> saving model '{args.save}' to disk...")
    np.save(args.save, model, allow_pickle=True)

    
    name = args.name or f"{args.optimizer} {'-'.join(map(str, args.layer))} lr={args.learning_rate}"
    with open(args.history, "w") as f:
        json.dump({"name": name, "args": vars(args), "history": hist}, f, indent=2)
    print(f"> metrics history saved to {args.history}")
    plot_curves(hist)
 
 
if __name__ == "__main__":
    main()
