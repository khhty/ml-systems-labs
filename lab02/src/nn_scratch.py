import random
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---- seeds (вимога методички) ----
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
INIT_STD = 0.1   # N(0, 0.01): variance 0.01 -> std 0.1


# ---- дублює вивід у results/log.txt ----
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


log = open(RESULTS / "log.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log)

# ================= Task 1: dataset =================
digits = load_digits()
X = digits.data / 16.0
y = digits.target
Y = np.eye(10)[y]

X_tr, X_te, Y_tr, Y_te, y_tr, y_te = train_test_split(
    X, Y, y, test_size=0.30, stratify=y, random_state=42)

print("=== Task 1 ===")
print("X_tr:", X_tr.shape, "X_te:", X_te.shape)
print("Y_tr:", Y_tr.shape, "Y_te:", Y_te.shape)
print("Pixel range:", X.min(), "-", X.max())
print("Example label:", y[0], "->", Y[0])


# ================= Task 2: parameters =================
def init_params(W, rng, std=INIT_STD):
    shapes = [(64, W), (W, W), (W, 10)]
    p = {}
    for i, s in enumerate(shapes, start=1):
        p[f"W{i}"] = rng.normal(0.0, std, size=s)
        p[f"b{i}"] = np.zeros(s[1])
    return p


print("\n=== Task 2 ===")
for Wd in (16, 64):
    p = init_params(Wd, np.random.default_rng(0))
    print(f"\nW = {Wd}")
    for k, v in p.items():
        print(f"  {k}: {v.shape}")
    print("  Total parameters:", sum(v.size for v in p.values()))


# ================= Task 3: forward =================
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(X, p):
    z1 = X @ p["W1"] + p["b1"]; a1 = np.maximum(0, z1)
    z2 = a1 @ p["W2"] + p["b2"]; a2 = np.maximum(0, z2)
    z3 = a2 @ p["W3"] + p["b3"]; out = softmax(z3)
    return out, (X, z1, a1, z2, a2)


print("\n=== Task 3 ===")
for Wd in (16, 64):
    p = init_params(Wd, np.random.default_rng(0))
    probs, _ = forward(X_tr[:32], p)
    print(f"W={Wd}: output shape {probs.shape}, "
          f"max|row sum - 1| = {np.abs(probs.sum(axis=1) - 1).max():.2e}")
    assert probs.shape == (32, 10)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
print("Forward pass checks passed")


# ================= Task 4: loss =================
def cross_entropy(probs, Y):
    return -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))


def loss_fn(p, X, Y):
    probs, _ = forward(X, p)
    return cross_entropy(probs, Y)


print("\n=== Task 4 ===")
for Wd in (16, 64):
    p = init_params(Wd, np.random.default_rng(0))
    print(f"W={Wd}: untrained loss = {loss_fn(p, X_tr, Y_tr):.4f}   ln(10) = {np.log(10):.4f}")


# ================= Task 5: backward + gradient check =================
def backward(probs, Y, cache, p):
    X, z1, a1, z2, a2 = cache
    N = X.shape[0]
    dz3 = (probs - Y) / N
    g = {}
    g["W3"] = a2.T @ dz3;  g["b3"] = dz3.sum(axis=0)
    da2 = dz3 @ p["W3"].T
    dz2 = da2 * (z2 > 0)
    g["W2"] = a1.T @ dz2;  g["b2"] = dz2.sum(axis=0)
    da1 = dz2 @ p["W2"].T
    dz1 = da1 * (z1 > 0)
    g["W1"] = X.T @ dz1;   g["b1"] = dz1.sum(axis=0)
    return g


def gradient_check(p, X, Y, n_checks=10, eps=1e-5, seed=123):
    rng = np.random.default_rng(seed)
    probs, cache = forward(X, p)
    g = backward(probs, Y, cache, p)
    names = ["W1", "W2", "W3"]
    passed = 0
    seen = set()
    print(f"{'param':<7}{'index':<11}{'analytic':>14}{'numeric':>14}{'rel.err':>11}")
    while len(seen) < n_checks:
        name = names[rng.integers(len(names))]
        idx = tuple(int(rng.integers(s)) for s in p[name].shape)
        if (name, idx) in seen or g[name][idx] == 0.0:
            continue
        seen.add((name, idx))
        old = p[name][idx]
        p[name][idx] = old + eps; lp = loss_fn(p, X, Y)
        p[name][idx] = old - eps; lm = loss_fn(p, X, Y)
        p[name][idx] = old
        num = (lp - lm) / (2 * eps)
        ana = g[name][idx]
        rel = abs(ana - num) / max(abs(ana) + abs(num), 1e-12)
        ok = rel < 1e-6
        passed += ok
        print(f"{name:<7}{str(idx):<11}{ana:>14.6e}{num:>14.6e}{rel:>11.2e}  {'OK' if ok else 'FAIL'}")
    print(f"\nPassed: {passed}/{n_checks} (need >= 9) -> {'PASS' if passed >= 9 else 'FAIL'}")
    return passed


print("\n=== Task 5: gradient check ===")
p_check = init_params(16, np.random.default_rng(7))
if gradient_check(p_check, X_tr[:16], Y_tr[:16]) < 9:
    print("Gradient check FAILED, stopping.")
    sys.exit(1)


# ================= Task 6: training =================
def train(Wd, lr, epochs=50, bs=32, seed=0):
    rng = np.random.default_rng(seed)
    p = init_params(Wd, rng)
    losses = []
    n = X_tr.shape[0]
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            probs, cache = forward(X_tr[b], p)
            g = backward(probs, Y_tr[b], cache, p)
            for k in p:
                p[k] -= lr * g[k]
        ep_loss = loss_fn(p, X_tr, Y_tr)
        losses.append(ep_loss)
        print(f"  W={Wd} lr={lr} epoch {ep:>2}/{epochs}  loss {ep_loss:.4f}")
    acc = np.mean(np.argmax(forward(X_te, p)[0], axis=1) == y_te)
    return losses, acc


print("\n=== Task 6: training ===")
lrs, widths = [0.001, 0.01, 0.1], [16, 64]
results = {}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

for ax, Wd in zip(axes, widths):
    for lr in lrs:
        losses, acc = train(Wd, lr)
        results[(Wd, lr)] = (losses[-1], acc)
        ax.plot(range(1, 51), losses, label=f"lr={lr}")
    ax.set_title(f"Hidden width W={Wd}")
    ax.set_xlabel("Epoch")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()

axes[0].set_ylabel("Training cross-entropy loss")
plt.tight_layout()
plt.savefig(RESULTS / "loss_curves.png", dpi=150)
plt.close()

print("\nSummary:")
print(f"{'W':<6}{'lr':<8}{'final loss':>12}{'test acc':>10}")
for (Wd, lr), (l, a) in results.items():
    print(f"{Wd:<6}{lr:<8}{l:>12.4f}{a:>10.4f}")


# ================= Task 7: timing =================
def time_passes(Wd, X, Y, runs=100):
    p = init_params(Wd, np.random.default_rng(0))
    for _ in range(10):
        pr, c = forward(X, p); backward(pr, Y, c, p)
    tf, tb = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        pr, c = forward(X, p)
        t1 = time.perf_counter()
        backward(pr, Y, c, p)
        t2 = time.perf_counter()
        tf.append(t1 - t0); tb.append(t2 - t1)
    return np.median(tf) * 1e6, np.median(tb) * 1e6


print("\n=== Task 7: timing (batch 32, median of 100) ===")
print(f"{'W':<5}{'forward, us':>14}{'backward, us':>15}{'b/f':>7}")
for Wd in (16, 64):
    f, b = time_passes(Wd, X_tr[:32], Y_tr[:32])
    print(f"{Wd:<5}{f:>14.1f}{b:>15.1f}{b/f:>7.2f}")

sys.stdout = sys.__stdout__
log.close()