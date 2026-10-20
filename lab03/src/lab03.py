import json
import random
import statistics
import time
from copy import deepcopy
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from thop import profile
from torch.utils.data import DataLoader, random_split
from torchinfo import summary

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "lab03" / "results"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
NUM_THREADS = 4
DEV = "cpu"

def set_seed(s: int = SEED) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
set_seed()
torch.set_num_threads(NUM_THREADS)

tf = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
full = torchvision.datasets.FashionMNIST(ROOT / "data", train=True, download=True, transform=tf)
test = torchvision.datasets.FashionMNIST(ROOT / "data", train=False, download=True, transform=tf)
train, val = random_split(full, [55000, 5000], generator=torch.Generator().manual_seed(SEED))
train_dl = DataLoader(train, batch_size=128, shuffle=True)
val_dl = DataLoader(val, batch_size=128)
test_dl = DataLoader(test, batch_size=128)


class AttnNet(nn.Module):
    """28x28 image -> 16 patches of 7x7 -> linear(49->64) -> 2 encoder layers -> mean pool -> linear(64->10)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(49, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        x = x.unfold(2, 7, 7).unfold(3, 7, 7)
        x = x.reshape(x.size(0), 16, 49)
        x = self.enc(self.proj(x))
        return self.head(x.mean(1))

def make_model(name: str) -> nn.Module:
    if name == "MLP":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                             nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    if name == "CNN":
        return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Flatten(), nn.Linear(32 * 7 * 7, 10))
    if name == "Attention":
        return AttnNet()
    raise ValueError(name)


HAND_LAYERS = {
    "MLP": {
        "Linear 784->256": 2 * 784 * 256,
        "Linear 256->128": 2 * 256 * 128,
        "Linear 128->10": 2 * 128 * 10,
    },
    "CNN": {
        "Conv 1->16, 28x28": 2 * 3 * 3 * 1 * 16 * 28 * 28,
        "Conv 16->32, 14x14": 2 * 3 * 3 * 16 * 32 * 14 * 14,
        "Linear 1568->10": 2 * 1568 * 10,
    },
    "Attention": {
        "Patch projection 49->64": 2 * 16 * 49 * 64,
        "QKV projections (x2 layers)": 2 * (2 * 16 * 64 * 3 * 64),
        "Scores QK^T (x2 layers)": 2 * (2 * 16 ** 2 * 64),
        "Attn x V (x2 layers)": 2 * (2 * 16 ** 2 * 64),
        "Output projection (x2 layers)": 2 * (2 * 16 * 64 * 64),
        "FFN 64->128->64 (x2 layers)": 2 * (2 * 16 * 64 * 128 * 2),
        "Head 64->10": 2 * 64 * 10,
    },
}
HAND_FLOPS = {k: sum(v.values()) for k, v in HAND_LAYERS.items()}


@torch.no_grad()
def latency_ms(model: nn.Module, batch_size: int, runs: int) -> float:
    """Median latency in ms: 10 warm-up forwards, then `runs` timed forwards."""
    model = model.to(DEV).eval()
    x = torch.randn(batch_size, 1, 28, 28, device=DEV)
    for _ in range(10):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def evaluate(model: nn.Module, dl: DataLoader) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(dl.dataset)


def train_model(model: nn.Module, epochs: int = 5) -> float:
    model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep}: val acc = {evaluate(model, val_dl):.4f}")
    return evaluate(model, test_dl)

results = {}
for name in ["MLP", "CNN", "Attention"]:
    print(f"\n===== {name} =====")
    set_seed()
    model = make_model(name)

    info = summary(model, input_size=(1, 1, 28, 28), verbose=0)
    print(info)
    (OUT / f"summary_{name}.txt").write_text(str(info), encoding="utf-8")
    n_params = sum(p.numel() for p in model.parameters())

    macs, _ = profile(deepcopy(model), inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
    lat1 = latency_ms(deepcopy(model), 1, 100)
    lat128 = latency_ms(deepcopy(model), 128, 20)
    print(f"params={n_params} hand={HAND_FLOPS[name]} thop(2*MACs)={int(2 * macs)} "
          f"lat_b1={lat1:.3f}ms lat_b128={lat128:.3f}ms")

    set_seed()
    acc = train_model(model)
    print(f"  TEST acc = {acc:.4f}")

    results[name] = dict(params=n_params, hand_flops=HAND_FLOPS[name],
                         thop_flops=int(2 * macs), latency_b1_ms=lat1,
                         latency_b128_ms=lat128, test_acc=acc,
                         hand_layers=HAND_LAYERS[name])

with open(OUT / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)


names = list(results)

plt.figure(figsize=(6, 4))
for n in names:
    plt.scatter(results[n]["params"], results[n]["test_acc"], s=90, label=n)
plt.xlabel("Parameters")
plt.ylabel("Test accuracy")
plt.title("Accuracy vs parameters")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "acc_vs_params.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
for n in names:
    plt.scatter(results[n]["latency_b1_ms"], results[n]["test_acc"], marker="o", s=90, label=f"{n} (batch 1)")
    plt.scatter(results[n]["latency_b128_ms"], results[n]["test_acc"], marker="s", s=90, label=f"{n} (batch 128)")
plt.xscale("log")
plt.xlabel("Median latency, ms (log scale)")
plt.ylabel("Test accuracy")
plt.title("Accuracy vs latency")
plt.grid(True)
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUT / "acc_vs_latency.png", dpi=150)
plt.close()

print("\nDone. Outputs saved in", OUT)