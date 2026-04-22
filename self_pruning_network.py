"""
self_pruning_network.py
=======================
Self-Pruning Neural Network on CIFAR-10.

The network learns to prune itself DURING training via learnable gate
parameters.  No post-training pruning — sparsity emerges from the loss.

Key design choices (and why they differ from naïve implementations)
--------------------------------------------------------------------
1. Gate scores initialised at 0.0  → sigmoid(0) = 0.5
   Starting at the decision boundary lets the sparsity penalty push
   gates downward from the very first step.  Starting at +1.0 (sigmoid
   ≈ 0.73) means the network has to fight its own initialisation.

2. Sparsity loss = mean gate value (normalised to [0,1])
   This makes λ directly comparable to the CE loss magnitude (~1.0–2.0).
   A raw SUM of 9 M gates would dwarf any CE signal.

3. λ warmup (0 → λ_target over first 10 epochs)
   Prevents the sparsity term from destroying representations before the
   CE loss can form useful features.  Without warmup, high λ collapses
   accuracy on epoch 1.

4. Entropy regulariser  β * mean( g*(1-g) )  added to the loss
   g*(1-g) is maximised at g=0.5 and zero at g=0 or g=1.
   *Minimising* this term pushes gates AWAY from 0.5 toward 0 or 1 —
   exactly the bimodal distribution we want.

5. Architecture: 3072→2048→1024→512→256→10 with BN+ReLU+Dropout
   Dropout prevents any single weight path from being indispensable,
   creating headroom for the gate mechanism to be selective.

Usage
-----
    python self_pruning_network.py          # full run
    python self_pruning_network.py --epochs 10  # quick smoke-test
"""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")          # headless — works without a display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Part 1 — PrunableLinear Layer
# ===========================================================================

class PrunableLinear(nn.Module):
    """
    A linear layer where every weight has a paired learnable gate.

    Forward computation
    -------------------
        gates         = sigmoid(gate_scores)      # ∈ (0, 1) per weight
        pruned_weight = weight ⊙ gates            # element-wise mask
        output        = x @ pruned_weight.T + bias

    Gradients flow through BOTH weight and gate_scores — the optimizer
    can simultaneously refine the weight values AND decide whether each
    weight should be active at all.

    Gate initialisation
    -------------------
    gate_scores is initialised to 0.0 + small noise.
    sigmoid(0) = 0.5 — the decision boundary.
    The sparsity loss immediately starts pushing unused gates toward 0,
    while the CE loss holds useful gates open.  This is the fastest path
    to a bimodal distribution.

    Contrast with init at +1.0 (sigmoid ≈ 0.73): gates start "mostly
    open" and the penalty must fight the initialisation, leading to slow
    polarisation and the "stuck at 0.65" failure mode.

    Parameters
    ----------
    in_features  : int
    out_features : int
    bias         : bool  (default True)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight — He/Kaiming uniform initialisation
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=0.01)

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # Gate scores — init near 0 so sigmoid(score) ≈ 0.5
        # Small noise breaks symmetry so individual gates can differentiate.
        # DO NOT initialise at +1.0 — that shifts gates to 0.73 and makes
        # them very hard to push below the 0.1 threshold.
        self.gate_scores = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01  # tiny noise around 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply gated linear transform.

        No torch.no_grad() here — gradients must flow through gates so
        the sparsity loss can update gate_scores via backprop.
        """
        # Differentiable gate values in (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Element-wise multiplication: zero-out pruned weights
        pruned_weight = self.weight * gates

        return F.linear(x, pruned_weight, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ===========================================================================
# Part 2 — Network Architecture
# ===========================================================================

class SelfPruningNet(nn.Module):
    """
    Feed-forward CIFAR-10 classifier using only PrunableLinear layers.

    Architecture
    ------------
    3072 → 2048 → BN → ReLU → Dropout(0.3)
         → 1024 → BN → ReLU → Dropout(0.3)
         →  512 → BN → ReLU → Dropout(0.2)
         →  256 → BN → ReLU
         →   10  (logits)

    Why this depth?
    A shallower network (e.g. 3072→1024→512→256→10) plateaus at ~60%
    accuracy. At that level almost every weight is mildly useful — no
    clear winners/losers — so gates stay near 0.5 and never polarise.
    A deeper, more capable network has redundant capacity, letting the
    gate mechanism identify and prune the truly unnecessary weights.

    Why Dropout?
    Dropout prevents any single weight from becoming indispensable (the
    network must spread information across paths). Combined with the gate
    sparsity loss, this creates strong pressure for bimodal gate values.

    Parameters
    ----------
    dropout_rate : float  (default 0.3)
    """

    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()

        self.fc1   = PrunableLinear(3072, 2048)
        self.bn1   = nn.BatchNorm1d(2048)
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2   = PrunableLinear(2048, 1024)
        self.bn2   = nn.BatchNorm1d(1024)
        self.drop2 = nn.Dropout(dropout_rate)

        self.fc3   = PrunableLinear(1024, 512)
        self.bn3   = nn.BatchNorm1d(512)
        self.drop3 = nn.Dropout(dropout_rate * 0.67)   # 0.2

        self.fc4   = PrunableLinear(512, 256)
        self.bn4   = nn.BatchNorm1d(256)

        self.fc5   = PrunableLinear(256, 10)            # output logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)                       # flatten 32×32×3 → 3072
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))
        return self.fc5(x)


# ===========================================================================
# Part 3 — Sparsity Loss (and Entropy Regulariser)
# ===========================================================================

def compute_sparsity_loss(model: nn.Module) -> torch.Tensor:
    """
    Normalised L1 penalty on gate values.

    Returns the MEAN gate value across all PrunableLinear layers.
    Value is in (0, 1):
      - Near 1.0 → all gates fully open  (no pruning)
      - Near 0.0 → all gates closed      (fully pruned)

    Using MEAN (not SUM) keeps the magnitude independent of model size,
    so λ is directly comparable to the cross-entropy loss (~1.0–2.0).

    Parameters
    ----------
    model : nn.Module  containing PrunableLinear layers

    Returns
    -------
    torch.Tensor  scalar sparsity loss ∈ (0, 1)
    """
    gate_sum   = torch.zeros(1, device=next(model.parameters()).device)
    gate_count = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates      = torch.sigmoid(module.gate_scores)
            gate_sum   = gate_sum + gates.sum()
            gate_count += gates.numel()

    return gate_sum / gate_count   # mean ∈ (0, 1)


def compute_entropy_reg(model: nn.Module) -> torch.Tensor:
    """
    Bimodality regulariser: mean of  gate * (1 - gate).

    g*(1-g) is maximised at g=0.5 and equals zero at g=0 or g=1.
    MINIMISING this term drives gates toward 0 or 1 — the bimodal
    distribution we want.  This is the key ingredient missing from most
    naïve implementations that get stuck at g≈0.5–0.7.

    The coefficient β is typically 0.1–0.5 times λ.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    torch.Tensor  scalar ∈ (0, 0.25)  (max is 0.25 when all gates = 0.5)
    """
    ent_sum   = torch.zeros(1, device=next(model.parameters()).device)
    gate_count = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates     = torch.sigmoid(module.gate_scores)
            ent_sum   = ent_sum + (gates * (1.0 - gates)).sum()
            gate_count += gates.numel()

    return ent_sum / gate_count   # mean ∈ (0, 0.25)


# ===========================================================================
# Part 4 — Data Loading
# ===========================================================================

def get_dataloaders(batch_size: int = 256, data_root: str = "./data"):
    """
    CIFAR-10 train/test loaders with standard augmentation.

    Augmentation choices
    --------------------
    RandomHorizontalFlip + RandomCrop: standard CIFAR-10 augmentation.
    ColorJitter: adds photometric variation, forces the network to learn
                 shape features rather than colour shortcuts, leading to
                 better CE loss and more selective gates.

    Parameters
    ----------
    batch_size : int   (default 256 — balanced for CPU and GPU)
    data_root  : str

    Returns
    -------
    (train_loader, test_loader)
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root=data_root, train=True,  download=True, transform=train_transform)
    test_ds  = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform)

    # num_workers=0 is safe on all platforms; increase if you have a GPU
    nw = 4 if device.type == "cuda" else 0
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=(device.type == "cuda"))
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=512, shuffle=False,
        num_workers=nw, pin_memory=(device.type == "cuda"))

    return train_loader, test_loader


# ===========================================================================
# Part 5 — Evaluation
# ===========================================================================

def evaluate_model(
    model: nn.Module,
    test_loader,
    threshold: float = 0.1,
    verbose: bool = True,
) -> tuple:
    """
    Compute test accuracy and sparsity level.

    The threshold is ONLY used here at evaluation time.
    Training always uses the continuous sigmoid — hard thresholding
    during training would break the gradient flow.

    Parameters
    ----------
    model       : trained SelfPruningNet
    test_loader : DataLoader
    threshold   : float  gates below this are counted as pruned
    verbose     : bool   print detailed stats

    Returns
    -------
    (test_accuracy_pct, sparsity_pct)
    """
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds   = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    test_acc = 100.0 * correct / total

    # Count pruned weights (evaluation threshold only)
    total_w = pruned_w = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates    = torch.sigmoid(module.gate_scores)
                total_w  += gates.numel()
                pruned_w += (gates < threshold).sum().item()

    sparsity = 100.0 * pruned_w / total_w

    if verbose:
        print(f"  Test Accuracy  : {test_acc:.2f}%")
        print(f"  Sparsity Level : {sparsity:.2f}%  (threshold={threshold})")
        print(f"  Active weights : {total_w - pruned_w:,} / {total_w:,}")
        print(f"  Pruned weights : {pruned_w:,} / {total_w:,}")

    model.train()
    return test_acc, sparsity


# ===========================================================================
# Part 6 — Training Loop
# ===========================================================================

def train_model(
    lambda_val: float,
    train_loader,
    test_loader,
    epochs: int = 50,
    lr: float = 3e-3,
    sparsity_threshold: float = 0.1,
    warmup_epochs: int = 10,
    entropy_beta: float = 0.3,
) -> tuple:
    """
    Train a fresh SelfPruningNet with the given sparsity penalty weight.

    Total loss formula
    ------------------
        effective_λ   = lambda_val * (epoch / warmup_epochs)   [warmup phase]
        sparsity_loss = mean gate value ∈ (0, 1)
        entropy_reg   = mean gate*(1-gate) ∈ (0, 0.25)        [pushes bimodal]
        total_loss    = CE  +  effective_λ * sparsity_loss
                           +  effective_λ * entropy_beta * entropy_reg

    Lambda warmup prevents the sparsity term from collapsing the network
    before the CE loss has established useful features (epochs 1–10).

    The entropy regulariser is the KEY addition: it actively pushes gates
    away from the 0.5 equilibrium toward 0 or 1, producing the bimodal
    histogram required for the assignment.

    Parameters
    ----------
    lambda_val        : sparsity penalty weight
    train_loader      : DataLoader
    test_loader       : DataLoader
    epochs            : int
    lr                : float  initial learning rate
    sparsity_threshold: float  used only in final evaluation
    warmup_epochs     : int    epochs over which λ is ramped from 0 to lambda_val
    entropy_beta      : float  coefficient for entropy reg relative to λ

    Returns
    -------
    (model, test_accuracy, sparsity_level, history_dict)
    """
    print(f"\n{'='*65}")
    print(f"  Training   λ = {lambda_val}   epochs = {epochs}   "
          f"warmup = {warmup_epochs}   entropy_β = {entropy_beta}")
    print(f"{'='*65}")

    model     = SelfPruningNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Label smoothing reduces over-confidence and helps gates discriminate
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {"ce": [], "sp": [], "ent": [], "total": []}

    model.train()
    for epoch in range(1, epochs + 1):
        run_ce = run_sp = run_ent = run_tot = n = 0.0

        # λ warmup: linearly increase from 0 → lambda_val over warmup_epochs
        # After warmup, keep at lambda_val for the rest of training.
        if epoch <= warmup_epochs:
            effective_lambda = lambda_val * (epoch / warmup_epochs)
        else:
            effective_lambda = lambda_val

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits        = model(images)
            ce_loss       = criterion(logits, labels)

            # L1-style: push mean gate value toward 0
            sparsity_loss = compute_sparsity_loss(model)

            # Bimodality: push gates away from 0.5 toward 0 or 1
            entropy_loss  = compute_entropy_reg(model)

            total_loss = (
                ce_loss
                + effective_lambda * sparsity_loss
                + effective_lambda * entropy_beta * entropy_loss
            )

            total_loss.backward()

            # Gradient clipping prevents gate_scores from exploding at high λ
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            run_ce  += ce_loss.item()
            run_sp  += sparsity_loss.item()
            run_ent += entropy_loss.item()
            run_tot += total_loss.item()
            n       += 1

        scheduler.step()

        avg_ce  = run_ce  / n
        avg_sp  = run_sp  / n
        avg_ent = run_ent / n
        avg_tot = run_tot / n
        history["ce"].append(avg_ce)
        history["sp"].append(avg_sp)
        history["ent"].append(avg_ent)
        history["total"].append(avg_tot)

        # Print every 5 epochs to reduce noise, plus epoch 1
        if epoch % 5 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  Epoch [{epoch:02d}/{epochs}]  "
                f"Total={avg_tot:.4f}  CE={avg_ce:.4f}  "
                f"Sparsity={avg_sp:.4f}  Entropy={avg_ent:.4f}  "
                f"eff_λ={effective_lambda:.3f}  LR={lr_now:.2e}"
            )

    print(f"\n  ── Final Evaluation ─────────────────────────────")
    test_acc, sparsity_level = evaluate_model(
        model, test_loader, threshold=sparsity_threshold)

    return model, test_acc, sparsity_level, history


# ===========================================================================
# Visualisation helpers
# ===========================================================================

def plot_gate_distribution(
    model,
    lambda_val,
    sparsity,
    accuracy,
    threshold: float = 0.1,
    save_path: str = "gate_distribution.png",
):
    """
    Two-panel gate value histogram for a single model.

    Left panel  : full range [0,1], log Y scale — shows both the
                  pruned spike near 0 and the active cluster.
    Right panel : active gates only (≥ threshold), linear Y scale —
                  shows the shape of the "surviving" distribution.

    A well-trained model with bimodal gates will show:
      - A tall spike near 0 on the left panel
      - A smooth hill peaking around 0.7–0.9 on the right panel

    Parameters
    ----------
    model     : trained SelfPruningNet
    lambda_val: float  used in plot title
    sparsity  : float  used in plot title
    accuracy  : float  used in plot title
    threshold : float  dividing line between pruned / active
    save_path : str
    """
    all_gates = []
    model.eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                g = torch.sigmoid(module.gate_scores).cpu().numpy().flatten()
                all_gates.append(g)
    all_gates = np.concatenate(all_gates)

    pruned = (all_gates < threshold).sum()
    active = len(all_gates) - pruned
    print(f"\n  Gate stats — λ={lambda_val}")
    print(f"  Gates <  {threshold}: {pruned:,}  ({100*pruned/len(all_gates):.1f}%)")
    print(f"  Gates >= {threshold}: {active:,}  ({100*active/len(all_gates):.1f}%)")
    print(f"  Mean gate: {all_gates.mean():.4f}   Std: {all_gates.std():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Learned Gate Distribution  —  λ={lambda_val}  |  "
        f"Sparsity={sparsity:.1f}%  |  Accuracy={accuracy:.2f}%",
        fontsize=13, fontweight="bold"
    )

    # --- Left: full range, log Y ---
    axes[0].hist(all_gates, bins=100, range=(0, 1),
                 color="steelblue", edgecolor="none")
    axes[0].set_yscale("log")
    axes[0].axvline(x=threshold, color="crimson", linestyle="--",
                    linewidth=2, label=f"Threshold ({threshold})")
    axes[0].set_xlabel("Gate Value  [σ(gate_score)]", fontsize=12)
    axes[0].set_ylabel("Count (log scale)", fontsize=12)
    axes[0].set_title("Full Range — Log Y", fontsize=11)
    axes[0].legend(fontsize=10)

    # --- Right: active gates only, linear Y ---
    active_gates = all_gates[all_gates >= threshold]
    if len(active_gates) > 0:
        axes[1].hist(active_gates, bins=80, color="darkorange", edgecolor="none")
        axes[1].set_xlabel("Gate Value  [σ(gate_score)]", fontsize=12)
        axes[1].set_ylabel("Count", fontsize=12)
        axes[1].set_title(
            f"Active Gates Only (≥{threshold})\n"
            f"{len(active_gates):,} weights  "
            f"({100*len(active_gates)/len(all_gates):.1f}%)",
            fontsize=11
        )
    else:
        axes[1].text(0.5, 0.5, "No active gates", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_all_gate_distributions(
    trained_models, lambdas, accuracies, sparsities,
    threshold: float = 0.1,
    save_path: str = "all_gate_distributions.png",
):
    """
    Side-by-side log-scale gate histograms for all three λ values.

    Useful for comparing how strongly each λ polarises the gates.

    Parameters
    ----------
    trained_models : list of SelfPruningNet
    lambdas        : list of float
    accuracies     : list of float
    sparsities     : list of float
    threshold      : float
    save_path      : str
    """
    colors = ["steelblue", "darkorange", "seagreen"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (model, lam, acc, spar) in enumerate(
            zip(trained_models, lambdas, accuracies, sparsities)):
        all_gates = []
        model.eval()
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, PrunableLinear):
                    g = torch.sigmoid(module.gate_scores).cpu().numpy().flatten()
                    all_gates.append(g)
        all_gates = np.concatenate(all_gates)

        axes[i].hist(all_gates, bins=100, range=(0, 1),
                     color=colors[i], edgecolor="none")
        axes[i].set_yscale("log")
        axes[i].axvline(x=threshold, color="crimson",
                        linestyle="--", linewidth=1.5)
        axes[i].set_title(
            f"λ={lam}\nAcc={acc:.1f}%  Sparsity={spar:.1f}%", fontsize=11)
        axes[i].set_xlabel("Gate Value  [σ(score)]")
        axes[i].set_ylabel("Count (log)" if i == 0 else "")

    fig.suptitle("Gate Distributions Across All Lambda Values",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def plot_training_curves(
    histories, lambdas,
    save_path: str = "training_curves.png",
):
    """
    CE loss and sparsity loss curves for all three models.

    Parameters
    ----------
    histories : list of dict  (keys: "ce", "sp", "ent", "total")
    lambdas   : list of float
    save_path : str
    """
    colors = ["steelblue", "darkorange", "seagreen"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, (hist, lam) in enumerate(zip(histories, lambdas)):
        epochs = range(1, len(hist["ce"]) + 1)
        axes[i].plot(epochs, hist["ce"], label="CE Loss",
                     color=colors[i], linewidth=2)
        axes[i].plot(epochs, hist["sp"], label="Sparsity Loss",
                     color=colors[i], linestyle="--", alpha=0.8, linewidth=1.5)
        axes[i].set_title(f"λ = {lam}", fontsize=12)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Loss")
        axes[i].legend(fontsize=9)
        axes[i].grid(alpha=0.3)

    fig.suptitle("Training Curves — CE Loss vs Sparsity Loss",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def print_results_table(lambdas, accuracies, sparsities):
    """
    Print a formatted summary table of results for all lambda values.

    Parameters
    ----------
    lambdas    : list of float
    accuracies : list of float
    sparsities : list of float
    """
    print("\n")
    print("┌──────────┬───────────────┬────────────────────┐")
    print("│  Lambda  │ Test Accuracy │  Sparsity Level    │")
    print("├──────────┼───────────────┼────────────────────┤")
    for lam, acc, spar in zip(lambdas, accuracies, sparsities):
        print(f"│ {str(lam):<8} │    {acc:6.2f}%     │       {spar:6.2f}%        │")
    print("└──────────┴───────────────┴────────────────────┘")


def print_per_layer_stats(model, lambda_val, threshold: float = 0.1):
    """
    Print per-layer breakdown of gate statistics.

    Parameters
    ----------
    model      : trained SelfPruningNet
    lambda_val : float  (for display)
    threshold  : float
    """
    print(f"\nPer-layer gate stats — λ={lambda_val}  (threshold={threshold})\n")
    print(f"{'Layer':<8} {'Shape':<18} {'Mean':>10} {'Min':>8} "
          f"{'Max':>8} {'Pruned%':>10}")
    print("-" * 68)
    model.eval()
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, PrunableLinear):
                g = torch.sigmoid(module.gate_scores)
                pct = (g < threshold).float().mean().item() * 100
                print(
                    f"{name:<8} {str(tuple(g.shape)):<18} "
                    f"{g.mean().item():>10.4f} "
                    f"{g.min().item():>8.4f} "
                    f"{g.max().item():>8.4f} "
                    f"{pct:>9.2f}%"
                )


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    """Parse command-line arguments for quick customisation."""
    p = argparse.ArgumentParser(description="Self-Pruning Neural Network on CIFAR-10")
    p.add_argument("--epochs",     type=int,   default=50,
                   help="Training epochs per model (default: 50)")
    p.add_argument("--batch-size", type=int,   default=256,
                   help="Batch size (default: 256)")
    p.add_argument("--lr",         type=float, default=3e-3,
                   help="Initial learning rate (default: 3e-3)")
    p.add_argument("--lambdas",    type=float, nargs="+",
                   default=[0.1, 0.5, 2.0],
                   help="Lambda values to train (default: 0.1 0.5 2.0)")
    p.add_argument("--threshold",  type=float, default=0.1,
                   help="Gate threshold for sparsity counting (default: 0.1)")
    p.add_argument("--warmup",     type=int,   default=10,
                   help="λ warmup epochs (default: 10)")
    p.add_argument("--entropy-beta", type=float, default=0.3,
                   help="Entropy reg coefficient relative to λ (default: 0.3)")
    p.add_argument("--data-root",  type=str,   default="./data")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # -----------------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------------
    _l = PrunableLinear(16, 8)
    _o = _l(torch.randn(4, 16))
    assert _o.shape == (4, 8), "PrunableLinear output shape mismatch"
    _g = torch.sigmoid(_l.gate_scores)
    print(f"\nGate init mean : {_g.mean().item():.4f}  (expected ≈ 0.50)")
    print(f"Gate init min  : {_g.min().item():.4f}")
    print(f"Gate init max  : {_g.max().item():.4f}")

    _net = SelfPruningNet()
    _total  = sum(p.numel() for p in _net.parameters())
    _gates  = sum(p.numel() for n, p in _net.named_parameters() if "gate" in n)
    print(f"\nModel parameters : {_total:,}  (gates: {_gates:,})")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print("\nLoading CIFAR-10 …")
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, data_root=args.data_root)
    print(f"Train batches: {len(train_loader)}  |  Test batches: {len(test_loader)}")

    # -----------------------------------------------------------------------
    # Train one model per lambda
    # -----------------------------------------------------------------------
    trained_models = []
    accuracies     = []
    sparsities     = []
    histories      = []

    for lam in args.lambdas:
        model, acc, spar, hist = train_model(
            lambda_val         = lam,
            train_loader       = train_loader,
            test_loader        = test_loader,
            epochs             = args.epochs,
            lr                 = args.lr,
            sparsity_threshold = args.threshold,
            warmup_epochs      = args.warmup,
            entropy_beta       = args.entropy_beta,
        )
        trained_models.append(model)
        accuracies.append(acc)
        sparsities.append(spar)
        histories.append(hist)

    # -----------------------------------------------------------------------
    # Results table
    # -----------------------------------------------------------------------
    print_results_table(args.lambdas, accuracies, sparsities)

    # -----------------------------------------------------------------------
    # Per-layer stats
    # -----------------------------------------------------------------------
    for lam, model in zip(args.lambdas, trained_models):
        print_per_layer_stats(model, lam, threshold=args.threshold)

    # -----------------------------------------------------------------------
    # Training curves
    # -----------------------------------------------------------------------
    plot_training_curves(histories, args.lambdas, save_path="training_curves.png")

    # -----------------------------------------------------------------------
    # Gate distribution — best model (by accuracy)
    # -----------------------------------------------------------------------
    best_idx = int(np.argmax(accuracies))
    print(f"\nBest model: λ={args.lambdas[best_idx]}  "
          f"(accuracy={accuracies[best_idx]:.2f}%)")
    plot_gate_distribution(
        model      = trained_models[best_idx],
        lambda_val = args.lambdas[best_idx],
        sparsity   = sparsities[best_idx],
        accuracy   = accuracies[best_idx],
        threshold  = args.threshold,
        save_path  = "gate_distribution.png",
    )

    # -----------------------------------------------------------------------
    # All three distributions side by side
    # -----------------------------------------------------------------------
    plot_all_gate_distributions(
        trained_models, args.lambdas, accuracies, sparsities,
        threshold = args.threshold,
        save_path = "all_gate_distributions.png",
    )

    print("\n✓ Done.  Files saved:")
    print("    gate_distribution.png")
    print("    all_gate_distributions.png")
    print("    training_curves.png")
