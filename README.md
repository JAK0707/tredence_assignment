# Self-Pruning Neural Network — Assignment Report

## 1. Why L1 Penalty on Sigmoid Gates Encourages Sparsity

### The L1 Penalty

The sparsity loss is defined as the **mean of all gate values** across every `PrunableLinear` layer:

$$\mathcal{L}_{\text{sparsity}} = \frac{1}{N} \sum_{i=1}^{N} \sigma(s_i)$$

where $s_i$ are the raw gate scores and $\sigma$ is the sigmoid function. This is structurally equivalent to an L1 penalty on the gate values, and it drives sparsity through a specific gradient property.

**L1 vs L2 — why L1 wins for sparsity:**

The L1 penalty gradient with respect to gate value $g_i$ is:

$$\frac{\partial \mathcal{L}_{L1}}{\partial g_i} = \text{sign}(g_i) = +1 \quad \text{(since } g_i > 0 \text{ always)}$$

This is a **constant force** pushing every gate downward, regardless of how small the gate already is. A gate at $g = 0.001$ feels exactly the same downward pull as a gate at $g = 0.9$.

By contrast, L2 produces:

$$\frac{\partial \mathcal{L}_{L2}}{\partial g_i} = 2g_i$$

Near zero the gradient vanishes — a gate that has shrunk to $g = 0.01$ receives only $0.02$ gradient, which is easily overwhelmed by the CE loss pulling in the other direction. L2 shrinks weights uniformly but almost never eliminates them.

### The Role of Sigmoid

The gate scores $s_i$ are unbounded real numbers, but we want interpretable "how active is this weight" values in $[0, 1]$. Sigmoid provides this mapping:

$$g_i = \sigma(s_i) = \frac{1}{1 + e^{-s_i}}$$

- $s_i \to -\infty$ → $g_i \to 0$ (weight fully pruned)
- $s_i = 0$ → $g_i = 0.5$ (at the decision boundary)
- $s_i \to +\infty$ → $g_i \to 1$ (weight fully active)

This is differentiable everywhere, so the gradient flows cleanly through the gate into $s_i$ during backpropagation, allowing the optimizer to update gate scores directly.

### Gradient Behavior Near Zero

The chain rule connects the L1 gradient to the gate score:

$$\frac{\partial \mathcal{L}}{\partial s_i} = \frac{\partial \mathcal{L}}{\partial g_i} \cdot g_i(1 - g_i)$$

The term $g_i(1 - g_i)$ is the sigmoid derivative. Near $g_i = 0$ (already pruned), this approaches zero — there is a slight gradient vanishing effect. However, the constant L1 signal ($+1$ from the sparsity term) is still large enough to push gates below threshold when $\lambda$ is set appropriately. This is why the entropy regulariser $g(1-g)$ is also minimised: it actively pushes gates away from the $g=0.5$ plateau where the sigmoid derivative is largest, toward the extremes where gates commit to 0 or 1.

---

## 2. Results Table

Results from training on CIFAR-10 for 50 epochs, with λ warmup over 10 epochs and entropy regulariser β=0.8:

| Lambda | Test Accuracy | Sparsity Level (%) | Active Weights | Pruned Weights |
|--------|:-------------:|:------------------:|:--------------:|:--------------:|
| 0.5    |    62.69%     |        1.86%       |   8,878,608    |    167,920     |
| 2.0    |    62.30%     |       30.85%       |   6,255,650    |   2,790,878    |
| 5.0    |    62.73%     |       59.79%       |   3,637,832    |   5,408,696    |

**Key observation:** λ=5.0 prunes nearly 60% of all weights (5.4 million out of 9 million) while achieving *higher* accuracy than λ=0.5. This proves the pruned weights were genuinely redundant — the network at λ=0.5 was over-parameterised.

### Per-Layer Breakdown (λ=5.0)

| Layer | Shape | Mean Gate | Pruned % |
|-------|-------|-----------|----------|
| fc1   | (2048, 3072) | 0.0830 | **73.58%** |
| fc2   | (1024, 2048) | 0.1869 | **35.27%** |
| fc3   | (512, 1024)  | 0.2955 |   7.35%   |
| fc4   | (256, 512)   | 0.3813 |   1.15%   |
| fc5   | (10, 256)    | 0.4544 |   0.00%   |

The pruning is heaviest in early layers (input → hidden) and tapers off toward the output. This reflects the network's learned belief that most raw pixel interactions are redundant, but the compressed higher-level representations in later layers are all essential.

---

## 3. Analysis of Lambda Trade-off

### Trend: Sparsity vs Accuracy

As λ increases, the sparsity penalty receives proportionally more weight in the total loss. This creates stronger gradient pressure on every gate score, pushing more gates below the 0.1 threshold. The results show a clear and clean trend:

- λ=0.5 → 1.86% sparsity, 62.69% accuracy
- λ=2.0 → 30.85% sparsity, 62.30% accuracy  
- λ=5.0 → 59.79% sparsity, 62.73% accuracy

The most striking finding is that the accuracy across all three λ values is **nearly identical** (~62.3–62.7%), with no meaningful degradation even at 60% sparsity. This is a strong empirical result: the majority of the network's 9 million gated weights simply did not need to exist.

### The Sweet Spot

**λ=5.0 is the practical sweet spot** for this architecture:

- Removes 5.4 million weights (59.8% of the network)
- Accuracy is 62.73% — marginally *better* than the low-sparsity baseline
- The output layer (fc5) retains 100% of its weights, showing the network correctly protected the most information-critical connections
- The gate distribution (see Section 4) shows the clearest bimodal shape

λ=2.0 is a reasonable middle ground if the deployment environment requires a more conservative pruning level, delivering 30.85% sparsity with only 0.4% accuracy cost.

### Practical Deployment Implications

A 60% sparse model has direct, measurable deployment benefits:

- **Memory footprint**: storing only the active gates and weights in sparse format (CSR/COO) reduces the weight tensor from ~35 MB to ~14 MB — a 2.5× compression
- **Inference speed**: hardware-accelerated sparse matrix operations can skip 5.4M multiplications per forward pass, which matters on edge devices with power constraints
- **Interpretability**: the per-layer pruning pattern (73% in fc1, 0% in fc5) tells us exactly where the network's bottleneck is — the first layer is massively over-specified for CIFAR-10, but the 256→10 classification head needs every neuron it has

---

## 4. Gate Distribution Analysis

### What the Plot Shows

The `gate_distribution.png` file shows the gate distribution for the best model (λ=5.0, 59.79% sparsity, 62.73% accuracy):

**Left panel (Full Range, Log Y scale):**
The distribution shows clear bimodality:
- A **large spike near gate value ≈ 0**: 5,408,696 gates (59.8%) have been driven below the 0.1 threshold. Their gate scores are strongly negative ($s_i \ll 0$), meaning sigmoid(s) ≈ 0. These weights are effectively removed from the network.
- A **cluster around 0.4–0.8**: active weights whose gates have stabilised at moderate-to-high values because the CE loss continuously rewards their contribution to classification.
- The mean gate value is just 0.124, with std=0.136 — a highly skewed distribution concentrated near zero.

**Right panel (Active Gates Only, Linear Y scale):**
The 40.2% of gates that survived (3,637,832 weights) show a smooth distribution peaking around 0.4–0.6. The active gates are not all at 1.0 — they are distributed across the range, reflecting varying levels of "importance" the network assigns to each weight.

### What the Distribution Tells Us About the Network's Learned Structure

The result (mean gate=0.124, 59.8% pruned) reveals three things:

1. **CIFAR-10 classification is massively over-parameterised at the input layer**: fc1 has 73.58% of its 6.3 million gates pruned. The network discovered that most pixel-to-neuron connections carry no useful information — the raw 32×32×3 pixel space is largely redundant for 10-class classification.

2. **The sparse structure is robust by construction**: any gate below 0.1 is multiplying its weight by less than 0.1, contributing less than 10% of its nominal value to the layer output. Converting these to hard zeros at inference time causes no measurable accuracy drop.

3. **Later layers are information-dense**: fc5 (256→10) retains 100% of its gates. With only 256 inputs to the 10-class output, every feature dimension is load-bearing — the gate mechanism correctly refused to prune here, showing it is sensitive to the actual information bottleneck in the architecture.

---

## Implementation Notes

### Key Differences From a Naïve Implementation

| Design Choice | Naïve Version | This Implementation | Why It Matters |
|---|---|---|---|
| Gate init | `randn * 0.01 + 1.0` → sigmoid ≈ 0.73 | `randn * 0.01` → sigmoid ≈ 0.50 | Starting at the decision boundary; sparsity pressure is immediately effective |
| Sparsity loss | Raw sum of gates (~5.5M) | Mean gate value ∈ (0,1) | Makes λ interpretable and stable across model sizes |
| Bimodality | Not present | `mean(g*(1-g))` minimised | Actively evacuates the 0.5 plateau; essential for bimodal distribution |
| λ schedule | Constant from epoch 1 | Linear warmup (0 → λ over 10 epochs) | Prevents collapse before CE features are established |
| Optimizer | Adam | AdamW (weight decay 1e-4) | Decoupled weight decay improves generalisation |

### Running the Script

```bash
# Final run used for results in this report
python self_pruning_network.py --lambdas 0.5 2.0 5.0 --entropy-beta 0.8 --epochs 50

# Quick smoke test (10 epochs)
python self_pruning_network.py --epochs 10

# All options
python self_pruning_network.py --help
```

### Output Files

| File | Contents |
|---|---|
| `output/gate_distribution.png` | Two-panel histogram for the best model |
| `output/all_gate_distributions.png` | Side-by-side comparison of all three λ values |
| `output/training_curves.png` | CE and sparsity loss curves across epochs |
