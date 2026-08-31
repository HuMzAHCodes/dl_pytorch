# Notebook 2 — Training Fixes · Lab Report
### Gradient Autopsy · Phase 2 (part 1)

**Project:** Gradient Autopsy — a from-scratch diagnostic study of why deep networks fail to train and how modern techniques fix them.
**Phase:** 2, part 1 of 2 (heal the *training* failures exposed in Notebook 1).
**Framework:** PyTorch · **Platform:** Google Colab (T4 GPU) · **Dataset:** CIFAR-10.

---

## 1. Purpose of this notebook

Notebook 1 built a deliberately fragile network and reproduced its vanishing-gradient failure. This notebook applies, one at a time, the techniques the field developed to cure that failure — non-saturating activations, principled initialization, batch normalization, better optimizers, and a well-chosen learning rate — and measures the effect of each with the same per-layer gradient instrumentation. The controlling discipline throughout is to change exactly one factor while holding the architecture, seed, and data fixed, so that any improvement is attributable to the intervention under test. Every result is a measured "cure" for a specific mechanism of the "broken" baseline.

---

## 2. Experiments and results

### 2.1 Activation sweep (C9)

The fragile eight-layer network was trained four times, changing only the activation function, and the first-to-last-layer gradient ratio was recorded (smaller is healthier).

| Activation | First/last gradient ratio | Interpretation |
|------------|---------------------------|----------------|
| sigmoid (baseline) | ~26,000,000× | catastrophic vanishing |
| ReLU | ~214× | early layers rescued |
| LeakyReLU | ~213× | essentially identical to ReLU |
| SELU | ~11× | most balanced |

Switching from sigmoid to ReLU lifted the first-layer gradient by roughly four orders of magnitude (from ~10⁻⁸ to ~10⁻⁴), and SELU produced the most uniform gradient flow of all. The four heatmaps, plotted on a shared color scale, showed the characteristic dark-topped "broken" pattern for sigmoid dissolving into near-uniform color for the non-saturating activations. The activation function alone — with no other change — substantially heals the vanishing gradient.

### 2.2 Initialization sweep (C10)

Holding the activation fixed at ReLU, only the weight-initialization scheme was varied, and the epoch-1 gradient was measured (initialization being a starting-condition effect).

| Initialization | Epoch-1 first-layer gradient | Final test accuracy |
|----------------|------------------------------|---------------------|
| default | 1.7 × 10⁻⁴ | 10.0% (chance) |
| Xavier/Glorot | 2.8 × 10⁻² | 19.9% |
| He/Kaiming | 1.1 × 10⁻¹ | 23.1% |

Even with ReLU already in place, default initialization left the network stuck at chance-level accuracy, while He initialization — matched to ReLU — lifted the first-layer gradient by three orders of magnitude and moved test accuracy to 23%. This demonstrates that a good activation function is necessary but not sufficient: the *scale* of the initial weights independently determines whether the signal survives, and the two fixes stack. Notably, Xavier produced a slightly better *ratio* than He but weaker absolute gradients, which is why the absolute first-layer magnitude, not the ratio alone, is the more honest measure here.

### 2.3 Batch normalization (C11)

BatchNorm was added to the *sigmoid* network — the worst-case activation — and the gradient flow compared with and without it.

| Configuration | First/last gradient ratio |
|---------------|---------------------------|
| BatchNorm off | ~6,800,000× |
| BatchNorm on | ~1.0× |

BatchNorm reduced the gradient imbalance from nearly seven million to approximately one — near-perfect balance across all layers — without changing the activation. The two heatmaps were the most dramatic pair in the study: a dark-topped broken panel beside an almost uniformly bright one.

The notebook additionally demonstrated BatchNorm's train/eval behaviour, a detail frequently misunderstood. The first BatchNorm layer's `running_mean` buffer began at all zeros, shifted to small non-zero values after a few training steps (BatchNorm learning statistics from the data in `train()` mode), and the same input produced outputs differing by a mean absolute value of ~0.237 between `train()` and `eval()` modes. This is direct evidence that BatchNorm is stateful and behaves differently in the two modes — the reason omitting `model.eval()` at inference silently corrupts predictions.

### 2.4 Optimizers (C12)

Two optimizers were implemented from scratch — plain SGD and SGD with momentum — and compared against library Adam on the healed network.

| Optimizer | Final loss |
|-----------|-----------|
| SGD (hand-rolled) | 1.1923 |
| SGD + momentum (hand-rolled) | 1.0408 |
| Adam (library) | 1.0867 |

Both momentum and Adam outperformed plain SGD, consistent with theory: momentum accumulates a velocity across steps to accelerate through flat regions, and Adam additionally adapts the step size per parameter. Implementing momentum by hand made its relationship to SGD concrete — it is plain SGD plus a single velocity-accumulation line. On a task this small the curves are noisy and the margin between momentum and Adam is within run-to-run variance, so the robust conclusion is that both adaptive methods beat plain SGD, not that momentum beats Adam — a claim the seed-repeat verification in the synthesis notebook is designed to settle.

### 2.5 Learning-rate sweep (C13)

The healed network was trained three times under Adam, varying only the learning rate, to expose the three canonical regimes.

| Learning rate | Behaviour |
|---------------|-----------|
| 1 × 10⁻⁵ (too low) | crawl — loss flat and high, barely decreasing |
| 1 × 10⁻³ (good) | converge — smooth, steady descent to the lowest loss |
| 1.0 (too high) | diverge — loss spiked to ~16, oscillated violently, never settled well |

The plot showed all three regimes cleanly. A subtle but important observation: the too-high run's *final* loss (2.485) was below its *starting* loss (2.633), so a naive final-number check labelled it "converged" — yet its trajectory revealed an early spike to sixteen times the starting loss and persistent instability. This is a reminder that a single endpoint metric can mislead; the loss *trajectory* is what reveals training health. It also illustrates Adam's robustness: an adaptive optimizer barely recovered from a learning rate that would have sent plain SGD to infinity.

---

## 3. What Phase 2 (part 1) establishes

Every mechanism that made the Notebook 1 baseline fail to train has now been matched to a specific, measured cure. The saturating activation was replaced (ReLU/SELU restored gradient flow); the initial weight scale was corrected (He initialization revived the early-layer signal and moved accuracy off chance); the drift of activation statistics during training was controlled (BatchNorm balanced gradients to ~1× even on sigmoid); the update rule was improved (momentum and Adam beat plain SGD); and the dominant hyperparameter was characterised (a good learning rate is the difference between crawling, converging, and diverging). These fixes are complementary and stackable rather than mutually exclusive. What remains unaddressed is *generalisation* — the overfitting failure from Notebook 1 — which is the subject of Notebook 3, along with transfer learning for the underlying data-scarcity problem.

---

## 4. Conceptual & interview questions (advanced)

High-difficulty, interview-oriented questions on the concepts exercised in this notebook, with model answers.

---

**Q1. ReLU's derivative is exactly 1 for all positive inputs, so one might expect it to perfectly preserve gradient magnitude across depth. Yet its first-to-last ratio was ~214×, while SELU's was ~11×. Explain mechanically why ReLU does *not* perfectly preserve gradients, and what SELU does differently.**

ReLU's derivative is 1 on the positive side but exactly 0 on the negative side, and at initialization roughly half of any layer's pre-activations are negative. So while ReLU does not *shrink* the gradient the way a sigmoid's sub-unit derivative does, it *zeroes* the gradient for every unit whose input is negative — on average halving the number of active pathways at each layer. Across eight layers this repeated halving attenuates and imbalances the signal, producing the residual ~214× ratio. There is also no mechanism keeping the *scale* of activations constant from layer to layer, so variance can drift. SELU (scaled exponential linear unit) is constructed to be **self-normalizing**: its specific scale and negative-branch parameters are chosen so that, under suitable conditions, activations converge toward zero mean and unit variance as they propagate, and its negative branch has a non-zero (exponential) slope so gradient still flows through negative units. The combination of a non-zero negative-side gradient and variance self-stabilization is why SELU yields the most balanced gradient flow of the activations tested. The general lesson is that "non-saturating" removes the sigmoid catastrophe but is not the same as "gradient-preserving"; the latter also requires controlling how many pathways stay active and how activation variance evolves with depth.

---

**Q2. In the initialization sweep, Xavier produced a better first-to-last *ratio* than He, yet He gave higher accuracy and was reported as the better scheme. Reconcile these two facts, and state precisely when Xavier would be the correct choice over He.**

The ratio measures *balance* between the first and last layers' gradient magnitudes; it says nothing about the *absolute* strength of the learning signal. Xavier's initialization, derived assuming a symmetric activation with unit derivative around zero, under-scales the variance for a ReLU network (which zeroes half its inputs and therefore halves the forward variance). The result can be a well-*balanced* but globally *weak* gradient — a flat, quiet network where every layer learns slowly. He initialization accounts for ReLU's halving by using twice the variance (2/fan_in), producing a stronger absolute signal that trains faster, even if its balance ratio is marginally worse. Accuracy depends on the absolute signal strength driving actual weight updates, so He wins. Xavier is the correct choice when the activation is symmetric and roughly linear near the origin — tanh or sigmoid — because there the "halving" assumption behind He does not apply, and Xavier's variance is the matched one. The principle: initialization must be matched to the activation's forward variance behaviour; He for ReLU-family, Xavier for tanh/sigmoid-family.

---

**Q3. BatchNorm reduced the gradient ratio on a *sigmoid* network from ~6.8 million to ~1.0 without changing the activation. Sigmoid still saturates — so why does normalizing the pre-activations rescue the gradient so completely? Identify the exact quantity BatchNorm controls that matters for the sigmoid's derivative.**

The sigmoid's derivative is `σ(z)(1 − σ(z))`, which is near its maximum (0.25) when `z` is near zero and collapses toward zero as `|z|` grows — this saturation is the source of the vanishing gradient. What sends `z` to large magnitudes is drift in the *distribution* of each layer's pre-activations as weights change and as signal accumulates through depth. BatchNorm standardises exactly that pre-activation distribution to approximately zero mean and unit variance per mini-batch, which keeps `z` concentrated near zero — precisely the region where the sigmoid's derivative is largest and non-vanishing. So although the sigmoid nonlinearity is unchanged, its *operating point* is held in the high-derivative zone at every layer, so the per-layer derivative factors stay near their maximum rather than collapsing, and their product across depth no longer vanishes. The quantity BatchNorm controls is the mean and variance of the pre-activations feeding each nonlinearity; controlling that quantity is equivalent to preventing saturation. The learnable scale and shift (γ, β) let the network partially undo the normalization if needed, but the gradient rescue comes from the normalization step, not from γ and β.

---

**Q4. A model trained with BatchNorm gives excellent validation accuracy during training but produces near-random predictions when you load it for single-image inference in a deployed service. The weights are correct. Diagnose the bug precisely, and explain why it manifests specifically at single-image inference.**

The model is in `train()` mode at inference, and/or is receiving batches of size one. In `train()` mode BatchNorm normalizes using the *current batch's* statistics; at inference it should instead use the *running* statistics accumulated during training, which requires `model.eval()`. Two failure modes follow. First, if `eval()` was never called, a batch of one image is normalized by its own mean and variance — which reduces every channel to approximately zero and destroys the signal, because a single sample has no meaningful within-batch variance to normalize against. Second, even in `eval()`, if the running statistics were never properly accumulated (too few training steps, or a train/eval mode bug during training), the frozen statistics are wrong. It manifests specifically at single-image inference because with a large batch the batch statistics are a decent estimate of the true statistics, so `train()`-mode inference on big batches can look deceptively fine — the corruption only becomes catastrophic when the batch is too small to estimate statistics, i.e. batch size one. The fix is to call `model.eval()` before inference so BatchNorm uses its frozen running mean and variance, independent of batch size. This is the practical consequence of the train/eval buffer behaviour demonstrated in C11.

---

**Q5. Momentum was implemented as `v = β·v + grad; p -= lr·v`. Derive why, for a constant gradient `g`, the effective step size approaches `lr·g/(1−β)` rather than `lr·g`, and explain what this implies for choosing the learning rate when adding momentum to an existing SGD setup.**

With a constant gradient `g`, the velocity update `v ← β·v + g` is a geometric series. Starting from zero, after many steps the velocity converges to `v* = g + β·g + β²·g + … = g/(1−β)`, since the series sums to `1/(1−β)`. The parameter step is `lr·v*`, so the effective steady-state step is `lr·g/(1−β)`. For the common `β = 0.9`, `1/(1−β) = 10`, so momentum's steady-state step is roughly ten times larger than plain SGD's for the same gradient and learning rate. The practical implication is that momentum effectively *amplifies* the learning rate in consistent-gradient directions. When adding momentum to a tuned SGD setup, one should therefore typically *reduce* the base learning rate (a rule of thumb is by roughly the `(1−β)` factor) to avoid overshooting or divergence — a learning rate that was stable for plain SGD can become unstable once momentum multiplies the effective step. This also explains momentum's benefit: in directions where the gradient is consistent it builds up a large effective step (fast progress through flat regions), while in oscillating directions the sign changes cancel in the velocity sum, damping the oscillation.

---

**Q6. In the learning-rate sweep, the "too high" run (lr = 1.0) was flagged "converged" by the endpoint check because its final loss was below its starting loss, yet its trajectory clearly showed divergence. Design a robust programmatic criterion for detecting training instability that would correctly flag this run, and explain why each component is necessary.**

An endpoint comparison is insufficient because a run can spike catastrophically mid-training and then partially recover to below its starting point. A robust instability detector should combine several signals over the whole trajectory. First, check whether the loss ever exceeded its initial value by a large factor (for example, `max(loss) > k · loss[0]` with `k` around 2–3) — this catches the early spike to sixteen that the endpoint check missed. Second, check for non-finite values (`any(isnan or isinf)`) — true divergence often produces NaN. Third, quantify late-phase volatility, for example the standard deviation of the loss over the final fraction of steps relative to a converged baseline, or the fraction of steps where the loss increased — a healthy run's loss is near-monotone late in training, whereas the unstable run oscillates. Fourth, compare the smoothed final loss against a known-good reference rather than only against the start, since "better than random initialization" is a very low bar. Each component covers a different failure signature: the max-ratio test catches transient explosions, the NaN test catches hard divergence, the volatility test catches persistent oscillation that never settles, and the reference comparison catches "converged to a bad place." The too-high run fails the max-ratio and volatility tests even though it passes the naive endpoint test, so a criterion combining them flags it correctly. The general principle is that training health is a property of the *trajectory*, and any single scalar summary can be gamed by a pathological curve.

---

**Q7. All of Notebook 2's fixes were demonstrated on the small subset with short epoch budgets, and each was shown in isolation. A reviewer objects that isolated single-fix results may not compose — that fixing the activation, initialization, and normalization together might interact non-additively. Is the objection valid, and how would you empirically establish whether these fixes compose or interfere?**

The objection is valid in principle: deep-learning interventions do interact, and single-factor results do not guarantee additivity. He initialization is derived specifically for ReLU, so its benefit is entangled with the activation choice; BatchNorm partially subsumes the role of careful initialization because it renormalizes activations regardless of how they were initialized, so stacking BatchNorm on top of He may show diminishing returns rather than additive gains; and BatchNorm changes the effective gradient scale, which interacts with the optimal learning rate. To establish composition empirically, one would run a factorial (ablation) experiment rather than one-factor-at-a-time: enumerate the combinations of {activation, initialization, BatchNorm} — ideally the full 2×2×2 grid or a chosen subset — under identical seeds, epochs, and learning rate, and measure a common outcome (final accuracy and the gradient ratio) for each. Additivity would show as the combined effect approximating the sum of individual effects; interference or redundancy would show as sub-additive gains (e.g. He + BatchNorm barely better than BatchNorm alone), and synergy as super-additive gains. One would also re-tune the learning rate within each cell, because a fixed learning rate confounds the comparison when BatchNorm shifts the gradient scale. This factorial design is precisely the kind of controlled study the project's final synthesis moves toward, and it is why the single-fix results here are framed as mechanism demonstrations rather than as claims about the optimal combination.

---

*End of Notebook 2 report.*
