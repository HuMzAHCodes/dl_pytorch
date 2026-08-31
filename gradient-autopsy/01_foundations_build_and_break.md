# Notebook 1 — Foundations (Build & Break) · Lab Report
### Gradient Autopsy · Phase 1

**Project:** Gradient Autopsy — a from-scratch diagnostic study of why deep networks fail to train and how modern techniques fix them.
**Phase:** 1 of 2 (build the fragile network, reproduce and measure its failures).
**Framework:** PyTorch · **Platform:** Google Colab (T4 GPU) · **Dataset:** CIFAR-10.

---

## 1. Purpose of this notebook

Phase 1 builds a deliberately fragile deep network and reproduces its two characteristic failure modes — the **vanishing gradient** (a training failure) and **overfitting** (a generalization failure) — measuring each with internal instrumentation rather than inferring it from the loss curve. It also establishes two foundational comparisons that motivate the architecture and loss choices used throughout: **MLP vs CNN** and **cross-entropy vs MSE**. Every result here is a "before" baseline; the interventions that cure these failures are the subject of Phase 2.

---

## 2. Experiments and results

### 2.1 MLP vs CNN — architectural efficiency on images

A fully-connected network and a convolutional network were trained on CIFAR-10 for six epochs under identical seed, data, and epoch budget. Parameters, test accuracy, and training time were recorded.

| Model | Parameters | Test accuracy | Training time |
|-------|-----------|---------------|---------------|
| MLP   | 411,146   | 51.0%         | 94.4 s |
| CNN   | 76,298    | **65.8%**     | 112.6 s |

The CNN achieved **14.8 percentage points higher test accuracy while using roughly 5.4× fewer parameters**. The trajectories are as informative as the endpoints: the MLP plateaued near 51% for its final three epochs, whereas the CNN was still improving at epoch six with train and test accuracy close together. This demonstrates that convolution's parameter efficiency on images is structural — weight sharing, locality, and translation invariance — rather than a matter of raw capacity. Flattening an image into a vector discards the spatial relationships a CNN exploits, so the MLP must spend far more parameters to reach a worse result.

### 2.2 Cross-entropy vs MSE — the loss shapes the gradient

The same network was trained for one epoch two ways: cross-entropy on raw logits, and mean squared error applied to softmax probabilities against a one-hot target. The mean first-layer gradient norm in epoch one was measured.

| Loss | Epoch-1 first-layer gradient norm |
|------|-----------------------------------|
| Cross-entropy | 2.63652 |
| MSE           | 0.08784 |

Cross-entropy produced a gradient roughly **30× larger** than MSE. The measurement was taken early in training deliberately, because that is when the model is most often confidently wrong — precisely the situation in which a good classification loss should produce a strong corrective signal. MSE fails to do so: passed through a saturated softmax, its gradient nearly vanishes, so the network barely learns from its most serious mistakes. This is why cross-entropy, not MSE, is the correct loss for classification.

### 2.3 The vanishing gradient — the broken heatmap

The fully fragile network — eight convolutional layers deep, sigmoid activations, default initialization, no batch normalization — was trained while logging the gradient norm of every layer each epoch. The result was rendered as a layer × epoch heatmap on a log scale.

The heatmap showed a smooth descent from a bright output layer (log₁₀ gradient ≈ −1) to a near-black input layer (log₁₀ gradient ≈ −7). Quantitatively, the output-layer gradient was on the order of **8,000,000× larger** than the input-layer gradient throughout training. The consequence was visible in the loss: it remained near 2.31 and accuracy near 10% (chance level for ten classes) across all epochs. The network could not learn because its early layers received effectively no learning signal.

Crucially, the network was **broken but not dead** — it continued to run without numerical failure. Depth eight was chosen precisely to produce a dramatic but still-functional failure, so that Phase 2's cure would show a clean before/after contrast. The gradient history was saved to disk for later side-by-side comparison with the healed network.

### 2.4 Overfitting — the widening gap

A capable, healthy CNN was trained on the fixed 4,000-image subset for twenty-five epochs, recording train and test accuracy each epoch.

The two curves began together but diverged steadily: training accuracy climbed to roughly 65% while test accuracy plateaued near 54%, producing a final train–test gap of **+0.108**. The divergence accelerated in the later epochs, marking the point at which the model shifted from learning generalizable patterns to memorizing training-set specifics. The small dataset combined with a capable model is the classic recipe for this behaviour. The curves were saved for comparison against the regularized models of Phase 2.

---

## 3. What Phase 1 establishes

Two distinct failure modes have been reproduced and measured rather than merely described. The vanishing gradient is an **optimisation** failure — the learning signal cannot reach the early layers — and was made visible through per-layer gradient instrumentation. Overfitting is a **generalisation** failure — the model performs far better on seen data than unseen — and was made visible through the train–test gap. Each is deliberately left unfixed. Phase 2 introduces the specific techniques that address them: better activations, initialization, and normalization for the vanishing gradient; dropout, weight decay, and augmentation for overfitting; and transfer learning for the underlying data-scarcity problem.

---

## 4. Conceptual & interview questions (advanced)

The following are high-difficulty, interview-oriented questions on the concepts exercised in this notebook, with model answers.

---

**Q1. The instrumentation records `weight.grad.norm()` after `loss.backward()` and before `optimizer.step()`. Explain precisely why recording after `step()` but before the next `zero_grad()` would still be valid, whereas recording after the next `zero_grad()` would not — and what, if anything, `step()` mutates about `.grad`.**

`optimizer.step()` reads each parameter's `.grad` and uses it to update the parameter's data in place; it does **not** modify or clear `.grad` itself. Therefore the gradient tensor is still intact immediately after `step()`, and recording there would yield the same values as recording before `step()`. What clears `.grad` is `zero_grad()` (or `zero_grad(set_to_none=True)`, which sets it to `None`). Once that has run, `.grad` is either an all-zeros tensor or `None`, so any norm computed after it reflects nothing about the batch that was just processed. The only invalid recording points are before `backward()` (stale or `None`) and after `zero_grad()` (cleared). This distinction matters because a subtle bug — logging one line too late — produces a heatmap of zeros that looks like a catastrophic vanishing gradient but is actually an instrumentation artifact.

---

**Q2. MSE on a classification task produces weak gradients "when the model is confidently wrong." Derive this intuition through the softmax Jacobian. Why does cross-entropy avoid the same trap even though it also passes through softmax?**

Consider a class whose softmax probability is `p`. For MSE against a one-hot target, the loss gradient with respect to the logit involves the softmax Jacobian, whose diagonal terms scale like `p(1 − p)`. When the model is confidently wrong, the *correct* class has `p ≈ 0`, so `p(1 − p) ≈ 0` — the gradient through that path is suppressed exactly when the error is largest. This is a saturation effect identical in character to the vanishing gradient: a bounded nonlinearity's derivative collapses at its extremes, and the learning signal dies. Cross-entropy avoids this because its gradient with respect to the logits simplifies to `(p − y)` — the raw difference between predicted probability and target. The `p(1 − p)` factor from the softmax Jacobian is algebraically cancelled by the `1/p` term in the derivative of the log. So when the model is confidently wrong (`p ≈ 0`, `y = 1`), cross-entropy's gradient is close to `−1` — maximally strong — precisely where MSE's is near zero. The loss function does not merely score the model; it determines the shape of the optimisation landscape.

---

**Q3. In the broken heatmap, the last-layer gradient is ~8 million times the first-layer gradient, yet the loss is completely flat rather than the late layers learning while the early ones stall. Reconcile this — if the output layers have healthy gradients, why doesn't the network learn *anything*?**

A healthy gradient at the output layer is necessary but not sufficient for learning. The output layer's job is to classify the *features* handed to it by the layers below. In this network those features are near-random and effectively frozen, because the early layers — which build the low-level representations — receive no gradient and never update from their random initialization. The output layer can only learn a linear separation of whatever representation it is given; if that representation carries no class-discriminative structure (random features from an untrained stack of sigmoids), there is nothing for even a well-gradiented classifier to latch onto. Moreover, sigmoid saturation compresses activations toward a narrow range, so the features passed forward are not only random but low-variance. The result is a network that is essentially a linear classifier on noise — its loss sits at the entropy of the class prior (≈ ln 10 ≈ 2.30) and does not move. This is why the vanishing gradient is fatal rather than merely slowing: representation learning, which is the entire point of depth, never begins.

---

**Q4. The MLP–CNN comparison was run at unequal parameter counts (411k vs 76k), and the CNN won. A skeptic argues this is confounded — maybe the MLP overfit its extra parameters and would win with regularization, or maybe matched parameters would flip the result. How would you design the comparison to make the causal claim airtight, and why is the unequal-parameter result arguably already the stronger evidence?**

To isolate architecture as the causal factor, one would hold everything else fixed and vary only the parameter budget across a sweep — training MLPs and CNNs at several matched parameter counts (e.g. 20k, 75k, 200k) under identical seeds, epochs, optimizer, and regularization, and comparing accuracy at each budget. If the CNN dominates across the whole curve, architecture is established as the driver independent of capacity. One would also check the train accuracies to rule out the overfitting explanation: if the MLP's *train* accuracy is also lower, it is underfitting the structure, not overfitting the parameters. That said, the unequal-parameter result is arguably already stronger for the specific claim being made. The claim is not "CNNs are better at equal capacity" but "convolution uses parameters more efficiently on images." A CNN beating an MLP that has 5.4× *more* parameters is a direct demonstration of efficiency — the CNN extracts more signal per parameter. The confound the skeptic raises (MLP overfitting) is testable and, in this run, refuted by the MLP's low *train* accuracy plateau (~58%): it was not memorizing and failing to generalize; it simply lacked the inductive bias to represent the task well at all.

---

**Q5. Batch normalization is left off in the fragile baseline. Beyond "it's one of the fixes we test later," give the precise reason its presence would invalidate the Phase 1 vanishing-gradient measurement — and identify the one property of BatchNorm that does the work.**

Including BatchNorm would partially cure the very failure Phase 1 exists to expose, destroying the before/after contrast. The mechanism is specific: BatchNorm re-standardises each layer's pre-activations to roughly zero mean and unit variance per mini-batch. Sigmoid saturates — and its derivative collapses toward zero — when its inputs drift to large magnitudes; BatchNorm continuously pulls those inputs back into the sigmoid's high-derivative region around zero. This keeps the local derivative away from zero at every layer, so the product of derivatives across depth (which is what backpropagation multiplies) no longer collapses. In other words, the property doing the work is not the learnable scale/shift but the **normalisation of activation statistics**, which prevents saturation and thereby keeps the per-layer gradient factors near unity. With BatchNorm present, the deep sigmoid network would train and its early-layer gradients would be healthy — the heatmap would look "fixed" before any fix was applied, and the experiment would prove nothing. The same reasoning explains why the baseline also uses sigmoid rather than ReLU and default rather than He initialization: each withheld technique is a variable whose effect Phase 2 measures, so none may be present in the control.

---

**Q6. The gradient heatmap plots `log10(grad + 1e-12)`. A colleague proposes normalising each layer's gradient by that layer's number of weights before taking the log, arguing the raw norm unfairly favors larger layers. Is this a legitimate concern, and would the normalization change the conclusion?**

The concern is partially legitimate in general but does not change the conclusion here. A layer's gradient *norm* aggregates over all its weights, so a wider layer can have a larger norm simply from having more terms — comparing raw norms across layers of very different sizes can conflate "stronger per-weight signal" with "more weights." A per-weight or root-mean-square normalisation (dividing the norm by the square root of the parameter count) is a defensible way to compare the *typical* gradient magnitude per weight. However, in this architecture all convolutional layers share the same width and kernel size, so they have near-identical parameter counts; the normalisation would rescale every conv row by essentially the same constant and leave the input-to-output gradient *ratio* — the thing the heatmap is showing — unchanged. The eight-orders-of-magnitude collapse from output to input is far larger than any correction a per-weight normalisation could introduce. So while the colleague's principle is sound and worth adopting for comparisons across heterogeneous layer sizes, applying it here would not alter the diagnosis: the early layers are starved of signal by a margin that dwarfs any size-based bookkeeping.

---

**Q7. All comparisons re-seed immediately before building each model. Explain what specific failure this prevents, and describe a realistic scenario where forgetting to re-seed would produce a wrong scientific conclusion that still looks plausible.**

A pseudo-random generator is a sequence, and every draw advances its state. Building a model consumes random numbers to initialise its weights, advancing the generator; the next model built therefore starts from a different point in the sequence and gets different initial weights. Re-seeding immediately before each build guarantees both models start from the identical initial-weight distribution, so any measured difference is attributable to the variable under test rather than to initialisation luck. The realistic failure: suppose one compares ReLU against LeakyReLU on a moderately deep network, seeding once at the top. LeakyReLU, built second, happens to draw an initialisation that lands in a better basin; it reaches 2% higher accuracy. One concludes LeakyReLU is superior and reports it. But the effect is within initialisation variance — repeat with the seeds swapped and ReLU might win. The conclusion looks plausible (there is a real number backing it, and a plausible story about "leakage helping gradient flow"), yet it is an artifact of uncontrolled initialisation. This is why headline claims in the full project are additionally verified across multiple seeds with a reported spread: a single seeded run controls for *reproducibility*, but only repetition controls for *initialisation-driven chance*.

---

*End of Notebook 1 report.*
