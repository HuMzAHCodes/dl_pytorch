# Notebook 3 — Generalization · Lab Report
### Gradient Autopsy · Phase 2 (part 2)

**Project:** Gradient Autopsy — a from-scratch diagnostic study of why deep networks fail and how modern techniques fix them.
**Phase:** 2, part 2 of 2 (cure the *generalization* failure exposed in Notebook 1).
**Framework:** PyTorch · **Platform:** Google Colab (T4 GPU) · **Dataset:** CIFAR-10.

---

## 1. Purpose of this notebook

Notebook 1 exposed two failures: the vanishing gradient (an optimisation failure, cured in Notebook 2) and overfitting (a generalisation failure). This notebook cures the second. Using the same fixed 4,000-image CIFAR-10 subset that produced the original overfitting gap, it applies — and measures — the standard generalisation toolkit: dropout, L2 weight decay, and early stopping (regularisation); data augmentation; and transfer learning from a pretrained ResNet-18. As throughout the project, each technique is treated as a testable claim, and the controlling discipline is to change one variable at a time while holding the data and seed fixed.

## 2. Notebook architecture (how the project is split)

The project is divided into four notebooks so that a Colab session timeout never costs more than one notebook of work. All shared infrastructure — the configuration and seeding (C1), the data pipeline (C2), the fragile CNN (C3), and the instrumented training loop (C4) — lives in a single file, `gradient_autopsy_common.py`, and is loaded into every notebook with `%run`. Because that `%run` restores components C1–C4, each notebook begins at its *next* component rather than at the start: Notebook 1 covered C5–C8, Notebook 2 covered C9–C13, and **this notebook begins at C14**. Results are written to Google Drive as `.npy` arrays as each component finishes, so the final synthesis notebook (Notebook 4) can load every earlier result without re-running anything. (Note the two numbering systems: a "cell" is a physical cell in one notebook; a "C-number" is a global project component, C1 through C18.)

---

## 3. Experiments and results

### 3.1 Regularisation (C14)

A capable CNN was trained on the fixed 4,000-image subset under four configurations, measuring the final train/test gap (smaller is better).

| Configuration | Final train–test gap |
|---------------|----------------------|
| baseline (no regularisation) | +0.107 |
| dropout (0.5) | +0.050 |
| L2 weight decay (1e-3) | +0.087 |
| dropout + weight decay | +0.049 |

The baseline gap of +0.107 reproduced Notebook 1's overfitting result (+0.108) almost exactly — a reproducibility check, since the seed and data were identical. **Dropout more than halved the gap**, while weight decay helped only modestly, and combining both was barely better than dropout alone (+0.049 vs +0.050). The honest reading is that dropout did nearly all of the work here and weight decay added little on top of it — a consequence of *where* each regulariser acts (discussed in Q1 below).

Early stopping was demonstrated separately. Rather than triggering cleanly, the run illustrated the underlying dynamic: test accuracy rose to a peak of ~0.556 around epoch 35, then declined to 0.539 by epoch 40 while training accuracy kept climbing to 0.710 — the gap widening from +0.020 to +0.171. That decline-after-peak is precisely the condition early stopping is designed to catch (halt at the peak and keep the best model).

### 3.2 Data augmentation (C15)

The same 4,000-image subset was rebuilt with random horizontal flips and crops applied to the training data only (the test set stayed on the plain transform). Training accuracy was measured on the plain subset so the gap comparison was apples-to-apples.

| Setting | Final train–test gap | Final test accuracy |
|---------|----------------------|---------------------|
| no augmentation | +0.109 | 0.543 |
| augmentation | +0.059 | 0.501 |

Augmentation roughly **halved the overfitting gap**. An important and honest caveat: test accuracy did *not* improve on this 25-epoch budget (it fell slightly, 0.543 → 0.501). The gap shrank mainly by pulling training accuracy down, not by lifting test accuracy. This is expected — augmented data is effectively "new" every epoch, so the model fits more slowly and needs a longer budget before its test accuracy overtakes the baseline. The correct conclusion is that augmentation reduced overfitting, with its test-accuracy payoff deferred to longer training, rather than that it "failed."

### 3.3 Transfer learning (C16)

An ImageNet-pretrained ResNet-18 was adapted to the 4,000-image subset (inputs resized to 224px and normalised with ImageNet statistics) in two regimes, and compared against the from-scratch CNN.

| Approach | Best test accuracy |
|----------|--------------------|
| from-scratch CNN (baseline) | ~0.540 |
| transfer — feature extraction (frozen backbone) | 0.760 |
| transfer — fine-tuning (unfrozen `layer4`) | 0.841 |

This was the strongest result in the project. Feature extraction — training only the new classification head, about **0.05% of the parameters** — raised test accuracy by 22 points over the from-scratch network. Fine-tuning the last residual block at a small learning rate added a further 8 points, reaching 0.841. On limited data, reusing a feature extractor pretrained on 1.2M images decisively beats anything trainable from scratch.

---

## 4. What this notebook establishes

The overfitting failure from Notebook 1 now has a measured set of cures, each attacking the problem from a different angle: dropout and weight decay constrain the *model*; augmentation enriches the *data*; transfer learning sidesteps the underlying *data scarcity* entirely by importing knowledge from a larger dataset. Two honest nuances emerged that matter more than a clean success story: regularisers can be redundant with one another (weight decay added little once dropout was present), and a technique's benefit can live on a different axis than the one being measured (augmentation shrank the gap without raising short-budget test accuracy). With training fixes (Notebook 2) and generalisation fixes (this notebook) both complete, the only remaining work is synthesis: verifying the headline findings across seeds and assembling the before/after comparison — the subject of Notebook 4.

---

## 5. Conceptual & interview questions (advanced)

**Q1. Dropout more than halved the overfitting gap, but weight decay barely helped and added almost nothing on top of dropout. Give a mechanistic reason, tied to where each regulariser acts and where this network's capacity lives.**

Each regulariser acts in a different place, and the network's overfitting pressure is concentrated in one of them. Dropout was placed in the classifier head, directly before the final linear layer — the point where pooled features are mapped to class scores on only 4,000 images, which is where memorisation is most acute. Zeroing half those features each step forces the classifier to spread its reliance across many features rather than memorising specific ones, attacking overfitting exactly where it occurs. L2 weight decay, by contrast, applies a uniform penalty to *all* weights, most of which sit in the convolutional backbone — but conv layers have few parameters (weight sharing), so they weren't the main source of overfitting, and penalising them yields little. Because the memorisation bottleneck is the dense head, dropout (which targets it) dominates, and once dropout has largely removed the head's capacity to memorise, weight decay has little left to constrain — hence "both" ≈ "dropout alone." The general lesson: match the regulariser to where the model's excess capacity actually is.

**Q2. A BatchNorm-style gotcha applies to dropout too. If you ran inference with dropout accidentally left on (`model.eval()` not called), would predictions be biased or merely noisy, and why?**

Merely noisy, not biased — because PyTorch uses inverted dropout. During training, dropout zeroes a fraction p of activations *and scales the survivors by 1/(1−p)*, which preserves the expected value of each activation. So at inference with dropout mistakenly active, each forward pass drops a different random subset but the *expected* output still matches the training-time distribution — predictions are not shifted systematically in any direction. What you lose is determinism and per-pass accuracy: every forward pass gives a slightly different answer, and single-pass accuracy drops because half the signal is discarded each time. Averaging many such stochastic passes is exactly Monte-Carlo Dropout, used for uncertainty estimation — the same accidental behaviour, turned deliberate. So: unbiased in expectation, noisy per pass.

**Q3. Augmentation halved the gap but test accuracy fell slightly on a 25-epoch budget. A reviewer says "the experiment shows augmentation hurts — drop it." Rebut this rigorously.**

The reviewer is reading a single endpoint on too short a timescale. Augmentation's mechanism is to make the training data effectively new every epoch, which both suppresses memorisation (hence the halved gap) and slows per-epoch fitting (the model never sees an identical image twice). On a 25-epoch budget the augmented model is therefore still "behind" on raw fitting, so its test accuracy hasn't yet caught up — but the plain model is simultaneously heading into overfitting, where its test accuracy will plateau and then degrade. The standard pattern is that the augmented model overtakes the plain one on test accuracy with more epochs, and generalises better thereafter. A rigorous evaluation would extend both runs until each has reached its own best validation point (ideally with early stopping) and compare *those* peaks, rather than comparing a fixed short epoch count. Dropping augmentation on the basis of one short run would discard the single most effective anti-overfitting tool in vision on the basis of a measurement taken before its benefit materialises.

**Q4. Feature extraction trained only ~0.05% of the parameters yet beat the from-scratch CNN by 22 points. Explain how training so few parameters can help so much, and what the frozen 99.95% is contributing.**

The frozen 99.95% is doing the hard part. ResNet-18's backbone was trained on 1.2M ImageNet images to build a general visual representation — edges and textures early, shapes and object parts deep — and those features transfer to CIFAR because low- and mid-level vision is largely task-independent. Feature extraction reuses that representation unchanged and only learns the final mapping from high-level features to the 10 CIFAR classes, which is a simple problem solvable from a few thousand examples. The from-scratch network, by contrast, must learn the entire feature hierarchy *and* the classifier from only 4,000 images — far too little data to learn good features — which is why it caps around 54%. The number of trained parameters is tiny, but it sits atop a feature extractor effectively worth millions of images of training; the value transferred is in the frozen weights, not the trained ones.

**Q5. Fine-tuning beat feature extraction (0.841 vs 0.760). Justify the two specific choices that made fine-tuning work: unfreezing only the last block, and using a learning rate 10× smaller than the head's.**

Both choices follow from how feature generality varies with depth. Early layers encode generic features (edges, textures) useful for almost any image task, so they need no adaptation; freezing them preserves that knowledge and prevents overfitting the limited data. The last block (`layer4`) encodes the most task-specific, high-level features, which were tuned for ImageNet's categories and benefit from adapting to CIFAR — so it is the block worth unfreezing. The small learning rate is essential because the pretrained weights are already good: the aim is to nudge them, not overwrite them. A large learning rate would take large, destructive steps that wash out the ImageNet knowledge the model is meant to retain, which commonly makes fine-tuning *worse* than feature extraction. The principle — unfreeze late, step gently — adapts the high-level features to the new task without catastrophic forgetting of the transferable ones.

**Q6. The transfer-learning data used ImageNet normalisation and 224px inputs, not CIFAR's native stats at 32×32. Why is matching the pretrained model's input conditions not optional?**

A pretrained model's weights are calibrated to the exact input distribution and scale it was trained on. The first convolution and every downstream layer assume ImageNet-normalised inputs; feeding CIFAR-normalised or raw pixels shifts the input statistics away from those assumptions and degrades the quality of every feature the backbone produces — the network is being handed inputs it was never designed to read. Scale matters similarly: ImageNet models were trained on ~224px images, and their receptive fields and pooling are sized for that; at 32×32 the spatial information is nearly exhausted by the time it reaches the deep layers, so the pretrained features cannot operate as intended. Resizing to 224 and applying ImageNet normalisation makes CIFAR images "look like" the data the backbone expects, so the transferred features remain valid. Mismatched preprocessing is a common, silent reason transfer learning underperforms in practice — matching input conditions is part of the method, not a detail.

**Q7. Across C14–C16, three different techniques all reduced overfitting. If you could keep only one for a *new* small-data image task, which would you choose and why — and under what condition would your answer change?**

For a new small-data image task, transfer learning is the first choice by a wide margin: it addressed the root cause — too little data to learn good features — and produced the largest effect (54% → 76–84%), whereas dropout and augmentation only mitigate overfitting of a model still learning features from scratch. The reasoning is that regularisation and augmentation make better use of limited data, but transfer learning *imports* the equivalent of vastly more data in the form of pretrained features, which is a categorically stronger lever when data is the binding constraint. The answer changes when transfer learning is unavailable or inapplicable: if no suitable pretrained backbone exists for the domain (e.g. a sensor or medical modality unlike ImageNet), or licensing/compute forbids it, then augmentation becomes the primary tool (it is broadly effective and cheap), with dropout and weight decay as complements. In short: transfer learning when a relevant pretrained model exists; augmentation-led regularisation when it does not.

---


