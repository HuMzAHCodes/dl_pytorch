# Deep Learning — Learning Approach & Project Plan

This document describes the overall approach for working through the Deep Learning course
(CS 405, NUST/MCS) — how the curriculum is divided into groups, how each group is run, the
projects attached to the groups, and the two final course deliverables. It is the master plan;
`PROGRESS.md` tracks what has actually been completed against it.

---

## 1. The core idea — a problem→solution chain

The theory topics are not grouped by category. They are grouped so that **each group ends by
exposing a problem that the next group solves.** A topic typically fixes a problem created by an
earlier topic (loss functions give something to optimize → backprop optimizes it → backprop creates
the vanishing-gradient problem → activations/init/BatchNorm solve that, and so on).

This chain is the spine of the whole curriculum. Before starting any group, we state its **spine
sentence**: *"This group ends when I can see [PROBLEM], which the next group fixes with [SOLUTION]."*

---

## 2. The per-group method — Notes → Labs → Project

Every group runs through the same three stages:

1. **Notes** — Problem/Fix/Why cards. Each concept is written as: the problem it addresses, the fix,
   and *why* the fix works. Short and problem-oriented, not encyclopedic (1–2 pages per concept).
2. **Labs** — a couple of lab tasks that **expose the problem, then solve it** — not merely
   demonstrate a technique working. Structure: reproduce the pain → apply the fix → show before/after.
3. **Project** — a group-specific project that **integrates** the group's concepts under realistic
   constraints (messy data, imbalance, limited data, latency).

**Boundary rule:** labs isolate one idea each; the project integrates them. Decide up front what the
labs prove vs what the project proves, and hold that line to avoid scope creep.

---

## 3. Standing policies

- **PyTorch only** — never Keras/TensorFlow. The goal is to "see inside the network" (custom training
  loops, gradient inspection, custom layers). Any Keras labs from the course manual are reimplemented
  in PyTorch.
- **Add related/industry-standard topics** when they genuinely earn their place (flagged as additions),
  rather than capping strictly at the listed topics.
- **Google Colab** for all work, with progressively heavier persistence precautions per group
  (Drive mounts, checkpointing, dataset caching, resumable training for the heavy final project).
- **Labs delivered cell-by-cell** (markdown + code cells clearly labeled) so the notebooks are built
  by hand, with concept-checking questions to confirm understanding.
- **Notes/reports as Markdown**, kept in the repo alongside the notebooks.

---

## 4. The four groups

Each group solves the problem the previous one exposed.

### Group 1 — Foundations
ANN, CNN, loss functions, backpropagation / gradient descent, first look at the vanishing gradient.
*Ends by exposing:* vanishing gradients, overfitting, slow training.

### Group 2 — Training & Generalization
Activation functions, weight initialization, batch normalization, optimizers, regularization,
data augmentation, transfer learning. *Every topic is a fix for a Group 1 problem.*
*Ends by exposing:* everything so far assumes independent inputs — no notion of order/sequence.

### Group 3 — Sequence Modeling
RNN, LSTM, GRU, deep RNNs, bidirectional RNNs, encoder–decoder.
*Ends by exposing:* RNNs are inherently sequential (slow, no parallelism) and struggle with very
long-range dependencies.

### Group 4 — Attention & Transformers
Attention mechanism and all transformer components. Solves Group 3's problem, and is required by the
final project.

---

## 5. The projects

### Group 1 + Group 2 (combined) — "Gradient Autopsy"
The projects for Groups 1 and 2 are **merged into a single project**, because its build→diagnose→cure
arc naturally spans both: Group 1 builds and breaks the network; Group 2 cures it. It lives in its own
folder (separate from the group folders, since it belongs to both).

**What it is:** take one deliberately fragile deep plain CNN (8–15 conv layers, no skip connections),
reproduce its failures with per-layer gradient instrumentation, then apply each modern fix as a
*measured intervention* and prove with numbers which technique cures which failure.

**Delivered as 4 notebooks + a shared `common.py`:**
- **Phase 1 — Build & Break** (Notebook 1): MLP vs CNN, cross-entropy vs MSE, the vanishing-gradient
  "broken" heatmap, overfitting.
- **Phase 2 — Fix & Generalize** (Notebooks 2 & 3):
  - Notebook 2 (training fixes): activations, initialization, BatchNorm, optimizers, learning-rate sweep.
  - Notebook 3 (generalization): regularization, data augmentation, transfer learning.
- **Synthesis** (Notebook 4): seed-repeat verification of the headline findings, and the side-by-side
  broken-vs-fixed gradient dashboard plus the final technique→fix→effect-size table.

Deliverable signatures: the side-by-side broken/fixed heatmaps on identical axes, and a final table
mapping each technique to the problem it solved and its measured effect size.

### Group 3 — project
The **sensor-based Human Action Recognition** (WISDM accelerometer/gyroscope data): compare
SimpleRNN vs LSTM vs GRU vs Bi-LSTM, plus a small encoder–decoder seq2seq task. Warms up directly for
the final project.

### Group 4 — project
Attention & a Transformer from scratch: implement self-attention and a small Transformer encoder by
hand, then fine-tune a pretrained transformer on the same task for comparison.

---

## 6. The two course deliverables (from the lab manual)

### Open-Ended Lab — Responsible-AI audit
Industry-standard version of the manual's "Bias and Uncertainty in AI": take a model for a
consequential task and deliver a full trustworthiness report in PyTorch — fairness metrics
(demographic parity, equalized odds, disparate impact), uncertainty quantification (MC-Dropout or
deep ensembles), out-of-distribution detection, an actual mitigation step with before/after numbers,
packaged as a mini model card.

### Final Project — Human Action Recognition (HAR), video
The capstone, and the reason the whole chain exists. A CNN backbone + LSTM/GRU + attention on video
(UCF101 or HMDB51). It **requires all four groups**: CNN (G1), transfer learning + regularization
(G2), sequence models (G3), attention (G4). The Group 3 sensor-based HAR is the warm-up; this is the
video-scale escalation.

---

## 7. The arc, end to end

Group 1 (build & break) → Group 2 (cure) — together the **Gradient Autopsy** project →
Group 3 (sequence modeling) → Group 4 (attention/transformers) → Open-Ended Lab (responsible AI) →
Final Project (HAR video, synthesizing all four groups).
