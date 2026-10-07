# Progress Tracker

Status of the Deep Learning course plan (see `PROJECT_APPROACH.md` for the full plan).
Last updated: 2026-10-07.

Legend: done · in progress · not started

---

## Group 1 — Foundations  —  COMPLETE
- Notes (Problem/Fix/Why): Part 1 (perceptron, loss, backprop) + Part 2 (CNN, vanishing gradient)
- Lab A — backprop by hand + vanishing-gradient reveal
- Lab B — MLP vs CNN on MNIST (params + accuracy + overfitting gap)
- Consolidated lab report (with interview questions)

## Group 2 — Training & Generalization  —  COMPLETE (notes + labs)
- Notes: 7 Problem/Fix/Why cards + problem->fix summary table
- Lab A — "Healing the Broken Network"
- Lab B — "Fighting Overfitting & Transfer Learning"
- Consolidated lab report (with interview questions)

---

## Combined project — "Gradient Autopsy" (Groups 1 + 2)  —  COMPLETE

All four notebooks, all four reports, both signature deliverables, seed-verified.

Shared infrastructure: `gradient_autopsy_common.py` (C1–C4 + C5 helpers), CIFAR-10 cached to Drive,
loaded into every notebook via `%run`.

### Notebook 1 — Foundations (Build & Break)  —  COMPLETE
- C5 MLP vs CNN: CNN 65.8% vs MLP 51.0%, 5.4x fewer params
- C6 CE vs MSE: cross-entropy gradient ~30x stronger
- C7 vanishing-gradient "broken" heatmap: early layers ~8,000,000x weaker
- C8 overfitting: train/test gap +0.108
- Report: `01_foundations_build_and_break.md`

### Notebook 2 — Training Fixes  —  COMPLETE
- C9 activations (26M x -> ~11x), C10 init (acc 10% -> 23%), C11 BatchNorm (6.8M x -> 1.0x + train/eval demo),
  C12 optimizers (hand-rolled SGD/Momentum), C13 LR sweep (crawl/converge/diverge)
- Report: `02_training_fixes.md`

### Notebook 3 — Generalization  —  COMPLETE
- C14 regularization (gap +0.107 -> +0.050), C15 augmentation (+0.109 -> +0.059),
  C16 transfer learning (54% -> 76% -> 84%)
- Report: `03_generalization.md`

### Notebook 4 — Synthesis  —  COMPLETE
- C17 seed-repeat verification: all three gradient-flow findings VERIFIED across 3 seeds
- C18 dashboard (ratio 6,835,678x -> 0.9x) + final technique->fix->effect-size table
- Report: `04_synthesis.md`

### Project documentation  —  COMPLETE
- `PROJECT_WRITEUP.md` (full project record: everything, problems faced, tradeoffs)
- `PROJECT_APPROACH.md`, `PROGRESS.md` (this file), `AGENTS.md` (local only)
- `Gradient_Autopsy_Project_Statement.md`

---

## Later in the plan  —  NOT STARTED
- Group 3 — Sequence Modeling (notes, labs, sensor-based WISDM HAR project)
- Group 4 — Attention & Transformers (notes, labs, Transformer-from-scratch project)
- Open-Ended Lab — Responsible-AI audit (fairness, uncertainty, OOD, mitigation, model card)
- Final Project — Human Action Recognition (HAR), video (UCF101/HMDB51), CNN+LSTM/GRU+attention

---

## Immediate next step
**Group 3 — Sequence Modeling.** Start with the notes (RNN, LSTM, GRU, deep/bidirectional RNNs,
encoder-decoder), then the labs, then the sensor-based WISDM HAR project. This reopens the chain toward
the HAR final project.

---

## Key working notes (for resuming cleanly)
- Dataset: CIFAR-10, config as a Python dict. Fixed fragile CNN depth = 8. Short epoch budget (5-8) for sweeps.
- Each notebook header: mount Drive -> restore CIFAR from Drive -> `%run` common.py. Set GPU runtime FIRST.
- CIFAR cached at `MyDrive/cifar_data/data`; restore into `./data` before `%run` to skip the download.
- Persistence lesson: `%run` runs common.py in a fresh namespace, so SAVE_DIR does not reach it and
  RESULTS_DIR defaults to the local VM (lost between sessions). Repoint RESULTS_DIR at a Drive path
  before saving, or use `%run -i`.
- Preferences locked in: no emojis in code; always give complete cells/files, not diffs.
