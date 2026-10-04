# Progress Tracker

Status of the Deep Learning course plan (see `PROJECT_APPROACH.md` for the full plan).
Last updated: 2026-10-04.

Legend: ✅ done · 🔄 in progress · ⏳ not started

---

## Group 1 — Foundations  ✅ COMPLETE
- ✅ Notes (Problem/Fix/Why): Part 1 (perceptron, loss, backprop) + Part 2 (CNN, vanishing gradient)
- ✅ Lab A — backprop by hand + vanishing-gradient reveal (per-layer gradient-norm plot)
- ✅ Lab B — MLP vs CNN on MNIST (param count + accuracy + overfitting gap)
- ✅ Consolidated lab report (with interview questions)

## Group 2 — Training & Generalization  ✅ COMPLETE (notes + labs)
- ✅ Notes: 7 Problem/Fix/Why cards (activations, init, batch norm, optimizers, regularization,
  augmentation, transfer learning) + problem→fix summary table
- ✅ Lab A — "Healing the Broken Network" (ReLU → He init → BatchNorm → optimizer comparison)
- ✅ Lab B — "Fighting Overfitting & Transfer Learning" (dropout + L2 + augmentation, then ResNet-18)
- ✅ Consolidated lab report (with interview questions)

---

## Combined project — "Gradient Autopsy" (Groups 1 + 2)  🔄 IN PROGRESS (2 of 4 notebooks done)

Shared infrastructure: ✅ `gradient_autopsy_common.py` (C1–C4 + C5 helpers), CIFAR-10 cached to
Drive, GPU runtime working, 4-notebook split to survive session timeouts.

### Notebook 1 — Foundations (Build & Break)  ✅ COMPLETE
- ✅ C5 MLP vs CNN → CNN **65.8%** vs MLP **51.0%** test acc, with **5.4× fewer** params (76k vs 411k)
- ✅ C6 CE vs MSE → cross-entropy first-layer gradient **~30× stronger** than MSE
- ✅ C7 vanishing-gradient "broken" heatmap → early layers **~8,000,000× weaker** than output
- ✅ C8 overfitting → train/test gap opened to **+0.108** on the fixed 4k subset
- ✅ Report: `01_foundations_build_and_break.md` (7 interview questions)

### Notebook 2 — Training Fixes  ✅ COMPLETE
- ✅ C9 activation sweep → ratio **26M× (sigmoid) → ~11× (SELU)**; ReLU/LeakyReLU/SELU all heal it
- ✅ C10 init sweep → He init lifts first-layer gradient ~1000×; accuracy **10% → 23%**
- ✅ C11 BatchNorm → rescues even sigmoid (**6.8M× → 1.0×**); plus train/eval buffer demo (diff 0.237)
- ✅ C12 optimizers → hand-rolled SGD/Momentum + Adam; momentum & Adam beat plain SGD
- ✅ C13 LR sweep → crawl (1e-5) / converge (1e-3) / diverge (1.0, spiked to loss ~16)
- ✅ Report: `02_training_fixes.md` (7 interview questions)

### Notebook 3 — Generalization  ⏳ NOT STARTED
- ⏳ C14 regularization (dropout, L2 weight decay, early stopping)
- ⏳ C15 data augmentation (flips/crops/jitter, train only)
- ⏳ C16 transfer learning (ResNet-18 feature-extraction + fine-tuning vs from-scratch)
- ⏳ Report: `03_generalization.md`

### Notebook 4 — Synthesis  ⏳ NOT STARTED
- ⏳ C17 seed-repeat verification (top ~3 findings across 3 seeds, mean ± spread)
- ⏳ C18 side-by-side broken-vs-fixed dashboard + final technique→fix→effect-size table
- ⏳ Report: `04_synthesis.md`

---

## Later in the plan  ⏳ NOT STARTED
- ⏳ Group 3 — Sequence Modeling (notes, labs, sensor-based WISDM HAR project)
- ⏳ Group 4 — Attention & Transformers (notes, labs, Transformer-from-scratch project)
- ⏳ Open-Ended Lab — Responsible-AI audit (fairness, uncertainty, OOD, mitigation, model card)
- ⏳ Final Project — Human Action Recognition (HAR), video (UCF101/HMDB51), CNN+LSTM/GRU+attention

---

## Immediate next step
Start **Notebook 3** of Gradient Autopsy — begin with **C14 (regularization)**.

---

## Key working notes (so a fresh session can resume cleanly)
- Dataset: CIFAR-10, config as a Python dict (notebook style). Fixed fragile CNN depth = 8
  ("sick but alive"). Short epoch budget (5–8) for sweeps.
- Each notebook starts with the 3-cell header: mount Drive → restore CIFAR from Drive → `%run`
  `common.py`. Set GPU runtime FIRST, then run the header (so `device=cuda`, not cpu).
- CIFAR cached at `MyDrive/cifar_data/data`; restore it into `./data` before `%run` to skip the
  ~30-min re-download.
- Results saved to Drive (`.npy`) as each notebook runs, so Notebook 4 can load earlier outputs
  without re-running.
- Review checklist for the project build: (1) ReLU is NOT zero-centered — frame activation-mean
  logging to show that, don't imply it is; (2) keep author name consistent; (3) depth guardrail —
  start ~8 layers, go deeper only if the baseline isn't broken enough; (4) CE-vs-MSE is the finicky
  experiment — already landed at ~30×.
