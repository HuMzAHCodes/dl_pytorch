# Gradient Autopsy — Complete Project Writeup

A from-scratch diagnostic study of why deep neural networks fail to train, and how modern
techniques fix them. This document is the full record of the project: what was built, how it was
structured, every technique applied and its tradeoffs, the problems hit along the way, and the final
results.

**Author:** Hassnain Bin Nisar (Naini) · **Framework:** PyTorch · **Platform:** Google Colab (T4 GPU) · **Dataset:** CIFAR-10

---

## 1. What this project is

Deep networks do not fail randomly — they fail in specific, measurable, predictable ways. A plain deep
CNN with no architectural safeguards (saturating activation, poor initialization, no normalization, no
skip connections) will reliably show vanishing gradients in its early layers and will overfit on limited
data. These are the default failure modes of deep learning before the field's standard fixes existed.

This project builds that failure on purpose, measures it with internal instrumentation rather than
intuition, and then applies — one at a time — the techniques developed to cure it. The goal is not a
high-accuracy model. The goal is **causal, numerical evidence** connecting each named technique (ReLU,
He init, BatchNorm, Adam, dropout, augmentation, transfer learning) to the specific failure it resolves.

It spans two study groups of a deep-learning course: Group 1 (Foundations) and Group 2 (Training
Dynamics and Generalization), merged into one project because the build-break-cure arc naturally covers
both.

---

## 2. The central method

Three principles drove every experiment:

1. **Reproduce the failure, do not just describe it.** Every problem is reproduced from a hand-written
   training loop with no black-box abstractions, so the mechanics are visible.
2. **Instrument internally.** Failures are detected from per-layer gradient statistics (gradient-norm
   heatmaps), not only from the loss curve — a vanishing gradient is visible in the gradients before it
   is visible in the loss.
3. **Change one variable at a time.** Architecture, seed, data, and epoch budget are held fixed while a
   single factor varies, so any measured difference is attributable to that factor. Seeds are reset
   immediately before building each model so comparisons start from identical weights.

Every concept is treated as a **testable claim with a before/after number**, not a fact to memorize.

---

## 3. How the project was structured

### 3.1 Phases

- **Phase 1 — Build and Break (Group 1):** build the fragile network and reproduce its two failures —
  the vanishing gradient and overfitting.
- **Phase 2 — Fix and Generalize (Group 2):** apply each cure as a measured intervention, split into
  training fixes (optimisation) and generalization fixes.
- **Synthesis:** verify the headline findings across seeds, and assemble the before/after story.

### 3.2 Four notebooks + one shared file

The project is split into four notebooks so a Colab session timeout never costs more than one notebook
of work. Shared infrastructure lives in `gradient_autopsy_common.py`, loaded into every notebook with
`%run`:

| File | Components | Contents |
|------|-----------|----------|
| `gradient_autopsy_common.py` | C1–C4 | config/seed/device, data pipeline, fragile CNN, instrumented training loop + heatmap |
| `01_foundations_build_and_break.ipynb` | C5–C8 | MLP vs CNN, CE vs MSE, broken heatmap, overfitting |
| `02_training_fixes.ipynb` | C9–C13 | activations, initialization, BatchNorm, optimizers, learning-rate sweep |
| `03_generalization.ipynb` | C14–C16 | regularization, augmentation, transfer learning |
| `04_synthesis.ipynb` | C17–C18 | seed-repeat verification, dashboard + final table |

Because `%run` restores C1–C4, each notebook starts at its *next* component rather than at C1. Two
numbering systems coexist: a "cell" is a physical cell in one notebook; a "C-number" is a global project
component (C1–C18).

### 3.3 Fixed experimental choices

- One dataset throughout: CIFAR-10.
- One fixed fragile architecture: a plain deep CNN, depth 8, no skip connections, defined once and never
  structurally changed.
- A fixed 4,000-image subset reserved for overfitting experiments (so overfitting is reproducible and
  comparisons are clean).
- Short epoch budgets (5–8) for sweeps — the object of study is *relative effect*, not leaderboard
  accuracy.

---

## 4. What each component did, and the result

### Phase 1 — Build and Break

- **C5 MLP vs CNN** — A fully-connected net and a CNN trained on CIFAR-10 at unequal parameter counts.
  Result: **CNN 65.8% vs MLP 51.0%** test accuracy, with the CNN using **5.4x fewer parameters** (76k vs
  411k). Convolution exploits image structure (local features, weight sharing, translation invariance);
  flattening throws it away.
- **C6 Cross-entropy vs MSE** — Same net, one epoch, two losses. Result: cross-entropy's first-layer
  gradient was **~30x larger** than MSE's. MSE passed through a saturated softmax produces near-zero
  gradients exactly when the model is confidently wrong.
- **C7 Vanishing-gradient heatmap** — The fully fragile net (sigmoid, no BN, default init, depth 8),
  gradients logged per layer. Result: output-layer gradient **~8,000,000x larger** than the input
  layer's; loss frozen at ~2.31, accuracy at chance. The signature "broken" heatmap.
- **C8 Overfitting** — A capable CNN on the 4,000-image subset. Result: train/test gap opened to
  **+0.108** over 25 epochs.

### Phase 2 — Training fixes

- **C9 Activations** — sigmoid vs ReLU vs LeakyReLU vs SELU. Result: last/first gradient ratio
  **~26,000,000x (sigmoid) -> ~11x (SELU)**; ReLU lifted the first-layer gradient ~6,600x.
- **C10 Initialization** — default vs Xavier vs He (ReLU net), measured at epoch 1. Result: He raised
  the first-layer gradient ~1000x and moved accuracy from **10% (chance) to 23%** — proving activation
  alone is not enough.
- **C11 BatchNorm** — on vs off on the sigmoid net, plus a train/eval buffer demonstration. Result:
  ratio **~6,800,000x -> ~1.0x**; running_mean shifted from zeros after training, and train vs eval
  output differed by 0.237 (proving BN is stateful and mode-dependent).
- **C12 Optimizers** — hand-rolled SGD and SGD+Momentum vs library Adam. Result: final loss **1.19 (SGD)
  -> 1.04 (momentum)**; both adaptive methods beat plain SGD.
- **C13 Learning-rate sweep** — too-low / good / too-high. Result: 1e-5 crawls, 1e-3 converges (loss
  1.14), 1.0 diverges (spiked to ~16 before oscillating).

### Phase 2 — Generalization

- **C14 Regularization** — dropout, L2 weight decay, early stopping. Result: gap **+0.107 (baseline) ->
  +0.050 (dropout)**; weight decay only modest (+0.087), and "both" (+0.049) barely better than dropout
  alone.
- **C15 Data augmentation** — flips/crops on train only. Result: gap **+0.109 -> +0.059** (halved),
  though test accuracy did not rise on the short 25-epoch budget.
- **C16 Transfer learning** — ImageNet ResNet-18, feature extraction then fine-tuning. Result: test
  accuracy **54% (from scratch) -> 76% (frozen backbone) -> 84% (fine-tune layer4)**, training only
  ~0.05% of parameters in the feature-extraction case.

### Synthesis

- **C17 Seed-repeat verification** — the three core gradient-flow findings re-measured across seeds 42,
  0, 123. Result: activation, initialization, and BatchNorm effects all **VERIFIED** (conditions
  separated by orders of magnitude, spread small relative to the gap).
- **C18 Dashboard + final table** — the broken-vs-fixed heatmap on identical axes (ratio **6,835,678x ->
  0.9x**) and the full technique-to-problem-to-effect-size table.

---

## 5. Techniques and their tradeoffs

| Technique | What it fixes | Tradeoff / cost |
|-----------|---------------|-----------------|
| CNN over MLP | parameter inefficiency on images | assumes spatial/grid input; not for unordered data |
| Cross-entropy | weak gradients from MSE on classification | only for classification, not regression |
| ReLU | vanishing gradient (saturating activation) | not zero-centered; "dying ReLU" (dead units never recover) |
| LeakyReLU / SELU | dying ReLU / self-normalization | extra hyperparameters; SELU needs matching init and architecture |
| He initialization | signal dies at the start | must match the activation (He for ReLU, Xavier for tanh/sigmoid) |
| BatchNorm | activation drift / saturation during training | stateful (train vs eval differ); breaks on batch size 1; adds compute |
| Momentum | slow SGD convergence | amplifies the effective learning rate (~1/(1-beta)), so LR often needs lowering |
| Adam / AdamW | slow, LR-sensitive training | can generalize slightly worse than tuned SGD; extra memory for moment estimates |
| Learning rate (tuning) | crawl vs diverge | the single most sensitive hyperparameter; no free lunch |
| Dropout | overfitting at the dense head | slows convergence; must be disabled at eval (model.eval()) |
| L2 weight decay | overfitting via large weights | little effect where capacity is not (e.g. conv layers); another hyperparameter |
| Early stopping | training past the generalization peak | needs a held-out validation signal to monitor |
| Data augmentation | overfitting / data scarcity | slower per-epoch fitting; test gains need longer budgets; transforms must preserve labels |
| Transfer learning | too little data to learn features | needs a relevant pretrained model; must match its input size and normalization |

A recurring theme: **the fixes are complementary, not interchangeable, and several interact.** He init
is entangled with ReLU; BatchNorm partly subsumes careful initialization; momentum changes the effective
learning rate; dropout and weight decay can be redundant (dropout did most of the work in C14). None is a
universal "best practice" — each answers a specific failure.

---

## 6. Problems faced, and how they were solved

This project hit several real-world obstacles. They are recorded here because working through them was
part of the learning.

1. **GPU not actually attaching.** Selecting "T4 GPU" in the Colab menu showed the GPU as selected, but
   `torch.cuda.is_available()` returned False, because the runtime had connected as CPU before the change
   and `device` was captured at that moment. Fix: disconnect and delete the runtime, reconnect with GPU
   selected first, then run the header so `common.py` captures `device=cuda`.

2. **CIFAR-10 download taking 30+ minutes.** A slow Colab node downloaded the 170 MB dataset at ~90 kB/s.
   Fix: after the first download, copy the dataset to Drive once (`MyDrive/cifar_data/data`) and restore
   it from there into `./data` at the start of every later session — reducing a 30-minute download to a
   few seconds, permanently.

3. **MLP-vs-CNN comparison came out as a tie.** The first run had a tiny CNN that merely matched the MLP,
   which did not demonstrate the point. Fix: give the CNN enough capacity (width 64, three conv blocks,
   still far smaller than the MLP) so it clearly wins — proving the efficiency claim rather than a tie.

4. **CE-vs-MSE effect risked being ambiguous.** This was flagged in advance as the finickiest experiment.
   Fix: apply MSE to softmax-probabilities-vs-one-hot and measure the first-layer gradient at epoch 1,
   where the saturation effect is sharpest. The effect came out clean at ~30x.

5. **Learning-rate "too high" did not diverge.** At lr=1e-1, Adam's adaptive scaling kept the run
   converging (just noisier), so the "diverge" regime was not visible. Fix: push the too-high rate to
   1.0, where even Adam destabilizes (loss spiked to ~16) — making the three regimes distinct.

6. **Early stopping did not trigger cleanly.** In the demo, test accuracy kept nudging up through epoch
   35, so patience never ran out in 40 epochs. This was kept and reported honestly: the run still
   illustrated the principle (test peaked at ~0.556 around epoch 35, then declined to 0.539 while train
   kept climbing — the exact condition early stopping catches).

7. **Results lost between sessions (the biggest lesson).** `%run` executes `common.py` in a *fresh*
   namespace, so it never saw the `SAVE_DIR` set in each notebook's header; `RESULTS_DIR` therefore
   defaulted to a local VM path, and every `.npy` saved by Notebooks 1–3 was written to ephemeral storage
   that is wiped when a session ends. The synthesis notebook, running in a new session, could not load
   them. Fix: repoint `RESULTS_DIR` explicitly at a Drive path before saving, and regenerate the two
   heatmaps the dashboard needs directly in the notebook (removing the cross-session dependency). General
   lesson: in Colab, persistence must target Drive explicitly, and `%run` does not share the notebook's
   variables unless invoked with `%run -i`.

---

## 7. Headline results

- The fragile network's early-layer gradient was **~8 million times** smaller than its output layer's —
  the vanishing gradient, measured directly.
- A single technique (BatchNorm) collapsed that ratio from **6,835,678x to 0.9x** — from a network that
  could not learn to one with healthy gradient flow at every depth.
- Overfitting's train/test gap was **roughly halved** by dropout and again by augmentation independently.
- Transfer learning raised test accuracy on limited data from **54% to 84%** while training a tiny
  fraction of the parameters.
- All three core gradient-flow findings held up across three random seeds — they are real effects, not
  lucky runs.

The complete technique-to-problem-to-effect-size table is in `04_synthesis.md` and the saved
`final_table.md`.

---

## 8. Repository structure

```
gradient-autopsy/
  gradient_autopsy_common.py
  notebooks/
    01_foundations_build_and_break.ipynb   + .md report
    02_training_fixes.ipynb                + .md report
    03_generalization.ipynb                + .md report
    04_synthesis.ipynb                     + .md report
  docs/
    PROJECT_WRITEUP.md   (this file)
    Gradient_Autopsy_Project_Statement.md
```

Each notebook has a companion Markdown report with its results and a set of high-difficulty,
interview-oriented concept questions with answers.

---

## 9. How to run

1. Set the Colab runtime to **T4 GPU first** (Runtime -> Change runtime type).
2. Upload `gradient_autopsy_common.py` to the root of Google Drive (`MyDrive/`).
3. In each notebook, run the header cells in order: mount Drive, restore CIFAR from the Drive cache,
   `%run` the common file, confirm `device=cuda`.
4. Run the notebooks in order (1 -> 4). Results save to Drive; the synthesis notebook regenerates the
   heatmaps it needs, so it does not depend on earlier sessions' files.

---

## 10. Conclusion

The project proved, with causal and numerical evidence, that deep networks fail in predictable ways and
that each modern technique resolves a specific failure. A deliberately fragile network was built, shown
to fail in two distinct ways (a vanishing gradient — an optimisation failure; and overfitting — a
generalisation failure), and each failure was cured by the techniques the field developed for it, with
every cure measured against the baseline it improved. The two signature artifacts — the broken-vs-fixed
heatmap and the effect-size table — let a reader grasp the whole story at a glance: a specific set of
techniques converted a network that could not learn into one that could. The headline findings were
verified across seeds, so the conclusions rest on reproducible effects. Beyond the results, the project
surfaced practical lessons about GPU setup, dataset caching, controlled comparison, honest reporting of
imperfect runs, and cross-session persistence — the parts of real experimental work that a clean tutorial
usually hides.
