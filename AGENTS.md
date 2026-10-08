# AGENTS.md

Guidance for any AI coding assistant working in this repository. Read this first.

> This file is intentionally NOT pushed to GitHub — it is local working guidance only.

---

## What this repo is

A deep-learning learning project (course CS 405, NUST/MCS) worked through group by group. It contains
per-group notes and labs, plus a combined capstone project called Gradient Autopsy. Read
`PROJECT_APPROACH.md` for the full plan and `PROGRESS.md` for current status before doing anything.

---

## Hard rules (do not violate)

1. **PyTorch only.** Never use Keras or TensorFlow. The point is to see inside the network — custom
   training loops, direct gradient inspection, no `.fit()` / high-level training abstractions.
2. **The user builds; the assistant guides and checks.** For the projects the user writes/runs the code
   himself. Deliver code cell-by-cell (clearly labeled markdown vs code cells), explain it, and ask
   concept-checking questions — do not silently produce a finished notebook unless asked.
3. **Hand-written training loops.** forward -> loss -> backward() -> optimizer step -> zero_grad(),
   explicit and visible. Gradient norms are logged after backward(), before zero_grad().
4. **Controlled experiments.** Change one variable at a time; hold seed, data, epochs fixed. Re-seed
   immediately before building each model.
5. **Verify before delivering.** Code shown to the user should be runnable. Never fabricate experimental
   numbers in reports — use the user's actual run outputs, mark anything representative as such.

## User preferences (apply always)

- No emojis in generated code.
- When code needs changing, give the complete updated cell/file, not a diff or "change this line".
- Direct, informal, concise. Match requested formats exactly.
- Quizzes: keep all answer options similar in length so the correct one can't be guessed from length.

---

## Conventions

- Notes: Markdown, Problem/Fix/Why cards, one file per group (or part).
- Lab naming: descriptive, ordered (e.g. `01-backprop-and-vanishing-gradient`).
- Reports: one Markdown report per notebook/lab, ending with high-difficulty interview questions + answers.

---

## Gradient Autopsy specifics (project COMPLETE)

- Dataset CIFAR-10. Config as a Python dict. Fixed fragile CNN, depth 8, no skip connections.
- Shared infra in `gradient_autopsy_common.py`, loaded via `%run`.
- Colab: set GPU runtime first, then the 3-cell header (mount Drive -> restore cached CIFAR -> `%run`).
- Short epoch budget (5-8) for sweeps.
- Known gotcha: `%run` runs common.py in a fresh namespace, so SAVE_DIR does not reach it and RESULTS_DIR
  defaults to the local VM (wiped between sessions). Repoint RESULTS_DIR at Drive before saving, or
  `%run -i`.
- Four notebooks: 1 Foundations (C5-C8), 2 Training Fixes (C9-C13), 3 Generalization (C14-C16),
  4 Synthesis (C17-C18). All done.

---

## Current status

Groups 1 & 2 (notes + labs + reports), the Gradient Autopsy project, and **Group 3 — Sequence Modeling**
(notes + Lab A + Lab B + the UCI HAR project) are all COMPLETE. Group 3 switched its project dataset from
WISDM to UCI HAR Smartphones (WISDM's Fordham download is dead, no clean no-auth mirror). **Next up:
Group 4 — Attention & Transformers** (notes, labs, Transformer-from-scratch project), then the Open-Ended
Lab (responsible-AI audit) and the HAR video final project. Always re-check `PROGRESS.md` for the live state.

Group 3 gotcha worth remembering: to backprop into the INPUT through a cuDNN RNN (e.g. gradient-through-time
measurement), the model must be in `train()` mode — `eval()` raises "cudnn RNN backward can only be called
in training mode". Fine when the model has no dropout/BN, since train/eval forward outputs are identical.