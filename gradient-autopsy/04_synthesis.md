# Notebook 4 — Synthesis · Lab Report
### Gradient Autopsy · Finale

**Project:** Gradient Autopsy — a from-scratch diagnostic study of why deep networks fail and how modern techniques fix them.
**Phase:** Synthesis (verify the findings, assemble the before/after story).
**Framework:** PyTorch · **Platform:** Google Colab (T4 GPU) · **Dataset:** CIFAR-10.

---

## 1. Purpose of this notebook

This notebook closes the project. Notebooks 1-3 reproduced the failures and applied the fixes one at a time; this notebook does two things that turn a set of individual experiments into a coherent result. First, it verifies the headline "healing" findings across multiple random seeds, so they can be reported as genuine effects rather than lucky initializations. Second, it assembles the two signature deliverables: the side-by-side broken-versus-fixed gradient heatmap, and the final technique-to-problem-to-effect-size table.

## 2. Notebook architecture and a reproducibility lesson

The project is split into four notebooks so that a Colab session timeout never costs more than one notebook of work. All shared infrastructure — configuration and seeding (C1), the data pipeline (C2), the fragile CNN (C3), and the instrumented training loop (C4) — lives in a single file, `gradient_autopsy_common.py`, and is loaded into every notebook with `%run`. Because that one line restores components C1-C4, each notebook begins at its *next* component rather than at the start: Notebook 1 covered C5-C8, Notebook 2 covered C9-C13, Notebook 3 covered C14-C16, and this notebook begins at C17. Two numbering systems coexist and should not be confused: a "cell" is a physical cell within one notebook, while a "C-number" is a global project component (C1 through C18).

A reproducibility lesson emerged in this notebook and is worth recording. The `%run` magic executes `gradient_autopsy_common.py` in a *fresh* namespace, so the shared file never saw the `SAVE_DIR` variable set in each notebook's header. As a result its `RESULTS_DIR` defaulted to a local path on the Colab virtual machine rather than Google Drive, and every intermediate `.npy` array saved by Notebooks 1-3 was written to ephemeral VM storage that is wiped when a session ends. When this notebook — running in a new session on a new VM — tried to load those files for the dashboard, they were gone. The fix was twofold: repoint `RESULTS_DIR` explicitly at a Drive path before saving, and regenerate the two heatmaps the dashboard needs directly in this notebook (two short training runs), removing the cross-session file dependency entirely. The broader lesson is that cross-session persistence in Colab must target Drive explicitly, and that `%run` does not share the calling notebook's variables unless invoked with `%run -i`.

---

## 3. Seed-repeat verification (C17)

A single run can mislead: an effect might reflect a lucky or unlucky random initialization rather than a property of the technique. The three core "healing" findings were therefore re-measured across three seeds (42, 0, 123), each as a single forward/backward pass, and summarized as mean plus or minus standard deviation. A finding was treated as verified when the two conditions' mean-plus-or-minus-spread bands did not overlap.

| Finding | Broken condition | Fixed condition | Verdict |
|---------|------------------|-----------------|---------|
| Activation (last/first gradient ratio) | sigmoid: order 10^7 | ReLU: order 10^2 | VERIFIED |
| Initialization (first-layer gradient norm) | default: order 10^-4 | He: order 10^-1 | VERIFIED |
| BatchNorm (last/first gradient ratio) | off: order 10^7 | on: order 10^1 | VERIFIED |

All three were verified: in every case the two conditions were separated by several orders of magnitude while the spread across seeds stayed small relative to that separation. These are the effects that matter most to the project's thesis — the components of the vanishing-gradient cure — and they hold across random initializations, not just on one seed.

## 4. The dashboard and the final table (C18)

### 4.1 Broken versus fixed

The dashboard places the fragile sigmoid network beside the same network with BatchNorm added, on identical axes and a shared color scale so the comparison is honest. The broken panel shows the signature vanishing-gradient pattern — near-black early-layer rows grading to a bright output layer — while the fixed panel is nearly uniform, indicating healthy gradient flow at every depth. The headline contrast from this run: the last-to-first-layer gradient ratio collapsed from **6,835,678x** (broken) to **0.9x** (fixed). A network whose early layers received essentially no learning signal became one in which every layer learns at a comparable rate, by the addition of a single technique.

### 4.2 Final effect-size table

| Technique | Problem addressed | Measured effect |
|-----------|-------------------|------------------|
| MLP vs CNN | FC nets are parameter-inefficient on images | CNN 65.8% vs MLP 51.0% test acc, 5.4x fewer params |
| Cross-entropy vs MSE | MSE gives weak gradients on classification | CE first-layer gradient ~30x larger than MSE |
| ReLU (vs sigmoid) | Vanishing gradient (saturating activation) | last/first grad ratio 26,000,000x -> ~11x |
| He init (vs default) | Signal dies at initialization | first-layer gradient ~1000x stronger; acc 10% -> 23% |
| BatchNorm | Gradient imbalance / saturation during training | ratio 6,800,000x -> ~1.0x (on sigmoid net) |
| Adam / Momentum | Slow convergence (plain SGD) | final loss 1.19 (SGD) -> 1.04 (momentum) |
| Learning rate | Crawl / divergence | good lr converges (1.14); too-high diverges (spike to ~16) |
| Dropout | Overfitting (model memorizes small data) | train/test gap +0.107 -> +0.050 |
| Data augmentation | Overfitting (data scarcity) | gap +0.109 -> +0.059 |
| Transfer learning | Too little data to learn features from scratch | test acc 54% -> 76% (freeze) -> 84% (fine-tune) |

Every row carries a real measured number, and every technique is tied to a specific failure that was first reproduced and measured before the fix was applied. This is the project's central claim made concrete: each named technique is not a general best practice taken on faith, but a specific, quantified answer to a specific problem.

---

## 5. Project conclusion

The project set out to prove, with causal and numerical evidence, that deep networks fail in predictable ways and that each modern technique resolves a specific failure. That has been demonstrated end to end. A deliberately fragile network was built and shown to fail in two distinct ways — a vanishing gradient (an optimisation failure, visible in per-layer gradient statistics before it was visible in the loss) and overfitting (a generalisation failure, visible in the train/test gap). Each failure was then cured by the techniques the field developed for it — non-saturating activations, principled initialisation, and normalisation for the gradient; regularisation, augmentation, and transfer learning for generalisation — and every cure was measured against the baseline it improved. The two signature artifacts, the broken-versus-fixed heatmap and the effect-size table, let a reader grasp the whole story at a glance: a specific set of techniques converted a network that could not learn into one that could. The headline findings were additionally verified across random seeds, so the conclusions rest on reproducible effects rather than single fortunate runs.

---

## 6. Conceptual & interview questions (advanced)

**Q1. Why is seed-repeat verification necessary at all? If a single run shows sigmoid's gradient ratio is 26 million and ReLU's is 11, isn't that gap obviously real without repeating it?**

For effects this large the repeat mostly confirms what is already clear, but it is necessary as a matter of method rather than for this particular gap. Any single run fixes one random initialization, and initialization affects the measured quantity — a gradient norm on a randomly initialized network is itself a random variable. A gap of six orders of magnitude is far larger than initialization variance, so it would survive any seed; but the discipline of verifying exists for the cases where the effect is small, where a 2-3% accuracy difference could easily be initialization luck rather than a real improvement. Reporting mean plus or minus spread makes the distinction explicit: when the spread is tiny relative to the gap (as here), the effect is real; when the bands overlap, the claimed effect is within noise and should not be reported as a finding. Applying the same standard uniformly — including to effects that obviously pass — is what keeps the methodology honest, because it removes the temptation to demand rigor only for results one happens to doubt.

**Q2. The "fixed" heatmap used BatchNorm added to the sigmoid network, not the fully-corrected ReLU-plus-He-plus-BatchNorm network. Is that a fair representative of "fixed," and why might it actually be the better choice for the dashboard?**

It is both fair and arguably the better choice, because it is a controlled comparison. The broken and fixed panels differ in exactly one factor — BatchNorm off versus on — with the activation (sigmoid), initialization (default), and depth held identical. That isolation is what makes the dashboard an honest demonstration: any difference between the panels is attributable to BatchNorm alone, not to a bundle of simultaneous changes. Had the fixed panel used the fully-corrected network, it would also be healthy, but the comparison would conflate three interventions and could not attribute the cure to any one of them. Choosing the single most dramatic single-factor fix — BatchNorm, which drove the ratio from millions to about one even on the worst-case sigmoid activation — gives the clearest and most rigorous before/after. The cumulative effect of all the fixes together is documented separately in the effect-size table; the dashboard's job is a clean, single-variable visual, which the BatchNorm comparison delivers.

**Q3. The reproducibility bug (results saved to a local VM path and lost between sessions) did not corrupt any single notebook's results, yet it is called an important lesson. Why does it matter, and what general principle does it illustrate about experiment infrastructure?**

It matters because it is exactly the class of bug that is invisible until the moment it costs you, and in a longer or higher-stakes project it would cost much more. Within any single notebook the local save worked and the results displayed correctly, so nothing looked wrong; the failure only surfaced when a later session needed the earlier outputs and found them gone. The general principle is that persistence and reproducibility must be verified against the actual boundary they are meant to survive — here, a session restart — not merely observed to "work" within one session. An experiment pipeline should treat its artifacts as needing to outlive the compute that produced them: outputs go to durable storage explicitly, that storage is confirmed by reading the artifacts back in a fresh environment, and dependencies on ephemeral state are removed where possible (as was done by regenerating the heatmaps in-notebook rather than loading them). The deeper lesson is that a result you cannot reload is a result you cannot reproduce, and reproducibility is the property the entire project is organized around.

**Q4. Looking at the final table as a whole, several techniques address "overfitting" and several address "gradient flow." If you had to explain to a newcomer why these are two fundamentally different categories of problem, how would you frame it?**

They live at two different stages of the learning process and have opposite relationships to data. Gradient-flow problems — vanishing gradients, bad initialization, saturation — are *optimisation* failures: they concern whether the network can learn *at all*, whether the training signal physically reaches the parameters that need to change. A network with a vanishing gradient fails on its own training data; more data would not help, because the early layers never receive a signal regardless of how many examples pass through. Overfitting is the opposite situation: the network optimises *too well* on the data it has, fitting its specifics rather than the general pattern, and performs worse on unseen data than on seen data. It is a *generalisation* failure, and it is fundamentally about having too little data relative to model capacity — which is why its cures either constrain the model (dropout, weight decay), enrich the data (augmentation), or import knowledge from more data (transfer learning). The clean framing for a newcomer: gradient-flow fixes make the network *able* to learn; generalisation fixes make sure that what it learns *transfers*. The project's two exposed failures, and the two families of cures, map exactly onto these two stages.

---

*End of Notebook 4 report. The Gradient Autopsy project is complete.*
