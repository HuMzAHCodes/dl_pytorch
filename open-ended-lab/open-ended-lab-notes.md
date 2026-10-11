# Open-Ended Lab — Responsible AI · Notes

**The theme of this group:** Groups 1-4 were about making a model that *works* — high
test accuracy. This group is about everything test accuracy hides. A model that scores
0.91 on a held-out set can still be unfair to some users, dangerously overconfident,
oblivious to inputs unlike anything it trained on, and impossible for anyone else to
use safely because nobody wrote down what it does. Responsible-AI auditing is the
discipline of finding those failures *before* deployment does. We run the audit on the
HAR models already built.

Format for every card: **Problem -> Fix -> Why**. Here "Problem" is a way a
high-accuracy model still fails in the real world, "Fix" is how you measure or address
it, and "Why" is the reasoning.

Cards:
1. The deployment gap - why one accuracy number is not enough
2. Fairness - aggregate accuracy hides who the model fails
3. Calibration & uncertainty - confidence that does not match reality
4. Out-of-distribution detection - knowing when you do not know
5. Mitigation - fixing one problem and measuring the change
6. The model card - documenting honestly so others can use it safely

> Spine sentence for this group:
> *This group ends when I can take a trained model and produce an honest account of how
> it behaves beyond its headline accuracy - who it fails, whether its confidence can be
> trusted, whether it notices unfamiliar inputs - fix at least one of those failures,
> and document the result so someone else could deploy it responsibly.*

---

## Card 1 - The deployment gap: one number is not enough

### Problem
Every project so far was judged by a single test-set accuracy. But a test set is a
sample from the same distribution as training, scored in aggregate. That one number
is silent about four things that decide whether a model is safe to deploy: whether it
performs equally across different groups of users, whether its confidence scores mean
anything, whether it can recognise an input unlike its training data, and whether
anyone else knows its limits. A model can be excellent on the test set and still cause
harm on all four.

### Fix
Treat evaluation as a *multi-dimensional audit*, not a scalar. For each deployment
concern there is a concrete measurement: subgroup accuracy (fairness), calibration
error (uncertainty), OOD detection rate (novelty), and a model card (documentation).
This group builds each one.

### Why
- **Why aggregate accuracy hides failures.** An average over a population can stay high
  while a minority subgroup is badly served - the majority dominates the mean. The same
  average says nothing about *how* wrong the model is when it is wrong, or about inputs
  the test set never contained.
- **Why this matters more as stakes rise.** For a toy classifier it is academic; for a
  medical, financial, or safety system, an unaudited model is a liability. The HAR model
  is a useful stand-in: imagine it driving a health-monitoring wearable, where failing
  for certain users or misreading a fall would matter.

**Links forward:** the next four cards are the four axes of the audit.

---

## Card 2 - Fairness: aggregate accuracy hides who the model fails

### Problem
A model's overall accuracy is an average over everyone. If it works well for most users
but poorly for a particular group - a different body type, age, gait, or demographic -
the average can still look excellent while that group is systematically failed. In HAR
the data is collected from multiple human subjects; the model may simply be better for
subjects who resemble the majority of the training set.

### Fix
Measure performance *disaggregated by group* rather than only in aggregate. For HAR the
natural grouping is the subject ID (the dataset records which person each window came
from): compute per-subject or per-group accuracy and look at the spread, especially the
worst-served group, not just the mean. Fairness metrics (accuracy gap, worst-group
accuracy) make the disparity explicit.

### Why
- **Why per-group, not aggregate.** A 91% average could be 97% for most subjects and
  70% for a few; only disaggregation reveals it. The worst-group number is often the one
  that matters for deployment, because it bounds the harm to the least-served user.
- **Why it happens.** Models fit the bulk of the training distribution; underrepresented
  patterns get less signal and are learned worse. This is the Group 2 data-scarcity
  lesson wearing a fairness hat - too little data *for a subgroup* rather than overall.

**Links forward:** fairness asks *who* the model fails. The next card asks whether, when
it is about to fail, it at least *knows* it might.

---

## Card 3 - Calibration & uncertainty: confidence that matches reality

### Problem
A classifier outputs a probability-like score (softmax), and we usually take the top one
as "confidence". But these numbers are often badly *calibrated*: a model can say 99%
and be right only 80% of the time. Worse, modern deep networks tend to be
*overconfident*. A downstream system that trusts the confidence - "only act when the
model is >95% sure" - is then relying on a number that lies.

### Fix
- **Measure calibration.** A reliability diagram bins predictions by confidence and plots
  confidence against actual accuracy in each bin; a perfectly calibrated model lies on
  the diagonal. Expected Calibration Error (ECE) summarises the gap as one number.
- **Quantify uncertainty.** Softmax entropy (spread of the output distribution) is a cheap
  uncertainty signal; Monte-Carlo dropout (run the model several times with dropout on at
  inference and look at the variance) is a stronger one.

### Why
- **Why calibration is separate from accuracy.** A model can rank classes correctly
  (good accuracy) while its probability magnitudes are miscalibrated. Accuracy cares only
  about the argmax; calibration cares about whether the *number* on that argmax is
  truthful - a different property, and the one any confidence-thresholded system depends
  on.
- **Why deep nets are overconfident.** Trained to minimise cross-entropy, they are pushed
  to drive the correct-class probability toward 1, which inflates confidence beyond the
  true hit rate, especially without regularisation. That is why calibration must be
  measured, not assumed.

**Links forward:** calibration is about confidence on *in-distribution* inputs. But the
most dangerous overconfidence is on inputs the model was never trained for at all.

---

## Card 4 - Out-of-distribution detection: knowing when you do not know

### Problem
A classifier trained on six activities will, shown a seventh it has never seen (cycling,
falling, swimming), still output one of its six labels - often with high confidence. It
has no notion of "none of the above". In deployment, inputs that fall outside the
training distribution are inevitable, and silently forcing them into a known class is
exactly how confident, invisible failures happen.

### Fix
Add an **out-of-distribution (OOD) detector**: a score that flags inputs unlike the
training data so the system can abstain or defer to a human instead of guessing. Simple,
strong baselines use the model's own outputs - maximum softmax probability (low max ->
likely OOD) or softmax entropy (high entropy -> likely OOD). You evaluate a detector by
holding out a class during training, calling it "unknown" at test time, and checking
whether the score separates known from unknown (e.g. by AUROC).

### Why
- **Why models cannot abstain by default.** Softmax always normalises over the known
  classes, so it *must* assign all the probability mass to them; there is no "unknown"
  option built in. Over-confident OOD predictions are a structural consequence, not a
  bug to be trained away.
- **Why held-out-class evaluation is the honest test.** You cannot measure OOD detection
  with only in-distribution data. Removing a class from training and reintroducing it as
  "novel" at test time simulates the real situation - an input from a category the model
  never learned - and lets you quantify how well the score catches it.

**Links forward:** cards 2-4 find problems. A responsible audit does not stop at finding
- it fixes at least one.

---

## Card 5 - Mitigation: fix one problem and measure the change

### Problem
An audit that only lists failures is half a job. The point of measuring fairness,
calibration, and OOD behaviour is to *improve* them - and to prove the improvement with
the same measurement, not to assume it.

### Fix
Pick one concrete problem the audit surfaced and apply a targeted mitigation, then
re-measure:

- **Miscalibration -> temperature scaling.** Divide the logits by a single learned
  scalar (temperature) tuned on validation data; it softens overconfident probabilities
  without changing the argmax, so accuracy is untouched while ECE drops. The standard,
  cheap calibration fix.
- **A fairness gap -> reweighting or targeted augmentation** for the underserved group.
- **Weak OOD detection -> a better score** (e.g. energy-based) or a confidence threshold
  tuned on the held-out-class experiment.

### Why
- **Why temperature scaling is the model example.** It is the cleanest demonstration of
  the audit loop: one parameter, no retraining, accuracy preserved, and a clear
  before/after ECE. It shows that calibration is a property you can adjust independently
  of what the model predicts.
- **Why re-measure with the same metric.** A mitigation is a hypothesis; the only honest
  confirmation is the metric moving in the right direction on held-out data. This is the
  controlled-experiment discipline from the whole course, applied to responsible-AI
  properties.

**Links forward:** once measured and (partly) fixed, the model's behaviour has to be
*communicated* - otherwise the next person re-learns all of this the hard way.

---

## Card 6 - The model card: documenting honestly

### Problem
A trained model is an opaque artifact. Its accuracy, its subgroup disparities, its
calibration, its OOD behaviour, its intended use, and its known limits all live in the
head of whoever trained it. Anyone else who deploys it - a teammate, a product team, a
future you - is flying blind, and that is how models get used outside the conditions
they are valid for.

### Fix
Write a **model card**: a short, standardized document that states what the model is and
is not for. A good card covers intended use and out-of-scope use, the training and
evaluation data, aggregate *and* disaggregated performance, calibration and uncertainty
behaviour, known limitations and failure modes (including OOD), and ethical or safety
considerations.

### Why
- **Why a standard document.** Shared structure means a reader knows where to look for
  limitations and intended use, so the information actually gets read and acted on rather
  than buried. Model cards (Mitchell et al., 2019) are now an industry norm for exactly
  this reason.
- **Why it is part of the model, not an afterthought.** The audit results are only useful
  if they travel with the model. The card is where "works at 91% overall but 74% for
  subject X, overconfident until temperature-scaled, cannot detect novel activities"
  becomes a usable warning instead of a surprise in production.

**Links forward (out of this group):** with the audit done and documented, the toolkit is
complete - build a model (Groups 1-4), and evaluate it responsibly (this group). The HAR
video final project applies all of it end to end.

---

## Open-Ended Lab summary - problem -> fix map

| Problem (a way a high-accuracy model still fails) | Fix (how to audit / address it) |
|---------------------------------------------------|---------------------------------|
| One accuracy number hides deployment risks | Multi-axis audit - Card 1 |
| Aggregate accuracy hides who is failed | Disaggregated (per-group) accuracy - Card 2 |
| Confidence scores do not match real accuracy | Reliability diagram + ECE, entropy / MC-dropout - Card 3 |
| Model forces novel inputs into known classes | OOD score (max-softmax / entropy), held-out-class test - Card 4 |
| Finding problems without fixing them | Targeted mitigation + re-measure (temperature scaling) - Card 5 |
| Model behaviour lives only in the trainer's head | Model card - Card 6 |

Unlike Groups 1-4, these cards are not a chain of architecture fixes; they are the four
independent axes of a deployment audit (fairness, calibration, OOD, documentation) plus
the loop that ties them together: measure -> mitigate -> re-measure -> document.
