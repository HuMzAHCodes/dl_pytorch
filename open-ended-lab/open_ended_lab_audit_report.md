# Open-Ended Lab — Responsible-AI Audit Report
### Auditing the HAR Transformer

**Course:** CS 405 — Deep Learning (NUST/MCS).
**Phase:** Open-Ended Lab — Responsible AI.
**Framework:** PyTorch · **Platform:** Google Colab (T4 GPU).
**Subject model:** the HAR Transformer from the Group 4 project (UCI HAR Smartphones).

---

## 1. Purpose

Groups 1-4 were about building a model that *works* — the Group 4 Transformer reached
~0.91 test accuracy on UCI HAR. This lab asks the questions that single accuracy number
cannot answer, because they are the questions that decide whether a model is safe to
deploy: does it work for *everyone*, is its *confidence* trustworthy, can it recognise
inputs *unlike* its training data, and is any of this *written down* so others can use
it safely. The audit runs four measurements — fairness, calibration, out-of-distribution
behaviour, and documentation — plus one mitigation, on the exact model shipped by the
Group 4 project.

The framing in one sentence: **test accuracy is a scalar that hides everything that
makes a model safe or unsafe.** This report is the multi-dimensional picture it hides.

---

## 2. How the subject model was reused (saved in Group 4, loaded here)

This lab did not retrain the classifier it audits. It reuses the exact trained model
from the Group 4 project, which matters for a responsible audit: you want to audit the
*deployed* weights, not a fresh model that might behave differently.

**How it was saved (Group 4 project).** During training, after every epoch, the project
wrote a checkpoint to Google Drive with `torch.save`, bundling the model weights, the
optimizer state, the epoch index, and the full metric history into one file:

```python
torch.save({"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "history": history},
           os.path.join(CHECKPOINT_DIR, "har_transformer_seed0.pt"))
```

Because `CHECKPOINT_DIR` points at a mounted Google Drive path
(`/content/drive/MyDrive/har_project`), the file persists after the Colab session ends —
unlike the local VM, which is wiped. That is the persistence lesson from the Gradient
Autopsy, applied so a trained model survives across sessions and notebooks.

**How it was loaded (this lab).** The audit notebook rebuilds the *identical* architecture
(same `HARTransformer` class, same hyperparameters), then restores the saved weights into
it:

```python
model = HARTransformer().to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()
```

`state_dict` is just a dictionary mapping parameter names to tensors, so loading only
works if the architecture matches the one that produced it exactly — same layer names
and shapes. The lab therefore re-declares the model definition rather than importing a
differently-shaped one. The result is that the audited model is byte-for-byte the model
the project produced (overall test accuracy ~0.907 on load), and the lab includes a
fallback that trains a fresh model only if the checkpoint is missing. Separating training
(Group 4) from evaluation (this lab) through a saved checkpoint is itself good practice:
it makes the audit reproducible and guarantees we are auditing the real artifact.

---

## 3. Part 1 — Fairness: who does the model fail?

UCI HAR records which of its 30 subjects each window came from (9 subjects appear in the
test split). Overall accuracy averages over all of them, so it can look strong while a
particular person is poorly served. Disaggregating by subject:

| Metric | Value |
|--------|-------|
| Overall accuracy | 0.907 |
| Best subject (24) | 0.995 |
| Worst subject (9) | 0.712 |
| **Fairness gap (best − worst)** | **0.283** |

The headline 0.907 conceals a 28-point spread. Subject 9 is classified correctly only
71% of the time, and subject 10 is close behind (~0.75), while most subjects sit near or
above 0.95. The aggregate hides this because the well-served majority dominates the mean;
a few poorly-served subjects barely move it. For a deployed wearable this is the number
that matters — harm concentrates on the worst-served user, not on the average one.

---

## 4. Part 2 — Calibration: is the confidence honest?

To avoid tuning the fix on the data used to score it, the test set was split into a
*calibration half* (for fitting the temperature in Part 3) and an *evaluation half* (for
measuring calibration). A reliability diagram bins predictions by confidence and plots
confidence against actual accuracy; Expected Calibration Error (ECE) is the average gap.

**Result:** ECE = 0.049, with the reliability bars sitting *below* the diagonal — in the
high-confidence bins the model's accuracy is lower than its stated confidence. The model
is **overconfident**, the characteristic failure of deep networks trained with
cross-entropy, which pushes the correct-class probability toward 1 and inflates confidence
past the true hit rate. Any downstream rule of the form "act only when the model is >95%
sure" would be trusting numbers that overstate the truth.

---

## 5. Part 3 — Mitigation: temperature scaling

Temperature scaling divides the logits by a single scalar T, fit by minimising
cross-entropy on the calibration half, before softmax. T > 1 softens overconfident
probabilities. Because dividing all logits by the same constant never changes which is
largest, the predicted class — and therefore accuracy — is unchanged.

| Metric | Before | After |
|--------|--------|-------|
| Fitted temperature | — | 1.763 |
| ECE (evaluation half) | 0.049 | 0.021 |
| Accuracy (evaluation half) | 0.904 | 0.904 |

The fitted temperature of 1.763 ( > 1, confirming overconfidence) more than halved the
ECE, moving the reliability bars onto the diagonal, while accuracy stayed at exactly
0.904. This is the cleanest demonstration of the audit loop — measure, mitigate,
re-measure — and shows calibration is a property you can fix independently of what the
model predicts. **If confidence scores will be used downstream, the temperature-scaled
version should be deployed.**

---

## 6. Part 4 — Out-of-distribution detection: knowing when it does not know

A classifier trained on six activities will force a seventh into one of its six labels.
To test this honestly, one activity (walking_downstairs) was held out of training
entirely; a 5-class model was trained on the rest, and at test time the held-out
activity's windows were presented as "novel". A good model should be less confident on
the novel activity, so its max-softmax probability and entropy should separate known from
novel. Separation is scored by AUROC (1.0 perfect, 0.5 chance).

| Metric | Value |
|--------|-------|
| Mean confidence, known activities | 0.965 |
| Mean confidence, novel activity | 0.965 |
| OOD AUROC (max-softmax) | 0.588 |
| OOD AUROC (entropy) | 0.589 |

This is the audit's most important finding. The model is **exactly as confident on an
activity it has never seen as on ones it knows** (0.965 vs 0.965), and the confidence
distributions overlap almost completely, so softmax-based detection barely beats chance
(AUROC ~0.59). Because walking_downstairs resembles the other walking classes, the model
confidently files it under a known label. This is the textbook "confident, invisible
failure": softmax normalises over the known classes only, so it has no outlet for "none
of the above" and was never trained to be uncertain off-distribution. A model that looks
91% accurate will mislabel a genuinely novel activity without hesitation, and its
confidence score gives almost no warning. A dedicated OOD method or a conservative
confidence threshold would be required before trusting predictions in the open world.

---

## 7. Part 5 — Model card

### Model Card: HAR Transformer (UCI HAR Smartphones)

**Model.** A from-scratch Transformer encoder (linear input projection + learnable CLS
token + sinusoidal positional encoding + 2 blocks of 4-head self-attention with
feed-forward, residuals and layer norm; ~68k parameters) classifying 128-timestep,
9-channel accelerometer/gyroscope windows into 6 activities.

**Intended use.** Educational and research classification of pre-segmented smartphone
inertial windows into: walking, walking upstairs, walking downstairs, sitting, standing,
laying.

**Out-of-scope use.** Any safety-, health-, or finance-critical decision; real-time
streaming without re-validation; populations unlike the 30 young-adult subjects in UCI
HAR; and any activity outside the six trained classes (see OOD below).

**Training & evaluation data.** UCI HAR Smartphones, authors' train/test split (7,352 /
2,947 windows), 30 subjects, raw inertial signals, per-channel standardised on training
statistics.

**Performance (aggregate).** Overall test accuracy 0.907 (0.913 best during Group 4
training).

**Performance (disaggregated) — FAIRNESS.** Per-subject accuracy ranges from 0.712
(subject 9) to 0.995 (subject 24): a fairness gap of 0.283. The model is substantially
worse for some subjects than the aggregate suggests.

**Calibration.** Overconfident out of the box (ECE 0.049). Temperature scaling (T = 1.763)
reduces ECE to 0.021 with no change in accuracy. Deploy the temperature-scaled version if
confidence scores are used.

**OOD behaviour.** On a held-out novel activity (walking_downstairs), max-softmax and
entropy separate known from novel inputs with AUROC 0.588 and 0.589 — barely above chance.
Mean confidence on novel inputs equals that on known inputs (0.965). The model will
confidently misclassify novel activities; an explicit OOD check or confidence threshold is
required before trusting a prediction in the open world.

**Known limitations.** Large per-subject disparity (worst subject 0.712); raw
overconfidence; near-blind to novel activities; validated only on short, pre-segmented
windows from a narrow subject pool.

**Ethical & safety considerations.** The fairness gap means deployment would serve some
users markedly worse than others; the OOD blindness means unfamiliar behaviour is labelled
confidently rather than flagged. Both must be addressed before any consequential use.

---

## 8. Conclusion

The same model that scores 0.907 overall is, on closer audit, unfair to some subjects
(gap 0.283), overconfident (ECE 0.049), and effectively unable to notice a novel activity
(OOD AUROC ~0.59). One of these — miscalibration — was mitigated cleanly by temperature
scaling, halving ECE at zero accuracy cost. The other two are documented as known
limitations in the model card so they travel with the model rather than surfacing in
production. This is the lesson of the group: a single accuracy number is the beginning of
evaluation, not the end, and responsible deployment requires measuring, fixing where you
can, and honestly documenting what remains.

---

## 9. Conceptual & interview questions (advanced)

**Q1. Temperature scaling cut ECE in half but left accuracy exactly unchanged. Why can
accuracy not change, and why is that a feature?**

Dividing every logit by the same positive constant is a monotonic transformation that
preserves their order, so the largest logit stays largest and the predicted class is
identical for every input — accuracy is mathematically unchanged. This is a feature
because calibration and accuracy are different properties: accuracy is about which class
is picked, calibration about whether the confidence on that pick is truthful. Temperature
scaling fixes the second without touching the first, so honest probabilities come at no
accuracy cost. A method that changed predictions would entangle the two concerns and risk
accuracy to fix confidence, which is exactly what you do not want.

**Q2. The fairness check found subject 9 at 0.712 versus 0.995 for subject 24, with 0.907
overall. Why did the aggregate hide this, and what is the danger of reporting only the
aggregate?**

The aggregate is a weighted average over all test windows; the well-served majority
dominates it and the few poorly-served subjects barely move it, so 0.907 looks strong and
the disparity is invisible. The danger is that deployment harm is not distributed like the
average — it concentrates on the worst-served group. A wearable shipped on "91% accurate"
would work poorly for people like subject 9, and the strong headline number would stop
anyone from looking. Reporting worst-group accuracy and the gap alongside the mean is what
makes that harm visible before it reaches users.

**Q3. OOD detection reused the model's own softmax outputs. What is the fundamental
limitation of that, and why does holding out a class give an honest estimate?**

Softmax is normalised over the known classes only, so it always distributes all
probability mass among them and has no "none of these" outlet; a confident peak can appear
even for an unseen category, because the network was never trained to be uncertain
off-distribution and the novel input's features may align with a known class. The audit
showed exactly this — identical confidence on known and novel inputs, AUROC ~0.59. So
softmax confidence is a heuristic OOD signal, not a calibrated one, and can fail silently.
Holding out a class gives an honest estimate because it recreates the real deployment
situation — an input from a category absent at training time — and measures, on genuinely
unseen data, whether the score separates known from novel. Testing OOD on perturbed
in-distribution data instead would be an easier problem and would overstate reliability.

**Q4. Walking_downstairs was a poor choice if the goal were to make OOD detection look
good, but arguably a good choice for an honest audit. Explain.**

Walking_downstairs resembles the other walking activities, so the model's learned features
place it near known classes and it is assigned a known label with high confidence — which
is precisely why detection scored near chance. A more distinct held-out activity (say, one
with a very different motion signature) would have produced a lower-confidence response and
a flatteringly high AUROC. For an honest audit the harder, more realistic case is the right
one: novel inputs in deployment will often be *near-neighbours* of known classes, not
exotic outliers, and those are exactly the cases where confident misclassification is most
likely and most dangerous. Reporting the easy case would overstate how safe the model is;
the near-neighbour case reveals the real weakness.

---

*End of Open-Ended Lab audit report. The four-group architecture arc plus this responsible-AI
audit complete the course toolkit. Remaining: the HAR video final project.*
