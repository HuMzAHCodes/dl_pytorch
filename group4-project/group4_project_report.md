# Group 4 — Project Report
### Transformer HAR: Attention vs Recurrence on Real Sensor Data

**Course:** CS 405 — Deep Learning (NUST/MCS).
**Phase:** Group 4 — Attention & Transformers, group project.
**Framework:** PyTorch · **Platform:** Google Colab (T4 GPU).
**Dataset:** UCI HAR (Human Activity Recognition Using Smartphones) — the same data as the Group 3 project.

---

## 1. The question this project answers

The Group 3 project ranked four recurrent cells on UCI HAR and showed the plain RNN
starved its early timesteps (recent/early gradient ratio ~2000x) and underperformed.
This project removes recurrence entirely: a Transformer encoder, assembled from the
from-scratch components built in Labs A and B, classifies the same 128-timestep
sensor windows. Because the dataset, task, and train/test split are identical, the
comparison against the Group 3 recurrent numbers is apples-to-apples. The project
also produces something the recurrent models cannot: an attention saliency map
showing which parts of the motion window the classifier used.

---

## 2. Architecture

The encoder reuses the hand-built pieces from Lab B and adapts them for continuous
sensor input:

- **Input projection.** A linear layer lifts the 9 sensor channels to a 64-dim model
  space. There is no token embedding — the input is already continuous, so projection
  replaces lookup.
- **CLS token.** A single learnable vector is prepended to the sequence; its final
  representation is what gets classified, and its attention over the 128 timesteps is
  the saliency map. This is the BERT/ViT pooling pattern.
- **Positional encoding.** Sinusoidal encoding over the 129 positions (CLS + 128
  timesteps), added to the projected input.
- **Transformer blocks.** Two blocks, each multi-head self-attention (4 heads) plus a
  position-wise feed-forward network, every sub-layer wrapped in a residual connection
  and layer normalization.
- **Classifier.** A linear layer on the CLS token's final representation.

The model has roughly 68k parameters — small, which turns out to matter for a dataset
of only 7,352 training windows.

Training mirrored the Group 3 project: Adam at 1e-3, 20 epochs, batch size 64, seed 0,
with model and history checkpointed to Drive after every epoch and resumable on a
timeout.

---

## 3. Results

### 3.1 Head-to-head with the recurrent models

| Model | Best test accuracy |
|-------|--------------------|
| SimpleRNN | 0.797 |
| LSTM | 0.896 |
| Bi-LSTM | 0.904 |
| GRU | 0.908 |
| **Transformer** | **0.913** |

The Transformer took the top spot, narrowly ahead of GRU. The margin over the gated
cells is small — this dataset is friendly to RNNs, being small and short-sequenced —
but the Transformer matched and slightly beat them while training in parallel, with no
recurrence and no vanishing gradient over distance.

### 3.2 Attention saliency over the motion window

Averaging the CLS token's attention per activity revealed a clear, interpretable split:

- **Static postures (sitting, standing)** concentrated attention on a few discrete
  timesteps — sharp bright spikes at specific positions — as if the model probes a
  handful of moments to confirm a steady pose.
- **Dynamic activities (walking and its variants)** spread attention across the window
  in an oscillating pattern, consistent with reading the periodic stride signal rather
  than any single instant.
- **Laying** drew low, nearly flat attention everywhere — minimal motion, so no
  particular timestep is decisive.

This is a genuine window into the model's reasoning, and it aligns with the physics of
the activities: periodic motion is distributed in time, a static pose is not.

---

## 4. Reading of the project

Accuracy alone makes the Transformer and the gated RNNs look interchangeable here, but
they differ on the axes that matter at scale. The Transformer trains in parallel across
timesteps (no sequential unroll), connects any two positions in one hop (no
vanishing gradient over distance — the exact failure measured in the Group 3 project),
and is interpretable through its attention. The RNNs win on efficiency at this scale:
linear rather than quadratic compute in sequence length, and a stronger sequential
inductive bias that makes them competitive on short windows with limited data. The
honest conclusion is that the Transformer is the more scalable and more interpretable
architecture, and that on this particular small, short-sequence task that scalability
buys only a narrow accuracy edge — the gap would be expected to widen on longer
sequences and larger datasets, which is exactly the regime the final HAR video project
moves toward.

The project also closes the Group 3 -> Group 4 arc cleanly: the same dataset that
exposed recurrence's limits is now solved by the architecture built to replace it,
and the attention maps make concrete what "the model looks at the whole sequence
directly" means.

---

## 5. Engineering notes

- **CLS-token pooling** gives a single, clean query whose attention is the saliency
  map; mean-pooling would have classified fine but scattered the interpretability
  across all positions.
- **Attention extraction was batched** (chunks of 512) when reading the full test set,
  because materializing the last block's full attention tensor for all 2,947 test
  windows at once is memory-heavy.
- **Checkpoint to Drive every epoch** — the Group 3 persistence lesson, applied from
  the start.

---

## 6. Conceptual & interview questions (advanced)

**Q1. The Transformer reached 0.913 despite Transformers being "data-hungry" and this
dataset being small. Why did it do well, and does that contradict the claim?**

It does not contradict it. The data-hungry reputation is about scale relative to task
difficulty: large language and vision Transformers must learn rich, general
representations from scratch, and attention's weak inductive biases (no built-in
locality or order) mean they need a lot of data to discover structure that a CNN or RNN
assumes for free. HAR is a narrow, easy task — six physically distinct activities,
short fixed windows, strong low-level cues (periodicity for walking, gravity direction
for static postures) — so a small two-block encoder has ample signal, and positional
encoding plus CLS pooling supply the bit of order-awareness needed. The result is
consistent with the claim: on a small, well-posed problem a small Transformer suffices,
and data-hungriness only bites when the representation to be learned is large and
general.

**Q2. Accuracy is nearly tied with the RNNs, yet we call the Transformer more capable
here. On what non-accuracy axes does it win, and where do the RNNs win?**

The Transformer trains in parallel across timesteps (no sequential dependency), uses
the GPU more fully, scales better to longer sequences, connects any two positions in
one hop (no vanishing gradient over distance — the Group 3 failure), and is
interpretable via its attention map. The RNNs win on efficiency at this scale: compute
linear in sequence length versus attention's quadratic cost, and a stronger sequential
inductive bias that makes them smaller and faster to reach the same accuracy on short
128-step windows with limited data. The Transformer is the more scalable and
interpretable choice; the RNN is the leaner one when sequences are short and data is
limited.

**Q3. The saliency is read from the CLS token's attention weights. What does that map
show, what does it not show, and how could it mislead?**

It shows, per timestep, how much weight the CLS token placed on that timestep in the
final block when forming the vector that was classified — where the pooled decision
drew its information. It does not show a causal account: attention weight is correlation
between the CLS query and each key, not proof those timesteps changed the output, and
information from unattended timesteps can still reach the CLS token through earlier
blocks that already mixed positions. It can mislead by making the model look sharply
focused when the computation was distributed — averaging over heads and examples can
manufacture a clean peak, and residual connections let the CLS token carry information
forward without re-attending, so low attention on a timestep does not mean it was
irrelevant. Attention maps are a useful hint, not a faithful explanation — a caution
that applies to real Transformers as much as to this one.

**Q4. Why prepend a CLS token at all, rather than mean-pooling the timestep outputs and
classifying that?**

Both work, and mean-pooling would likely reach similar accuracy; the CLS token is
chosen mainly for a cleaner readout and interpretability. A dedicated learnable token
gives the model a single slot whose only job is to aggregate whatever is relevant for
classification, and whose attention over the sequence is a ready-made, interpretable
saliency map — one query, one map. Mean-pooling forces equal contribution from every
position before the classifier sees them, which discards the model's own opinion about
which positions matter and spreads any interpretability across all positions. The CLS
design lets the aggregation itself be learned and inspected, which is why it is the
standard choice in BERT and ViT.

**Q5. The attention maps differed by activity — sharp spikes for static postures,
distributed attention for walking. Is this evidence the model "understands" the
activities, or could it arise from something shallower?**

It is suggestive but not proof of understanding, and the shallower reading is likely
closer to the truth. The pattern aligns with the physics — periodic motion is spread in
time, a static pose is not — which is encouraging, but the model has no concept of
"walking"; it has learned which timesteps carry the features that best separate the
classes under the training objective, and for a static posture a few samples of the
gravity direction suffice, so attention collapses onto a few points, while for walking
the discriminative signal is the periodicity, which is distributed. So the maps reflect
where the discriminative signal lives for each class, not semantic understanding. The
distinction matters: the saliency is real and useful for debugging and trust, but
reading it as comprehension would over-claim what a correlation-based attention weight
can establish.

---

*End of Group 4 project report. With this, the four-group architecture arc is complete
— perceptron to Transformer. Next: the Open-Ended Lab (responsible-AI audit) and the
HAR video final project.*
