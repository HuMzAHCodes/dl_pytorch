# Group 4 — Attention & Transformers · Consolidated Lab Report
### Labs A & B

**Course:** CS 405 — Deep Learning (NUST/MCS).
**Phase:** Group 4 — Attention & Transformers (labs).
**Framework:** PyTorch · **Platform:** Google Colab (T4 GPU).

This report covers the two Group 4 labs. Lab A builds scaled dot-product attention
from scratch and shows it defeats recurrence on a long-range lookup — and that its
weights are interpretable. Lab B adds the rest of the Transformer machinery —
positional encoding, multi-head attention, and the full block — on a task where
order matters, exposing exactly why each piece is needed. Everything is written by
hand: no `nn.Transformer`, no `nn.MultiheadAttention`.

---

## 1. The through-line from Group 3

Group 3 ended on a wall the recurrent models could not clear: information and
gradient both decay with distance (the HAR project's recent/early gradient ratio,
and the copy-reverse collapse), and a single context vector cannot scale. Group 4's
answer is to stop carrying information forward step by step and instead let every
position look at every other position directly. Lab A isolates that idea in its
simplest form; Lab B assembles it into a real Transformer block, and in doing so
shows that two of the block's four ingredients — residual connections and
normalization — are the Group 1/2 gradient-flow cures returning to hold up a deep
attention stack. Attention cures vanishing gradients across *distance*; residuals
and layer norm cure them across *stack depth*.

---

## 2. Lab A — Attention from Scratch: the Long-Range Lookup

### 2.1 Design

Scaled dot-product attention, `softmax(QK^T / sqrt(d_k)) V`, was implemented
directly and sanity-checked (output shapes correct, each query's weights sum to 1).
The test task is a **content-based recall**: a sequence of filler tokens with a
single "marked" token dropped at a random position; the label is the marked token's
value. Solving it requires locating the marked token — anywhere in the sequence —
and reading it. A one-query ("CLS-style") attention classifier was compared against
an LSTM and a plain RNN across sequence lengths. No positional encoding was used,
deliberately: the task is purely content-addressed, so position is irrelevant to
finding the token.

### 2.2 Results

**Accuracy vs sequence length (chance = 0.111):**

| T  | attention | LSTM  | RNN   |
|----|-----------|-------|-------|
| 10 | 1.000     | 1.000 | 1.000 |
| 30 | 1.000     | 1.000 | 0.349 |
| 60 | 1.000     | 0.902 | 0.236 |

**Attention localization (T = 50):** the trained model placed, on average, ~0.97 of
its attention weight exactly on the marked position, and in 100% of test examples
the single most-attended position was the true marked position — the attention
weight spiked on the target token wherever it sat.

### 2.3 Reading of Lab A

Attention's accuracy is flat in sequence length because the path between the query
and any position is a single hop, independent of distance. The LSTM must carry the
marked value in its hidden state across the intervening steps, so it degrades as
sequences grow (the Group 3 vanishing-gradient-over-time problem); the plain RNN
collapses almost immediately. The localization result turns the accuracy number into
a mechanism: the model is not pattern-matching, it is explicitly looking at the token
it needs. The cost of attention's constant path, not visible on this small task, is
its O(n^2) compute in sequence length.

---

## 3. Lab B — Order Matters: Positional Encoding, Multi-Head Attention & the Transformer Block

### 3.1 Design

The task is order-sensitive: the label is 1 if the **first** token's value exceeds
the **last** token's value. A sequence and its first-last swap share the same bag of
tokens but carry opposite labels, so the task cannot be solved without position
information. Three components were built from scratch: sinusoidal positional encoding,
multi-head self-attention (split into heads, scaled dot-product per head, concatenate,
project), and a Transformer block (multi-head attention and a position-wise
feed-forward network, each inside a residual connection followed by layer
normalization). An encoder was assembled as embedding (+ optional positional encoding)
-> N Transformer blocks -> mean-pool -> classify, and four configurations were
compared.

### 3.2 Results

**Ablation ladder (chance = 0.500):**

| Configuration | Test accuracy |
|---------------|---------------|
| no positional encoding, 4 heads | 0.506 |
| with positional encoding, 1 head | 0.952 |
| with positional encoding, 4 heads | 0.976 |
| with positional encoding, 4 heads, 2 blocks | 0.995 |

**Head specialization (with positional encoding, 4 heads):** averaging each head's
attention over many examples, head 0 concentrated on position 0 (~0.78 of its weight
on the first token) and head 3 on the last position (~0.89 on the last token), while
the remaining heads were split or diffuse. The model, unprompted, dedicated one head
to each of the two tokens the task compares.

### 3.3 Reading of Lab B

The ladder isolates each idea's contribution. Without positional encoding the model
is at chance — not merely worse, but mathematically pinned, because self-attention
followed by mean-pooling is permutation-invariant while the label is not. Adding
positional encoding breaks that symmetry and the accuracy jumps to 0.95. Multi-head
and depth each add a few points on top. The head-specialization result explains the
multi-head gain concretely: the task needs two specific positions, and separate heads
let the model extract both cleanly in parallel. That the roles emerged from gradient
descent, rather than being designed, is the reason interpreting real Transformer
heads is an empirical exercise.

---

## 4. The problem -> fix chain for Group 4

| Problem | Fix | Evidence in these labs |
|---------|-----|------------------------|
| Recurrence loses distant information | Attention (one-hop path to any position) | recall acc flat at 1.00 vs RNN 0.24 at T=60 |
| "Relevance" needs a concrete form | Scaled dot-product Q/K/V | attention weight ~0.97 on the target token |
| Self-attention ignores order | Positional encoding | 0.506 (no PE) -> 0.952 (PE) |
| One attention map, one relationship | Multi-head attention | 0.952 (1 head) -> 0.976 (4 heads); heads split on the two endpoints |
| Deep attention stacks won't train alone | Residual + layer norm (the block) | 2 blocks reach 0.995; same cures as Groups 1-2 |

---

## 5. Conceptual & interview questions (advanced)

**Q1. The attention model stayed at 1.000 accuracy from length 10 to 60 while the
LSTM fell off. What property of attention causes accuracy to be flat in sequence
length, and what does it cost?**

The path length between the query and any input position is constant — a single
attention step — regardless of how far apart they are. The query scores every
position directly and in parallel, so a token 60 steps away is no harder to reach
than one 2 steps away. The LSTM must carry the marked value in its hidden state across
every intervening step, and that memory degrades with distance (the Group 3
vanishing-gradient-over-time problem), so its accuracy falls as sequences grow. The
cost of attention's constant path is quadratic compute and memory: scoring every
position against every other is O(n^2), where the RNN was O(n). Attention buys
distance-independence and parallelism at the price of scaling badly in sequence
length.

**Q2. Lab A used no positional encoding and still hit 100%. Would adding it help, and
when would its absence break a task?**

It would not help on Lab A's task and could even distract, because that task is purely
content-addressed: the marked token is identifiable by its embedding alone, so the
query can find it by content without knowing any position. Positional encoding becomes
essential the moment the answer depends on where a token is or on the order of tokens —
which is exactly Lab B's task. There, self-attention followed by mean-pooling is
permutation-invariant, so without position information a sequence and its reordering
produce identical outputs while their labels differ, pinning accuracy at chance. The
contrast between the two labs is the cleanest possible statement of when position
information matters: content-addressed retrieval does not need it; anything
order-dependent cannot work without it.

**Q3. The Lab A query is a single learned vector, not derived from the input. How does
this differ from full self-attention, and why was the simpler version enough?**

In full self-attention every position produces its own query from its own embedding, so
the layer re-represents every position in light of all the others (an N-to-N operation).
Lab A used one fixed learned query shared across all inputs, so the layer produces a
single pooled vector (N-to-1) — essentially a learned, content-based pooling. It sufficed
because the task needs exactly one piece of information retrieved once (the marked value),
so a single query that learns "look for the token in the marked range" is enough. Full
self-attention is needed when every position must be re-represented for downstream
layers or when many relationships must be computed across the sequence at once — which is
what Lab B's Transformer block does.

**Q4. In Lab B, the no-positional-encoding model sat at 0.506 on a task the same
architecture solves at 0.95+ with PE. Explain why it is pinned at chance, not merely
"lower".**

Self-attention is permutation-equivariant: permute the input positions and the outputs
permute identically, with the same values. Mean-pooling over positions is then
permutation-invariant, so the model's prediction is literally unchanged under any
reordering of the input. But the label — is the first value greater than the last — flips
under reordering, and a sequence and its first-last swap have the same bag of tokens with
opposite labels. The model is mathematically forced to give both the same prediction, so
it cannot beat the base rate of 0.5. This is not a trainability issue that more epochs or
capacity could fix; it is an invariance the architecture imposes until positional
encoding breaks the symmetry.

**Q5. The Lab B heads specialized — one on the first position, one on the last. Was the
model told to do this, and what made this the solution it found?**

Nothing told it to; the specialization is emergent. The label depends on exactly two
positions, so a representation that cleanly extracts the first and last tokens makes the
final linear comparison trivial, and any configuration producing that representation gets
low loss. Multiple heads with independent Q/K/V projections give the model the freedom to
route one head to position 0 and another to the last position via their positional
encodings, so gradient descent naturally pushes it there. Head roles are a consequence of
task structure and optimization pressure, not a designed assignment — which is why
interpreting the heads of a real Transformer is an empirical exercise: you discover what
they learned, you cannot read it off the architecture.

**Q6. The Transformer block wraps attention and the feed-forward network each in
"residual then layer norm". Both first appeared in Groups 1-2 for CNNs. Why do they
reappear here, and what would a deep stack of attention blocks do without them?**

They reappear because the problem they solve — training signal degrading through many
layers — is about depth in general, not convolution, and a Transformer is a deep stack.
Residual connections give the gradient a direct additive path around each sub-layer, so a
deep stack of blocks trains without the gradient vanishing as it backpropagates through
every attention and feed-forward layer — the same skip-connection cure that healed the
deep CNN in the Gradient Autopsy. Layer normalization keeps each token's activations
well-scaled from layer to layer, preventing the drift and saturation that destabilize
deep training — the BatchNorm lesson of Group 2, in the per-token variant that does not
depend on batch or sequence-length statistics. Without them a deep stack would be badly
conditioned: gradients would shrink or explode across layers and activations would drift,
so it would train poorly past a few blocks. Attention solves vanishing gradients across
sequence distance; residuals and layer norm solve them across stack depth. Both are
needed, and both are reused from earlier groups.

---

*End of Group 4 Labs A & B report. Next: the Group 4 project — a Transformer encoder
applied to the UCI HAR data, head-to-head against the Group 3 recurrent models.*
