# Group 4 — Attention & Transformers · Notes

**The theme of this group:** Group 3 gave networks memory, but the memory was *sequential* - each
timestep waited for the previous one (slow, no parallelism), long-range dependencies still strained the
gradient, and the seq2seq context vector was a bottleneck. This group throws out recurrence entirely.
Instead of carrying information forward step by step, we let every position **look directly at every
other position** - that is attention, and stacking it is the Transformer: the architecture behind every
modern large language model.

Format for every card: **Problem -> Fix -> Why**.

Cards:
1. Attention - let the decoder look at all encoder states, not one context vector
2. Query / Key / Value and scaled dot-product - how attention actually computes
3. Self-attention - a sequence attending to itself, with no recurrence
4. Multi-head attention - several attention "views" at once
5. Positional encoding - putting order back in
6. The Transformer block - attention + feed-forward + residual + layer norm
7. Encoder-decoder Transformer and masked attention - the full architecture

> Spine sentence for Group 4:
> *This group ends when I can build a Transformer from scratch - attention replaces recurrence, so every
> position reaches every other in one hop (no vanishing gradient over distance), the whole sequence is
> processed in parallel (no sequential bottleneck), and there is no single context vector to squeeze
> through. With this, the architecture toolkit is complete - what remains is using it responsibly
> (Open-Ended Lab) and on real video (HAR final project).*

---

## Card 1 - Attention: stop squeezing everything through one vector

### Problem
In the Group 3 encoder-decoder, the decoder sees the entire input only through the encoder's final
hidden state - one fixed-size context vector. The project proved this collapses as sequences grow:
copy-reverse accuracy went from ~0.95 at length 5 to ~0.00 at length 20. Early input tokens get crushed
because one vector cannot hold an arbitrarily long input. The decoder needs access to the *whole* input,
not a lossy summary of it.

### Fix
**Attention** lets the decoder, at each output step, look back at *all* of the encoder's hidden states
and take a **learned weighted average** of them. The weights (how much to attend to each input position)
are computed fresh for every output step, so different output tokens can focus on different parts of the
input.

```
for each decoder step:
    score each encoder state against the current decoder state   # relevance of each input position
    softmax the scores -> attention weights (sum to 1)
    context = weighted sum of encoder states                     # a *different* context every step
```

### Why
- **Why it removes the bottleneck.** Information no longer has to pass through a single fixed vector.
  Every input position remains directly reachable; the decoder pulls exactly what it needs, when it
  needs it. The copy-reverse task becomes easy because the decoder can attend straight to the input
  position it is currently reversing.
- **Why a *weighted* average and not just "pick one".** The weights are produced by softmax, so the
  whole operation is smooth and differentiable - the model can *learn* where to look by gradient
  descent, instead of making a hard, non-differentiable choice.
- **Why this also helps gradients.** An output position can connect to any input position in a single
  attention step, so the gradient path between related tokens no longer grows with their distance - the
  Group 3 project's gradient-through-time tilt flattens out.

**Links forward:** so far attention still sits on top of RNNs (decoder attends to encoder). The next
cards ask: what *exactly* is being scored, and can we drop the RNN entirely?

---

## Card 2 - Query, Key, Value and scaled dot-product attention

### Problem
Card 1 said "score each encoder state against the current decoder state" but left *score* vague. We need
a concrete, learnable, efficient way to compute how relevant each position is to each other position.

### Fix
Cast attention as a **soft dictionary lookup** with three learned projections of the inputs:

- **Query (Q)** - what the current position is looking for.
- **Key (K)** - what each position offers, for matching against queries.
- **Value (V)** - the actual content each position contributes if attended to.

The relevance of a query to a key is their **dot product**; softmax turns the scores into weights; the
output is the weighted sum of values:

```
Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V
```

The `sqrt(d_k)` divisor (d_k = key dimension) is the **scaling** that gives the mechanism its name:
*scaled* dot-product attention.

### Why
- **Why separate Q, K, V.** Splitting "what I'm looking for" (Q), "what I can be matched on" (K), and
  "what I pass along" (V) lets the model learn each role independently - far more expressive than
  matching raw vectors against themselves.
- **Why the dot product.** It measures alignment between two vectors cheaply and in parallel for all
  pairs at once (one matrix multiply Q K^T), which is what makes attention fast on modern hardware.
- **Why divide by sqrt(d_k).** For large key dimension, dot products grow large in magnitude, pushing
  softmax into saturated regions where one weight is ~1 and the rest ~0 - and there the gradient nearly
  vanishes. Dividing by sqrt(d_k) keeps the scores at a sane scale so softmax stays soft and trainable.
  (A direct echo of the Group 1/2 lesson about keeping activations and gradients well-scaled.)

**Links forward:** nothing in this formula requires Q, K, V to come from *different* sequences. Let them
all come from the *same* sequence, and attention becomes something more powerful.

---

## Card 3 - Self-attention: a sequence attending to itself

### Problem
RNNs are inherently sequential: timestep `t` cannot be computed until `t-1` is done. That means no
parallelism (slow training) and a path length between two positions that grows with their distance
(the root of vanishing gradient over time). We want to relate any two positions directly and compute
them all at once.

### Fix
**Self-attention**: derive Q, K, and V all from the *same* input sequence, so every position attends to
every other position in that sequence. Position `i`'s output is a weighted blend of the values at all
positions, weighted by how well `i`'s query matches each position's key. No recurrence, no hidden state
passed along - the whole sequence is transformed in one parallel operation.

### Why
- **Why this kills the sequential bottleneck.** Every position's output is computed from the full
  sequence simultaneously - a few big matrix multiplies - so the entire layer runs in parallel instead
  of one-step-at-a-time. This is the single biggest reason Transformers train so much faster than RNNs
  on long sequences.
- **Why this kills vanishing-gradient-over-distance.** The path between any two positions is now
  *constant* - one attention hop - regardless of how far apart they are. Compare the Group 3 project,
  where the gradient had to survive one recurrent step per timestep of distance. Constant path length
  means the gradient no longer decays with distance.
- **The cost (the honest tradeoff).** Every position attends to every other, so compute and memory grow
  as the *square* of sequence length (O(n^2)). RNNs were linear in length. Transformers buy parallelism
  and direct long-range links at the price of quadratic cost - which is why very long sequences are a
  live research problem.

**Links forward:** one set of Q/K/V projections gives the model a single way to relate positions. Why
settle for one?

---

## Card 4 - Multi-head attention: several views at once

### Problem
A single self-attention computes one set of attention weights - one notion of "what's relevant to what".
But positions relate in many ways at once: in a sentence, one relationship is grammatical (subject-verb),
another is semantic (pronoun-referent), another positional. One attention map cannot capture all of them.

### Fix
**Multi-head attention** runs several self-attention operations ("heads") in parallel, each with its own
learned Q/K/V projections into a smaller subspace. Each head produces its own output; the heads are
concatenated and linearly projected back to the model dimension.

```
head_i = Attention(Q W_q^i, K W_k^i, V W_v^i)       # each head: its own subspace, its own weights
MultiHead = concat(head_1, ..., head_h) W_o
```

### Why
- **Why multiple heads help.** Each head can specialize in a different kind of relationship - one tracks
  short-range syntax, another long-range coreference, another position - and together they give a richer,
  multi-faceted representation than any single head could.
- **Why smaller subspaces per head.** Each head projects into dimension d_model/h, so h heads cost about
  the same as one full-dimension attention - you get diversity of views essentially for free, not at h
  times the cost.
- **Why concatenate then project.** The final projection W_o lets the model mix the heads' outputs into a
  single representation, learning how to combine the different relationship types.

**Links forward:** self-attention is permutation-blind - shuffle the inputs and the outputs just shuffle
with them. For sequences, that is a problem.

---

## Card 5 - Positional encoding: putting order back in

### Problem
An RNN processes left to right, so order is baked into the computation. Self-attention has no such
notion: it treats its input as an unordered *set* - "the cat sat" and "sat the cat" would produce the
same set of attention outputs. But in sequences, order *is* meaning. We removed recurrence and
accidentally threw away position with it.

### Fix
Add a **positional encoding** to each input embedding - a vector that depends on the position index -
before the first attention layer, so each token's representation carries *where* it is as well as *what*
it is. The original Transformer uses fixed sinusoids of different frequencies; modern variants often use
*learned* positional embeddings instead.

```
input_to_layer = token_embedding + positional_encoding(position)
```

### Why
- **Why addition works.** The position vector shifts each token's embedding in a position-dependent
  direction, so otherwise-identical tokens at different positions become distinguishable, and attention
  can learn to use relative position in its scores.
- **Why sinusoids (the original choice).** Different frequencies let the model represent a wide range of
  distances, and the smooth sinusoidal structure makes relative offsets easy to express - the encoding
  for position `p+k` is a fixed linear function of the encoding for `p`, so "k positions apart" is
  learnable. They also extrapolate to sequence lengths not seen in training.
- **Why this matters conceptually.** It's the price of giving up recurrence: order was free in an RNN;
  in a Transformer you must inject it explicitly. A clean example of a tradeoff, not a free lunch.

**Links forward:** we now have the complete attention machinery. Wrapping it into a reusable layer needs
two more ingredients - both of which we already met in Groups 1-2.

---

## Card 6 - The Transformer block: attention + feed-forward + residual + norm

### Problem
A single attention layer mixes information *across positions*, but it does no heavy per-position
processing, and stacking many attention layers directly would reintroduce the deep-network training
problems from Group 1 (vanishing gradients, unstable training). We need a block that is both expressive
and deeply stackable.

### Fix
The **Transformer block** combines four pieces into one reusable unit, stacked N times:

1. **Multi-head self-attention** - mixes information across positions.
2. **Position-wise feed-forward network** - a small MLP applied to each position independently, for
   per-token nonlinear processing.
3. **Residual (skip) connections** around each sub-layer.
4. **Layer normalization** on each sub-layer.

```
x = x + MultiHeadSelfAttention(x)      # attention sub-layer, with a residual around it
x = LayerNorm(x)
x = x + FeedForward(x)                  # feed-forward sub-layer, with a residual around it
x = LayerNorm(x)
```

### Why
- **Why residual connections (straight from Group 1/2).** They give the gradient a direct path around
  each sub-layer, so a deep stack of blocks trains without the signal vanishing - the exact skip-
  connection idea that healed the deep CNN, now holding up a deep Transformer. Attention solved
  vanishing over *distance*; residuals solve vanishing over *depth* of the stack.
- **Why layer norm (not batch norm).** Normalizing each token's features keeps activations well-scaled
  (the Group 2 BatchNorm lesson), stabilizing training. Layer norm is used because it normalizes per
  token, independent of batch size and sequence length - essential when sequences vary in length.
- **Why the feed-forward sub-layer.** Attention is mostly a weighted *average* of values - a linear mix.
  The per-position MLP adds the nonlinear, per-token transformation that gives the block its
  representational punch. Attention moves information between positions; the FFN processes it.

**Links back:** two of the four ingredients (residuals, normalization) are direct reuses of the Group 1/2
gradient-flow cures - the same fixes, holding up a new architecture.

**Links forward:** stacking these blocks gives an encoder. Generation needs one more idea.

---

## Card 7 - Encoder-decoder Transformer and masked attention

### Problem
Self-attention lets every position see every other - including positions *to its right*. That is fine
when encoding a known input, but fatal when *generating*: predicting the next token must not be allowed
to peek at the answer (the future tokens), or the model cheats at training and has nothing to generate
from at inference.

### Fix
Two ideas complete the architecture:

- **Masked (causal) self-attention** in the decoder: before softmax, set the attention scores for all
  future positions to minus infinity, so each position can attend only to itself and earlier positions.
- **Cross-attention**: decoder layers also attend to the *encoder's* outputs (Q from the decoder, K and V
  from the encoder) - this is Card 1's original decoder-to-encoder attention, now inside the Transformer.

The full **encoder-decoder Transformer**: an encoder stack builds rich representations of the input; a
decoder stack generates the output using masked self-attention (to respect causality) plus cross-attention
(to read the input).

### Why
- **Why masking.** It enforces the causal constraint - position `t` depends only on positions `<= t` -
  so the model learns a genuine next-token predictor that works identically at training and inference.
  (This is the honest, no-teacher-forcing-cheat version of the Group 3 Lab B decoder.)
- **Why keep an encoder and a decoder.** The encoder can use full bidirectional self-attention (it sees
  the whole input at once - no causality needed), while the decoder must be causal. Separating them lets
  each do the right thing; cross-attention is the bridge.
- **Why this is the endpoint.** Encoder-only stacks (e.g. BERT-style) are for understanding; decoder-only
  stacks (e.g. GPT-style) are for generation; the full encoder-decoder is for sequence-to-sequence like
  translation. All three are just arrangements of the same block from Card 6 - which is why building one
  Transformer from scratch teaches the whole family.

**Links forward (out of Group 4):** the architecture toolkit is now complete - from the perceptron to the
Transformer. What remains is not a new architecture but two applications of everything so far: making
these models **responsible** (fairness, uncertainty, out-of-distribution behavior - the Open-Ended Lab),
and applying the full stack to **real video** (the HAR final project: CNN features + sequence model +
attention).

---

## Group 4 summary - problem -> fix map

| Problem | Fix (card) |
|---------|------------|
| Seq2seq squeezes everything through one context vector | Attention over all encoder states - Card 1 |
| "Score relevance" needs a concrete, learnable form | Query/Key/Value + scaled dot-product - Card 2 |
| RNNs are sequential and long-range paths are long | Self-attention (parallel, constant path) - Card 3 |
| One attention map captures only one relationship | Multi-head attention - Card 4 |
| Self-attention ignores word order | Positional encoding - Card 5 |
| A deep stack of attention won't train on its own | Transformer block (residual + layer norm + FFN) - Card 6 |
| Generation must not peek at future tokens | Masked attention + encoder-decoder - Card 7 |
| (no new architecture) | **Responsible AI + HAR video final project** |

Every card answers a problem the previous one exposed - the same chain structure as Groups 1-3. Note how
often the cures are Group 1/2 ideas returning: scaling (sqrt d_k), residual connections, normalization.
The Transformer is not a break from everything before it - it is those lessons, reassembled around
attention.
