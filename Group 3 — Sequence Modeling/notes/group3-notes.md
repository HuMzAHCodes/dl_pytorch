# Group 3 — Sequence Modeling · Notes

**The theme of this group:** everything in Groups 1-2 assumed inputs are *independent* (one image,
one label, order irrelevant). This group handles **sequential** data - text, speech, sensor streams,
time series - where the meaning of the present depends on the past. We give networks *memory*.

Format for every card: **Problem -> Fix -> Why**.

Cards:
1. RNN - the recurrent cell, hidden state, and backprop-through-time
2. Vanishing / exploding gradient over time - the problem this group must solve
3. LSTM - gates and a cell state to remember long-range context
4. GRU - a simpler gated cell
5. Deep RNNs - stacking recurrent layers
6. Bidirectional RNNs - reading the sequence both ways
7. Encoder-decoder (seq2seq) - variable-length input to variable-length output

> Spine sentence for Group 3:
> *This group ends when I can model sequences with memory (LSTM/GRU/bidirectional/seq2seq) - but the
> models are inherently sequential (slow, no parallelism) and still strain on very long dependencies,
> and the seq2seq context vector is a bottleneck - which is exactly what Group 4 (attention/
> transformers) solves.*

---

## Card 1 - RNN: giving a network memory

### Problem
A feed-forward network (ANN/CNN) maps one input to one output with no notion of order or history.
Feed it a sentence and it treats each word independently; feed it a sensor reading and it has no idea
what came before. But in sequential data the *order* carries the meaning: "dog bites man" vs "man
bites dog" are the same words in a different order. We need a network whose output at each step
depends on what it has already seen.

### Fix
A **Recurrent Neural Network (RNN)** processes a sequence one timestep at a time, carrying a **hidden
state** `h` that acts as memory. At each step it combines the current input with the previous hidden
state:

```
h_t = activation(W_x * x_t + W_h * h_{t-1} + b)      # update memory with the new input
y_t = W_y * h_t                                      # output (optional, per step)
```

The *same* weights (`W_x`, `W_h`, `W_y`) are reused at every timestep - weight sharing across time,
just as a CNN shares weights across space. Training uses **backpropagation through time (BPTT)**:
unroll the loop over the sequence and backpropagate through every step.

### Why
- **Why the hidden state works.** `h_{t-1}` is a compressed summary of everything seen so far, so
  `h_t` lets the past influence the present. The network builds up context as it reads the sequence.
- **Why shared weights across time.** The rule for "how to update memory given a new input" should be
  the same at every position - so one set of weights is learned and applied at all timesteps. This
  keeps the parameter count small and lets the RNN handle sequences of any length.
- **Why BPTT.** The loss at a late timestep depends on early inputs *through* the chain of hidden
  states, so the gradient must flow backward through every step - the same chain-rule idea as normal
  backprop, applied along the time axis instead of the layer axis.

**Links forward:** that backward flow through many timesteps is a long chain of multiplications - which
is exactly where the next card's problem comes from.

---

## Card 2 - Vanishing / exploding gradient over time: the problem this group solves

### Problem
BPTT backpropagates the gradient through every timestep, and - just like depth in Group 1 - a long
chain means a long product of terms. If those terms are consistently < 1 the gradient **vanishes**
before it reaches the early timesteps; if consistently > 1 it **explodes**. The practical consequence:
a plain RNN effectively has a *short memory*. It learns short-range dependencies fine but forgets
long-range context - by the time it reaches the end of a long sentence, the gradient signal connecting
it back to the start has died.

This is the Group 1 vanishing gradient, moved from the **layer axis** to the **time axis**. (A nice
callback to the Gradient Autopsy project: same mechanism, different direction.)

### Fix
- **For exploding gradients:** **gradient clipping** - cap the gradient norm at a threshold so it
  can't blow up. Simple and effective.
- **For vanishing gradients:** this is the hard one, and it needs an architectural fix - a cell that
  can *carry information across many steps without repeated shrinking*. That is exactly what LSTM and
  GRU (the next cards) provide.

### Why
- **Why it happens specifically over long sequences.** Short sequences have short chains - little
  shrinkage. Length is what turns "each term slightly less than 1" into "product ~ 0", so long-range
  dependencies are the ones that suffer.
- **Why clipping fixes exploding but not vanishing.** Clipping caps a gradient that is *too large*;
  it can't resurrect a gradient that has already shrunk to zero. Vanishing needs a path along which
  the signal is *preserved*, not merely bounded - hence an architectural change rather than a
  numerical patch.

**Links forward:** LSTM and GRU exist precisely to give the gradient a protected path through time.

---

## Card 3 - LSTM: a cell that remembers

### Problem
A plain RNN's hidden state is overwritten at every step (`h_t` is a fresh function of `h_{t-1}`), so
information from long ago is repeatedly transformed and gradually washed out - the vanishing-gradient-
over-time problem from Card 2. We need a memory that can *hold* information across many steps
unchanged when needed, and only update it deliberately.

### Fix
The **Long Short-Term Memory (LSTM)** cell adds a separate **cell state** `C` - a conveyor belt that
runs straight through time with only minor, gated interactions - plus three **gates** (each a small
sigmoid network outputting values in [0,1]) that control information flow:

- **Forget gate** - what to erase from the cell state.
- **Input gate** - what new information to write into the cell state.
- **Output gate** - what to expose from the cell state as this step's hidden output.

```
the cell state C_t is updated as:  C_t = forget_gate * C_{t-1} + input_gate * candidate
the hidden output is:              h_t = output_gate * tanh(C_t)
```

### Why
- **Why the cell state fixes vanishing gradients.** The cell state is updated mostly by *addition*
  (gated add) rather than repeated multiplication by weight matrices. Addition doesn't shrink the
  gradient the way a long product does, so the signal can flow across many timesteps largely intact -
  a protected gradient highway through time. This is the core idea.
- **Why gates.** Multiplying by a value in [0,1] is a soft on/off switch. The forget gate can choose
  to keep a memory unchanged (multiply by ~1) for as long as it stays relevant, then erase it
  (multiply by ~0) when it's no longer needed. The network *learns* when to remember and when to
  forget, rather than being forced to overwrite every step.
- **Why three gates.** They separate the decisions: what to drop, what to add, and what to reveal -
  giving fine control over the memory. This is what lets LSTMs capture long-range structure that
  plain RNNs cannot.

**Links back:** this is the architectural cure for Card 2's vanishing gradient over time.

**Links forward:** LSTMs work well but are relatively heavy (three gates, a cell state - many
parameters). Can we get most of the benefit more cheaply? That's the next card.

---

## Card 4 - GRU: the lighter gated cell

### Problem
The LSTM solves the memory problem but at a cost: three gates plus a separate cell state means a lot
of parameters and computation per step. For many tasks that's more machinery than necessary.

### Fix
The **Gated Recurrent Unit (GRU)** simplifies the LSTM to **two gates** and no separate cell state:

- **Update gate** - how much of the past to keep vs how much new information to let in (it merges the
  LSTM's forget and input gates into one decision).
- **Reset gate** - how much of the past to forget when computing the new candidate state.

The hidden state itself carries the memory; there's no distinct cell state.

### Why
- **Why it works nearly as well as LSTM.** The update gate still provides the key ingredient - a gated,
  mostly-additive path that lets information (and gradients) persist across time - so it keeps the
  vanishing-gradient cure. It just packages the control into fewer gates.
- **Why fewer parameters is a real advantage.** Fewer gates means faster training, less memory, and
  less overfitting on smaller datasets - which matters when data is limited (a recurring theme from
  Group 2).
- **The honest tradeoff (LSTM vs GRU).** Neither is universally better. GRU is simpler and often
  trains faster; LSTM's extra expressiveness sometimes wins on very long or complex sequences. In
  practice you try both - which is exactly what the Group 3 project does.

**Links forward:** LSTM and GRU fix *memory*. The remaining cards add *capacity* and *context* on top.

---

## Card 5 - Deep (stacked) RNNs: more capacity

### Problem
A single recurrent layer can only learn so complex a mapping. Just as one conv layer isn't enough for
hard image tasks, one RNN layer isn't enough for hard sequence tasks - it can't build up hierarchical
temporal features.

### Fix
**Stack recurrent layers**: the sequence of hidden states output by layer 1 becomes the input sequence
to layer 2, and so on. Lower layers learn short-term, local patterns; higher layers combine them into
longer-term, more abstract structure.

### Why
- **Why depth helps here too.** The same reason it helped in CNNs: hierarchy. Early layers capture
  fine temporal detail, deeper layers capture broader patterns built from that detail. More
  representational power for complex sequences.
- **The tradeoff.** More layers means more parameters, slower training, and more overfitting risk -
  so depth is added only when the task needs it, and paired with the Group 2 regularizers (dropout
  between recurrent layers is standard).

**Links forward:** depth adds capacity, but every RNN so far only reads the sequence *forward* - it
only knows the past. Sometimes the future matters too.

---

## Card 6 - Bidirectional RNNs: using future context

### Problem
A standard RNN processes left-to-right, so its state at position `t` only reflects inputs up to `t` -
the past. But for many tasks the *right* context disambiguates the present. In "I read the book" vs
"I read the newspaper", the word after "read" settles its meaning; a forward-only model hasn't seen
it yet when processing "read".

### Fix
A **Bidirectional RNN** runs *two* RNNs over the sequence - one forward, one backward - and combines
their hidden states at each position (usually by concatenation). So each position's representation is
informed by **both** the past (forward pass) and the future (backward pass).

### Why
- **Why two directions help.** Each output now has access to the entire sequence - full context on
  both sides - instead of only what preceded it. For labeling tasks (sentiment, named-entity
  recognition, activity classification from a full window) this is a clear win.
- **The key limitation (and when you can't use it).** Bidirectional requires the *whole sequence to
  be available up front*, so it cannot be used for real-time or generative tasks where you produce
  output as input arrives (e.g. live transcription, next-word generation) - there is no "future" to
  read yet. It's for *classifying or labeling a complete sequence*, not for streaming generation.

**Links forward:** all of the above handle *one sequence*. But many tasks map an input sequence to a
*different* output sequence of possibly different length - translation, summarization. That needs a
new structure.

---

## Card 7 - Encoder-decoder (seq2seq): sequence in, sequence out

### Problem
Classification gives one label for a whole sequence; per-step tagging gives one label per input step.
But translation ("how are you" -> "comment allez-vous") maps a sequence to a *different* sequence of
*different length*, where the alignment between input and output positions isn't one-to-one. No single
RNN structure so far handles variable-length-in to variable-length-out.

### Fix
The **encoder-decoder (sequence-to-sequence)** architecture uses two RNNs:

- The **encoder** reads the entire input sequence and compresses it into a fixed-size **context
  vector** (its final hidden state) - a summary of the whole input.
- The **decoder** takes that context vector as its starting state and generates the output sequence
  one step at a time, feeding each generated token back in as the next input.

### Why
- **Why split into two.** Decoupling reading from writing lets the input and output be different
  lengths and different languages/modalities. The encoder's only job is to understand; the decoder's
  only job is to generate.
- **Why the context vector is the crux - and the bottleneck.** The decoder sees the *entire* input
  only through that one fixed-size vector. For short sequences that's fine, but for long inputs it's
  like summarizing a whole paragraph in a single sentence and then writing a translation from only
  that summary - early details get squeezed out. This **information bottleneck** is the central
  weakness of plain seq2seq.

**Links forward (out of Group 3):** the sequential nature of RNNs (each step waits for the previous
one - no parallelism, slow) and the seq2seq context-vector bottleneck are the two problems that
**attention** solves - by letting the decoder look back at *all* the encoder's hidden states directly,
and by removing recurrence so the whole sequence is processed in parallel. That is Group 4:
**Attention and Transformers.**

---

## Group 3 summary - problem -> fix map

| Problem | Fix (card) |
|---------|------------|
| Feed-forward nets have no memory of order | RNN + hidden state - Card 1 |
| Gradient vanishes/explodes across timesteps | Clipping (explode) + gated cells (vanish) - Card 2 |
| Plain RNN forgets long-range context | LSTM (cell state + 3 gates) - Card 3 |
| LSTM is heavy | GRU (2 gates, lighter) - Card 4 |
| One recurrent layer lacks capacity | Deep/stacked RNNs - Card 5 |
| Forward-only misses future context | Bidirectional RNN - Card 6 |
| Variable-length input -> variable-length output | Encoder-decoder (seq2seq) - Card 7 |
| Context vector bottleneck + no parallelism | **Attention / Transformers - Group 4** |

Every card answers a problem the previous one exposed - the same chain structure as Groups 1-2, now
along the time axis.
