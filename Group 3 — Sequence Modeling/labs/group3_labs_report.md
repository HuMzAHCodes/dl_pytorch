# Group 3 — Sequence Modeling · Consolidated Lab Report
### Labs A & B

**Course:** CS 405 — Deep Learning (NUST/MCS).
**Phase:** Group 3 — Sequence Modeling (RNN / LSTM / GRU, bidirectional, encoder-decoder).
**Framework:** PyTorch · **Platform:** Google Colab (T4 GPU).

This report covers the two Group 3 labs. Lab A reproduces the core failure of
recurrent networks — the vanishing gradient *over time* — and the LSTM's cure.
Lab B builds the two architectural ideas that sit on top of that fix —
bidirectionality and the encoder-decoder — and ends by exposing the
fixed-context-vector bottleneck that motivates attention in Group 4.

---

## 1. The through-line from Groups 1-2

Group 1 found that gradients die as they propagate backward through the *layers*
of a deep feedforward network: the per-layer gradient heatmap showed early
layers receiving a signal millions of times weaker than late layers. Group 3
takes the same phenomenon and rotates the axis. A recurrent network applies the
same cell repeatedly across *time*, so backpropagation through time (BPTT) is
mathematically a deep composition — one effective "layer" per timestep. The same
repeated multiplication by Jacobian terms that killed the signal across depth now
kills it across time. Lab A measures this directly; the rest of Group 3 is about
architectures that keep the signal alive and move information where it needs to
go.

---

## 2. Lab A — The Vanishing Gradient Over Time (RNN vs LSTM)

### 2.1 Design

The lab isolates one variable: the recurrent cell. A single `SeqClassifier`
wraps either `nn.RNN`, `nn.LSTM`, or `nn.GRU` with identical hidden size, the
same final linear head, and the same training budget; only the cell type
changes. The LSTM additionally initialises its forget-gate bias to 1.0, the
standard trick that holds the gate open at the start of training so the cell
state can carry information before the network has learned to manage it.

Two experiments run on top of this harness. Part 1 is a **memory task**: classify
a sequence by the sign of its *first* element, where every timestep is drawn from
the same uniform distribution so the relevant element cannot be identified by
magnitude — the only way to answer is to carry information from step 0 to the end.
Part 2 is a **deterministic gradient-through-time measurement**: a single
backward pass whose gradient with respect to each input timestep is recorded, so
the signal's decay can be seen without depending on training luck.

### 2.2 Results

**Part 1 (accuracy, noisy by nature).** At sequence lengths 10, 40, and 80 across
three seeds, the LSTM held its accuracy as the memory distance grew while the
plain RNN degraded and became erratic. At T=80 the LSTM averaged roughly 0.85
against the RNN's roughly 0.74, and the RNN's variance across seeds was far
larger — at long ranges it sometimes learned and sometimes did not. This
experiment is reported with its spread rather than a single number, because at
long T the outcome genuinely depends on initialization: the point is the trend
and the instability, not a precise figure.

**Part 2 (the clean proof).** The gradient-through-time measurement is
deterministic and tells the story without ambiguity. For the plain RNN, the
gradient with respect to early timesteps decayed exponentially — from order
1e-4 at the most recent step to order 1e-18 at the most distant — a straight line
on a log axis, which is the signature of repeated multiplication by a factor
below one. For the LSTM (forget gate forced open), the gradient stayed essentially
flat across all timesteps at order 1e-4: the cell state acts as a gradient
highway, and the signal from the first timestep reaches the loss almost
undiminished. Expressed as a ratio of the strongest to the weakest timestep
gradient, the RNN spanned roughly 1.6e8 while the LSTM sat near 1.1.

### 2.3 Reading of Lab A

The accuracy experiment shows the *consequence* (the RNN cannot reliably learn
long-range dependencies); the gradient measurement shows the *mechanism* (the
training signal physically does not reach the distant timesteps). As in the
Gradient Autopsy, the failure is visible in the gradient statistics before, and
more cleanly than, in the task metric. The LSTM's cell state is the fix, and the
flat gradient curve is the direct evidence that it works.

---

## 3. Lab B — Bidirectional Context & the Encoder-Decoder Bottleneck

### 3.1 Part 1 — Bidirectionality

A per-timestep tagging task was constructed so the correct label at position t
depends on position **t+1**: the label is 1 when the next value exceeds the
current one. A left-to-right model has not seen t+1 when it must answer at t, so
it is structurally handicapped; a bidirectional model, which runs a forward and a
backward pass and concatenates their hidden states, can see both sides.

**Result:** unidirectional LSTM **0.756**, bidirectional LSTM **0.983**, with
everything else held fixed. The unidirectional model is not badly trained — it is
near the ceiling of what is achievable without future information (it exploits the
present value's distribution and the fixed last-position label). The gap to the
bidirectional model is the value of genuinely seeing t+1, and it is an
architectural gap, not an optimisation one.

### 3.2 Part 2 — The encoder-decoder bottleneck

An encoder GRU compresses an entire input sequence into its final hidden state —
a single fixed-size context vector — and a decoder GRU generates the output from
that vector alone, one token at a time, feeding each produced token back in. The
task was **copy-reverse**: output the input digit sequence in reverse, a pure test
of whether the context vector can hold and recall every input token in order.
Training used teacher forcing at ratio 0.5; evaluation was free-running (ratio
0.0), so the decoder consumed its own predictions exactly as it would at
inference.

**Result (sweep over sequence length):**

| T  | per-token accuracy | exact-sequence accuracy |
|----|--------------------|-------------------------|
| 5  | ~0.98              | ~0.95                   |
| 10 | ~0.57              | ~0.02                   |
| 20 | ~0.33              | ~0.00                   |

For short sequences the single vector has room and the model nearly solves the
task. As length grows the early input tokens are crushed and exact-sequence
accuracy falls off a cliff at T=10, reaching zero by T=20. Per-token accuracy
fades more gently because the model still recalls the tail of the input (nearest
the context vector) and gets lucky on some positions — but a correct *whole*
sequence requires every token at once, which the bottleneck makes impossible.

### 3.3 Reading of Lab B

The two parts cover the two ways information fails to reach where it is needed.
Bidirectionality is about *direction* — making future context available at all.
The encoder-decoder bottleneck is about *capacity* — a single fixed vector cannot
scale with sequence length. The collapse curve in Part 2 is the exact problem
attention is designed to remove: instead of one context vector, let the decoder
attend to every encoder state at every step. Group 3 therefore ends pointing
directly at Group 4.

---

## 4. The problem -> fix chain so far

| Problem exposed | Fix | Evidence in these labs |
|-----------------|-----|------------------------|
| Gradient vanishes across time (RNN) | LSTM cell state (gradient highway) | grad ratio 1.6e8 -> ~1.1; flat vs exponential-decay curve |
| Model cannot see future context | Bidirectional RNN | 0.756 -> 0.983 on the t+1 task |
| One fixed context vector cannot scale | (attention — Group 4) | exact-seq acc 0.95 -> 0.00 as T grows |

---

## 5. Conceptual & interview questions (advanced)

### Lab A

**Q1. Why does the plain RNN's gradient-through-time curve come out as an almost
perfectly straight line on a log axis, and what does the slope encode?**

Because backpropagation through time multiplies the gradient by a recurrent
Jacobian term once per timestep it travels back. If that term has magnitude
roughly r at each step, then after k steps the gradient has been scaled by about
r^k, and taking the log turns that product into k·log(r) — linear in the number
of steps. A straight line on a log axis is therefore the visual signature of a
constant per-step multiplicative factor, and its slope is log(r): a steep
negative slope means r is well below one (vanishing), a flat line means r is near
one (healthy), and a positive slope would mean r above one (exploding). The RNN's
steep negative slope is a direct read-out of how far below one its effective
recurrent gain sits.

**Q2. The LSTM's gradient stayed flat only after you forced the forget-gate bias
to hold the gate open. Doesn't that make the demonstration a rigged result rather
than a property of the LSTM?**

No — it makes it a *controlled* result that isolates the mechanism. The claim
being demonstrated is that the cell state provides an undecayed path for the
gradient *when the forget gate is open*; forcing the gate open tests exactly that
claim, free of the confound that an untrained LSTM has random gate values and so a
random, uninformative gradient profile. In normal use the network *learns*
appropriate forget-gate values during training, and the forget-bias-of-1.0
initialisation exists precisely so the gate starts near-open and the highway is
available before the network has learned to manage it. So the demonstration shows
the architectural capability the LSTM is built to exploit; the training run in
Part 1 shows it actually being exploited. Showing the mechanism under controlled
conditions and the consequence under real training is stronger than conflating
the two.

**Q3. Part 1 accuracy was noisy and you reported it with its spread, while Part 2
was deterministic. Why keep a noisy experiment in the lab at all rather than rely
only on the clean one?**

Because the two answer different questions and the noisy one is closer to what
practitioners actually care about. Part 2 proves the *mechanism* — the signal
physically decays — cleanly and reproducibly, but a gradient magnitude is not a
task outcome. Part 1 shows the *consequence* on the thing that matters: whether
the model can learn a long-range dependency, and crucially that at long ranges the
RNN's success becomes initialization-dependent — sometimes it learns, often it
does not. That instability is itself a finding, and it would be hidden by
reporting only a single lucky or unlucky run, or by dropping the experiment for a
cleaner one. Reporting the spread honestly conveys "this architecture is
unreliable here," which is the practical lesson; the deterministic measurement
explains *why*. Keeping both is the same discipline as the Gradient Autopsy:
measure the mechanism cleanly, but also show it biting on a real task.

### Lab B

**Q4. The unidirectional LSTM scored 0.756 on the future-context task, not 0.5.
If it genuinely cannot see t+1, why is it well above chance?**

Because the task is not purely random with respect to the information the model
*does* have. Two effects lift it above a coin flip without ever seeing the future.
First, the last position is always labelled 0 (no successor exists), which is
learnable from position alone and is free accuracy. Second, the forward model can
exploit the present value: since the next value is an independent uniform draw, a
very large current value is more likely to be followed by a smaller one (label 0)
and a very small one by a larger one (label 1), so x[t] itself carries a
statistical hint about the label. That hint comes from the present, not the
future, so it buys accuracy above 0.5 without answering the real question. The
bidirectional model clears 0.98 because it reads the actual answer at t+1.

**Q5. In the seq2seq sweep, per-token accuracy at T=20 was about 0.33 while
exact-sequence accuracy was 0.00. Reconcile those two numbers and say which is the
more honest headline.**

They measure different things and the gap is diagnostic. Per-token accuracy is the
fraction of individual positions correct; exact-sequence accuracy requires every
position correct simultaneously. At T=20 the model still recovers about a third of
individual tokens — plausibly the tail of the input, freshest in the context
vector, plus some lucky guesses — but a correct whole sequence is roughly the
product of twenty per-position chances, so even 0.33 per token collapses exact
accuracy to essentially zero. For a task whose output is only useful if entirely
correct (reversing a list, translating a sentence), exact-sequence accuracy is the
honest headline; per-token accuracy flatters the model by awarding partial credit
for an output that, as a whole, is wrong. The divergence also tells us the failure
is partial recall — the vector remembers the end and forgets the beginning — not a
total breakdown.

**Q6. You trained with teacher forcing at ratio 0.5 but evaluated with it off. Why,
and how could that choice flatter the model if you were not careful?**

Teacher forcing feeds the ground-truth previous token into the decoder during
training instead of its own prediction. Early in training the decoder's outputs
are garbage, and feeding garbage back as input yields no useful learning signal,
so teacher forcing stabilises and speeds up training. But it creates a train/test
mismatch known as exposure bias: at inference there is no ground truth, the
decoder must consume its own predictions, and a single early error can derail the
remainder because mistakes compound down the sequence. If evaluation also used
teacher forcing, each step would be handed the correct history regardless of the
model's own errors, hiding exactly this failure mode and making the model look far
better than it is. Evaluating free-running (ratio 0.0) measures the model as it
will actually be used, which is why the bottleneck numbers in this lab are
trustworthy rather than inflated.

**Q7. The fix for the bottleneck is attention. Precisely what does attention change
about the information path, and why does that also help with the gradient problem
from Lab A?**

Attention removes the single fixed-size context vector as the sole bridge between
encoder and decoder. Instead of compressing the whole input into one vector, the
decoder at each output step computes a learned weighted combination of *all*
encoder hidden states, so information from any input position can flow directly to
any output position. Two consequences follow. First, capacity no longer bottlenecks
on one vector, so performance stops collapsing as sequence length grows — the Part
2 failure disappears. Second, and connecting back to Lab A, the path length between
two related tokens stops growing with their distance: a direct attention link is
effectively one step regardless of how far apart the tokens are, whereas BPTT had
to traverse one recurrent step per unit of distance. Shorter gradient paths mean
far less of the repeated-multiplication decay that produced the straight-line
vanishing curve, which is a large part of why attention-based models train on long
sequences where recurrent models struggle.

---

*End of Group 3 Labs A & B report. Next: the Group 3 project (sensor-based HAR on
WISDM), then Group 4 — Attention & Transformers.*
