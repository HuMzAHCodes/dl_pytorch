# Group 3 — Project Report
### Sensor HAR: Which Recurrent Cell Survives Real Sequences?

**Course:** CS 405 — Deep Learning (NUST/MCS).
**Phase:** Group 3 — Sequence Modeling, group project.
**Framework:** PyTorch · **Platform:** Google Colab (T4 GPU).
**Dataset:** UCI HAR (Human Activity Recognition Using Smartphones).

---

## 1. The question this project answers

Group 3's labs proved, on toy data, that a plain RNN loses the gradient from
distant timesteps while LSTM and GRU do not (Lab A), and that architecture — not
just training quality — decides what a recurrent model can represent (Lab B).
This project tests whether those findings matter on *real* sequences of realistic
length. Four recurrent architectures — SimpleRNN, LSTM, GRU, and bidirectional
LSTM — are trained under identical conditions to classify human activity from
phone-sensor time series, then ranked, and the ranking is explained by a direct
gradient-through-time measurement on the real data.

The project deliberately mirrors the Gradient Autopsy (Groups 1-2): reproduce a
failure on real data, measure it in the gradients before the metric, apply the
fix, and quantify the effect. The only change from Groups 1-2 is the axis — the
vanishing gradient now runs across time rather than across layers.

---

## 2. Data and setup

### 2.1 Dataset choice (a deviation from the original plan)

The plan named the WISDM dataset. WISDM's only first-party download (Fordham
University's site) has been offline for an extended period, and no clean,
no-authentication mirror was available, which would have forced a Kaggle-token
setup that adds friction without pedagogical value. The project therefore uses
the **UCI HAR Smartphones** dataset instead. It serves the identical purpose —
accelerometer- and gyroscope-based human activity recognition — and is in several
respects better suited to the task: it is reliably downloadable without
authentication, it is already windowed into fixed-length sequences, it ships a
predefined train/test split, and its raw inertial signals load directly into the
`(N, timesteps, channels)` shape the recurrent models require. The domain and the
path toward the HAR video final project are unchanged.

### 2.2 Shape and preprocessing

Each example is a 128-timestep window of 9 raw sensor channels (3-axis body
acceleration, 3-axis angular velocity, 3-axis total acceleration). The raw
**inertial signals** are used, not the dataset's 561 engineered features — feeding
hand-engineered features would defeat the purpose of testing whether the networks
can learn temporal structure themselves.

| Split | Shape | Notes |
|-------|-------|-------|
| Train | (7352, 128, 9) | 6 classes, roughly balanced (986-1407 per class) |
| Test  | (2947, 128, 9) | authors' predefined split |

Each channel was standardized to zero mean and unit variance using training
statistics only, to avoid test leakage and to stabilize RNN training.

### 2.3 Controlled comparison and Colab discipline

A single model class, `HARNet`, wraps whichever recurrent cell is named, with an
identical linear classification head reading the final timestep's hidden state
(the concatenation of forward and backward final states for Bi-LSTM). Hidden size
(64), optimizer (Adam), learning rate (1e-3), batch size (128), epoch budget (15),
and seed (0) were all held fixed; the recurrent cell was the single independent
variable. Gradient clipping at norm 5 was applied, and the pre-clip total gradient
norm was logged each epoch.

Every run checkpoints model state, optimizer state, epoch, and full history to
Google Drive after each epoch, and re-running resumes each model where it stopped.
This is the persistence lesson from the Gradient Autopsy applied preventively: a
Colab session timeout mid-sweep costs nothing.

---

## 3. Results

### 3.1 Four-way comparison

| Model | Best test acc | Training behavior |
|-------|---------------|-------------------|
| SimpleRNN | 0.797 | erratic — test accuracy oscillated (0.62, 0.56, 0.75, 0.62, 0.74...), gradient norm spiked to 27, 17, 13 across epochs |
| LSTM | 0.896 | smooth, monotone climb |
| Bi-LSTM | 0.904 | smooth; matched GRU |
| GRU | 0.908 | best; smoothest; smallest gradient norms (~0.4-0.6) |

The headline is not only that the RNN is about ten points worse, but that it is
*unstable*: its accuracy lurched between epochs and its gradient norms alternately
exploded and collapsed, while the gated cells trained calmly. Instability of this
kind is the behavioral fingerprint of gradients that do not flow cleanly through a
long unrolled network.

### 3.2 The mechanism — gradient through time

Each trained model was loaded, its input tensor was made to require gradients, and
a single forward/backward pass from the loss recorded the gradient magnitude at
each of the 128 input timesteps (averaged over a 512-example batch and the 9
channels). A model that routes learning signal to early timesteps shows a high,
flat curve; a model whose signal dies shows a curve that decays toward the early
(distant) timesteps.

| Model | Recent/early gradient ratio | Best test acc |
|-------|-----------------------------|---------------|
| SimpleRNN | 2062.2 | 0.797 |
| LSTM | 14.1 | 0.896 |
| Bi-LSTM | 3.6 | 0.904 |
| GRU | 3.8 | 0.908 |

The ratio measures how much more gradient the most recent ten timesteps receive
than the earliest ten. The plain RNN starves its early timesteps by a factor of
roughly two thousand — the first ~40 timesteps of every sequence contribute almost
nothing to its loss, so it cannot learn from them and survives to ~0.80 only on
the recent-timestep region it can still reach. The gated cells are 150-500x
flatter, keep real signal flowing to the earliest timesteps, and are
correspondingly ~10 points more accurate. The ordering of the gradient ratio
matches the ordering of accuracy almost monotonically: the flattest-gradient
models (GRU, Bi-LSTM) are the most accurate, and the steepest (RNN) is the least.

### 3.3 Interpretation

The two measurements are independent and tell one story. The accuracy table is the
consequence; the gradient-through-time curve is the cause. This is Lab A's
vanishing-gradient-over-time result reproduced on real sensor data, and it is the
Gradient Autopsy's per-layer vanishing-gradient heatmap rotated onto the time
axis. The cure is the same idea in both settings — a gated path (the LSTM/GRU cell
state) that carries the signal across many steps without multiplicative decay.

A secondary observation explains why the RNN does as *well* as it does rather than
collapsing to chance: three of the six activities (sitting, standing, laying)
differ mainly in the direction of gravity, a low-frequency cue carried in the
total-acceleration channels and readable from the recent timesteps the RNN can
still reach. Long-range memory is decisive for the walking variants, not for the
static postures, so the RNN loses ground selectively rather than everywhere.

---

## 4. Engineering notes and lessons

- **cuDNN requires training mode to backprop into inputs.** The gradient-through-
  time measurement initially failed with "cudnn RNN backward can only be called in
  training mode" while the model was in `eval()` mode. Because `HARNet` contains no
  dropout or batch normalization, `train()` and `eval()` produce identical forward
  outputs, so switching to `train()` for the measurement is correct and changes
  nothing about the result. The general lesson: cuDNN's fused RNN kernels only
  support the backward pass in training mode, which matters whenever gradients with
  respect to the input (not just the parameters) are needed for analysis.
- **Checkpoint to Drive, not to the VM.** Applied from the start here, closing the
  persistence gap that cost a session in the Gradient Autopsy.
- **Gradient norm is an early warning.** The RNN's exploding/collapsing per-epoch
  gradient norms flagged its instability before, and more sharply than, its noisy
  accuracy did.

---

## 5. Conceptual & interview questions (advanced)

**Q1. The gradient-through-time curves slope upward toward the recent timesteps for
every model, including GRU and Bi-LSTM. If the gated cells "solve" the vanishing
gradient, why isn't their curve perfectly flat?**

A perfectly flat curve is not the correct expectation, and its absence is not a
failure of the gated cells. Two effects tilt the curve legitimately. First, the
recency of influence is partly a property of the *task and data*, not only the
architecture: the most recent timesteps are genuinely more informative for a
last-step classifier, so some gradient concentration there is appropriate signal,
not decay. Second, even an ideal gated cell has a forget gate strictly below one
in practice, so some gentle discounting of distant steps is expected and even
desirable — total equality would mean the model could never prioritize recent
evidence. What distinguishes vanishing from healthy flow is the *magnitude* of the
falloff: the RNN's 2000x collapse means early timesteps are effectively
disconnected from the loss, whereas the gated cells' 3-14x gradient is a mild
gradient that still carries learnable signal to every timestep. The diagnosis is
the order of magnitude of the ratio, not deviation from a flat line.

**Q2. GRU slightly outperformed LSTM here (0.908 vs 0.896) with a flatter gradient
and far smaller gradient norms. Is that evidence that GRU is the better
architecture, and would you expect it to generalize?**

It is weak evidence for this task, not a general verdict, and it should not be
over-read from a single seed. GRU has fewer gates and parameters than LSTM, which
on a dataset of this size (about 7,000 training sequences) can help: a smaller,
better-conditioned model is easier to optimize and less prone to overfitting when
data is limited, which is consistent with both its smoother training and its edge
in accuracy. But a two-point gap on one seed is within the range that seed
variation could produce, so the honest claim is that GRU and LSTM are
*comparable* here, with GRU marginally ahead and better-behaved. Generalizing the
ranking would require repeating across seeds and ideally across datasets; the
literature shows GRU and LSTM trade places depending on task and data scale, with
LSTM sometimes favored on longer or more complex dependencies where its separate
cell state and output gate help. The defensible conclusion is narrow: on this
dataset, at this size, GRU was at least as good as LSTM and trained more stably.

**Q3. The plain RNN reached 0.797 despite starving its early timesteps by 2000x.
Does that undercut the claim that the vanishing gradient is the reason it
underperforms?**

No — it sharpens the claim rather than undercutting it. The vanishing gradient
does not predict that the RNN fails everywhere; it predicts that the RNN cannot
learn from information that lives far from the output step. The 0.797 is explained
precisely by which parts of the task do and do not need long-range memory: the
static postures (sitting, standing, laying) are separable from the recent-timestep
gravity direction the RNN can still reach, so it classifies those well, while the
dynamic activities that require integrating motion across the whole window are
where it loses ground. If the vanishing gradient were *not* the cause, the RNN's
errors would be spread uniformly and its gradient curve would be flat like the
gated cells'; instead the errors concentrate where long memory is needed and the
gradient curve collapses exactly over the timesteps the model fails to use. The
partial success is therefore consistent with, and diagnostic of, the vanishing-
gradient explanation — it localizes the failure instead of refuting it.

**Q4. Bidirectionality helped dramatically in Lab B (0.756 to 0.983) but barely
moved the needle here (Bi-LSTM 0.904 vs LSTM 0.896). Why the difference?**

Because the two tasks differ in whether the answer depends on future context, and
bidirectionality only pays when it does. Lab B's task was constructed so the label
at each position depended explicitly on the *next* input — a question a forward-
only model structurally cannot see — so adding a backward pass was the difference
between impossible and solved. HAR classification is a single whole-sequence label
produced after the entire window has been read, so a unidirectional model has
already consumed every timestep before it must decide; there is no "future" it is
missing at decision time. The backward pass in the Bi-LSTM then offers only a
modest representational convenience — some features may be easier to extract
reading right-to-left — which is worth a fraction of a point, not a leap. The
general principle: bidirectionality helps per-timestep or fill-in tasks where
future context is causally required, and helps little on sequence-level
classification where the model already sees the whole input before answering.

**Q5. This project ends Group 3. Lab B's encoder-decoder bottleneck pointed at
attention. Does anything in *this* project also motivate attention, or only Lab B?**

This project motivates attention from a second, independent direction. Lab B's
argument was about *capacity* — a single fixed context vector cannot hold a long
input. This project's gradient-through-time curves add the *optimization* argument:
even in the best gated cell, the gradient still has to travel one recurrent step
per timestep to reach a distant input, and the curve's upward tilt shows distant
steps remain somewhat disadvantaged. Attention shortens that path to effectively
one step between any output and any input, regardless of distance, which both
removes the capacity bottleneck Lab B exposed and flattens the gradient-distance
relationship this project measured. So the two Group 3 deliverables converge on the
same successor: Lab B shows attention is needed to carry enough information, and the
project shows it is needed to carry gradient efficiently across long ranges. Group
4 picks up both threads.

---

*End of Group 3 project report. The Group 3 chain — notes, two labs, and this
project — is complete. Next: Group 4 — Attention & Transformers.*
