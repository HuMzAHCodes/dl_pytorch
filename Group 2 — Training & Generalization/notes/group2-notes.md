# Group 2 — Making Networks Train Well · Notes

**The theme of this whole group:** every card here is a *fix* for a problem Group 1 exposed.
Group 1 built a network and watched it fail — vanishing gradients, overfitting, slow training.
Group 2 is the toolkit that makes deep networks actually trainable.

Format for every card: **Problem → Fix → Why**.

Cards:
1. Activation functions (ReLU, Leaky ReLU, SELU) — the vanishing-gradient fix
2. Weight initialization (Xavier, He) — starting in a healthy place
3. Batch normalization — keeping activations healthy *during* training
4. Optimizers (SGD → Momentum → RMSProp → Adam) — smarter, faster descent
5. Regularization (dropout, L2 weight decay, early stopping) — the overfitting fix
6. Data augmentation — the "not enough data" fix
7. Transfer learning — standing on a pretrained model's shoulders

> Spine sentence for Group 2:
> *This group ends when the network from Group 1 trains deep, fast, and generalizes —
> but it still can't handle ordered/sequential data, which is what Group 3 solves.*

---

## Card 1 — Activation functions: the vanishing-gradient fix

### Problem
Group 1 Lab A showed it directly: with **sigmoid/tanh**, the gradient shrinks as it backprops
through layers. Their derivatives are small (sigmoid's max is 0.25) and drop to ~0 when the neuron
**saturates** (input very large or very small). Multiply many small numbers across layers →
the gradient vanishes → early layers stop learning.

### Fix
Use activations whose gradient doesn't shrink the signal:

- **ReLU** `f(x) = max(0, x)`. The default. Derivative is exactly **1** for all positive inputs
  (and 0 for negative). No shrinking on the positive side.
- *(addition)* **Leaky ReLU** `f(x) = x if x>0 else 0.01x`. Fixes ReLU's one weakness (below).
- *(addition)* **SELU / ELU** — smooth variants that can self-normalize; useful in deep plain nets.

### Why
- **Why ReLU fixes vanishing gradients.** Its derivative is 1 on the positive side, so backprop
  multiplies by 1, not by 0.25. The signal passes through many layers without collapsing.
- **Why ReLU is also fast.** `max(0, x)` is trivial to compute — no exponentials like sigmoid.
- **The one catch — "dying ReLU."** If a neuron's input is always negative, its gradient is always
  0 and it never recovers — it's dead. **Leaky ReLU** fixes this by giving a small slope (0.01) for
  negative inputs, so a little gradient always flows.
- **Why not just always use the fancy ones?** ReLU is simple, fast, and works in the vast majority
  of cases. Reach for Leaky ReLU/SELU when you actually see dying neurons or need self-normalization.

**Links back:** this is the *first* fix we apply to the broken Group 1 network. Swap sigmoid → ReLU,
re-plot the per-layer gradient norms, and the staircase flattens.

**Links forward:** activations set the stage, but *how you initialize the weights* also decides
whether the signal survives the first forward/backward pass — that's Card 2.

---

## Card 2 — Weight initialization: starting in a healthy place

### Problem
Before training even begins, weights are random. If they're **too large**, activations explode; if
**too small**, activations (and gradients) shrink to nothing as they pass through layers. Bad
initialization can *re-create* the vanishing/exploding gradient problem on step one, even with ReLU.
A deep network is a long chain — small biases in scale compound layer over layer.

### Fix
Initialize weights with a variance **scaled to the layer's size**, so signal variance stays roughly
constant from layer to layer:

- **Xavier/Glorot init** — designed for **tanh/sigmoid**; scales by both input and output size.
- **He/Kaiming init** — designed for **ReLU** (accounts for ReLU zeroing half the inputs).
  Use this with ReLU networks. It's PyTorch's default for `nn.Linear`/`Conv2d` in many cases,
  but you set it explicitly for control.

### Why
- **Why scale by layer size?** Each neuron sums many inputs. More inputs → bigger sum → you must
  shrink each weight so the output variance doesn't blow up. The init formula does exactly this
  (variance ∝ 1/fan_in). Keeping variance ≈ 1 across layers keeps both the forward signal and the
  backward gradient alive.
- **Why He for ReLU specifically?** ReLU sets ~half its inputs to zero, halving the variance. He init
  compensates by doubling the variance (uses 2/fan_in), so the signal stays balanced. Using Xavier
  with ReLU slightly under-scales; using He is the matched choice.
- **Why it matters even with ReLU.** ReLU stops the gradient shrinking *through activations*, but
  bad initial weight *scale* can still kill the signal. Activation + init are a pair — you need both.

**Links back:** the second fix to the broken network — set He init, watch training start cleanly.

**Links forward:** good init helps at the *start*, but as weights change during training, the
distribution of activations drifts again. Keeping it healthy *throughout* training is Card 3.

---

## Card 3 — Batch normalization: keeping activations healthy during training

### Problem
Even with good activations and init, as the network trains, each layer's weights change — so the
**distribution of inputs to the next layer keeps shifting** (called *internal covariate shift*).
Layers are chasing a moving target. This slows training, makes it sensitive to learning rate, and
can push activations back into saturated/dead zones.

### Fix
**Batch Normalization (BatchNorm).** For each mini-batch, normalize a layer's outputs to roughly
zero mean and unit variance, then apply two *learnable* parameters (scale γ and shift β) so the
network can undo the normalization if it needs to.

```
normalize:  x̂ = (x − batch_mean) / sqrt(batch_var + ε)
rescale:    y  = γ · x̂ + β        (γ, β are learned)
```

In PyTorch: `nn.BatchNorm1d(features)` or `nn.BatchNorm2d(channels)`, placed after the linear/conv
layer (commonly before the activation).

### Why
- **Why normalizing helps.** It keeps each layer's inputs in a stable, well-behaved range, so no
  layer is chasing a drifting distribution. Training becomes faster and far less sensitive to the
  learning rate and to initialization.
- **Why the learnable γ and β?** Forcing every layer to exactly mean-0/var-1 would be too rigid —
  sometimes a layer *should* output a different scale. γ and β let the network learn the best
  scale/shift, so BatchNorm never costs expressiveness.
- **Why it also acts as mild regularization.** The batch statistics add a little noise (each batch
  is slightly different), which nudges the model away from overfitting — a small bonus on top of
  its main job.
- **Train vs eval difference (important gotcha).** During training it uses the *batch's* stats;
  during evaluation it uses *running averages* collected during training. This is exactly why
  `model.train()` / `model.eval()` matter — BatchNorm behaves differently in each mode.

**Links back:** third fix to the broken network — add BatchNorm, watch training speed up and
stabilize.

**Links forward:** activations, init, and BatchNorm keep the *signal* healthy. But the *update rule*
that actually moves the weights can also be slow or unstable — smarter optimizers are Card 4.

---

## Card 4 — Optimizers: smarter, faster descent

### Problem
Plain gradient descent (Group 1's `p -= lr * p.grad`) has real weaknesses: it's slow in flat
regions, oscillates in steep narrow valleys, gets stuck at saddle points, and forces you to hand-tune
one learning rate for *all* parameters. On hard loss surfaces, vanilla SGD crawls.

### Fix
A progression of increasingly smart update rules (each fixes the previous one's weakness):

- **SGD** — the baseline: step opposite the gradient. *(Often used with mini-batches.)*
- *(addition)* **Momentum** — accumulate a running "velocity" of past gradients; keep moving in a
  consistent direction. Speeds through flat areas, damps oscillation.
- *(addition)* **NAG (Nesterov)** — momentum that "looks ahead" before stepping; a bit more precise.
- *(addition)* **AdaGrad** — give each parameter its *own* learning rate, scaled down for frequently-
  updated params. Good for sparse features, but the rate decays too aggressively over time.
- *(addition)* **RMSProp** — fixes AdaGrad's over-decay by using a *moving average* of squared
  gradients instead of a growing sum.
- **Adam** — the default. Combines **Momentum** (direction) + **RMSProp** (per-parameter scaling).
  Robust, fast, works out-of-the-box. Also **AdamW** — Adam with correctly-decoupled weight decay,
  now preferred when you use L2 regularization.

### Why
- **Why momentum?** Averaging past gradients smooths the path — like a ball rolling downhill gaining
  speed — so you don't crawl in flat regions or zig-zag across narrow valleys.
- **Why per-parameter learning rates (Ada/RMS/Adam)?** Different parameters need different step
  sizes; one global rate is a compromise. Scaling each parameter's step by its own gradient history
  makes progress on all of them at once.
- **Why Adam is the default.** It needs little tuning, handles most problems well, and combines the
  two big ideas (momentum + adaptive rates). Start with Adam; drop to well-tuned SGD+momentum only
  when squeezing out the last bit of performance (common in vision research).
- **The key hyperparameter is still the learning rate.** Even Adam has one — too high diverges, too
  low crawls. This is the first knob you tune (Card 7 territory / the hyperparameter search).

**Links back:** the fourth fix — swap SGD → Adam on the network and watch it converge faster.

**Links forward:** the network now trains deep, fast, and stable. But a well-trained network can
still **overfit** — the Group 1 Lab B gap. That's Card 5.

---

## Card 5 — Regularization: the overfitting fix

### Problem
Group 1 Lab B showed it: train accuracy climbs above test accuracy — the model **memorizes**
training specifics that don't generalize. A powerful model with enough parameters *will* overfit,
especially with limited data.

### Fix
Techniques that constrain the model so it learns general patterns, not memorized noise:

- **Dropout** — during training, randomly "turn off" a fraction (e.g. 25–50%) of neurons each step.
  `nn.Dropout(p)`. Off automatically at eval time.
- **L2 weight decay** — add a penalty on large weights to the loss (keeps weights small/smooth).
  In PyTorch it's the `weight_decay=` argument on the optimizer.
- *(addition)* **L1 regularization** — penalty on absolute weight size; encourages sparsity.
- *(addition)* **Early stopping** — stop training when test/validation loss stops improving, before
  the model starts memorizing.

### Why
- **Why dropout works.** By randomly removing neurons, no single neuron can rely on specific others —
  the network must learn *redundant, robust* features. It's like training many smaller sub-networks
  and averaging them. This directly attacks memorization.
- **Why L2 weight decay works.** Large weights let the model fit sharp, specific quirks of the
  training data. Penalizing weight size forces smoother functions that generalize better. (This is
  the same "keep it simple" principle behind Occam's razor.)
- **Why early stopping works.** Overfitting happens *late* in training, once the easy general
  patterns are learned and the model starts fitting noise. Stopping at the validation-loss minimum
  catches the model at peak generalization.
- **How they combine.** These stack — dropout + weight decay + early stopping together. The goal is
  always the same: shrink the train/test gap from Group 1 Lab B.

**Links back:** apply these to Lab B's overfitting models and watch the shaded gap shrink.

**Links forward:** regularization helps you not-overfit the data you *have*. But what if you simply
don't have enough data? Cards 6 and 7 handle that.

---

## Card 6 — Data augmentation: the "not enough data" fix (part 1)

### Problem
Deep networks are data-hungry. With a small dataset, even a well-regularized model overfits, because
it has seen too few examples to learn what actually generalizes.

### Fix
**Data augmentation** — synthetically expand the dataset by applying label-preserving random
transformations to your existing data. For images: random crops, flips, rotations, brightness/contrast
jitter, small translations. In PyTorch: `torchvision.transforms` (e.g. `RandomHorizontalFlip`,
`RandomRotation`, `RandomResizedCrop`, `ColorJitter`).

### Why
- **Why it works.** A cat flipped horizontally is still a cat — but to the network it's a *new*
  training example. Each augmentation teaches the model that the label is invariant to that
  transformation (position, orientation, lighting), which is exactly the kind of generalization we
  want. You get many effective examples from each real one.
- **Why it's also regularization.** The model never sees the exact same image twice, so it can't
  memorize pixel-perfect specifics — it's forced to learn the underlying object. Augmentation is one
  of the most effective anti-overfitting tools in vision.
- **The one rule: labels must be preserved.** Don't apply a transform that changes the correct answer
  (e.g. flipping a "6"/"9" digit, or horizontally flipping text). Choose augmentations that make
  sense for *your* data.

**Links forward:** augmentation stretches a small dataset, but sometimes you need knowledge the data
alone can't provide. The biggest "not enough data" lever is Card 7.

---

## Card 7 — Transfer learning: standing on a pretrained model's shoulders

### Problem
Training a strong deep network from scratch needs huge data and compute — millions of images, days of
GPU time. Most real projects have neither. Starting from random weights wastes everything the vision
community has already learned about edges, textures, and shapes.

### Fix
**Transfer learning** — start from a model **pretrained** on a massive dataset (e.g. a ResNet/
EfficientNet trained on ImageNet), and adapt it to your task. Two main modes:

- **Feature extraction** — freeze the pretrained layers, replace only the final classification head,
  train just that head on your data. Fast, needs little data.
- **Fine-tuning** — unfreeze some (or all) pretrained layers and train them at a *small* learning
  rate so you adapt the features without destroying them.

In PyTorch: load a model from `torchvision.models` with pretrained weights, freeze parameters
(`requires_grad = False`), swap the final layer for one matching your number of classes, and train.

### Why
- **Why it works.** Early layers of a vision network learn *generic* features — edges, corners,
  textures — that are useful for almost any image task. Only the later layers are task-specific. So
  you reuse the generic feature extractor for free and only learn the task-specific part.
- **Why freeze first, fine-tune later.** With little data, training the whole network would overfit
  or wreck the good pretrained features. Freezing protects them; you only train the small head. If
  you have more data, you can then fine-tune deeper layers gently.
- **Why a small learning rate when fine-tuning.** The pretrained weights are already good — you want
  to *nudge* them, not overwrite them. A large LR would erase the very knowledge you're borrowing.
- **Why this is the biggest practical lever.** For most real-world image tasks, transfer learning
  from ImageNet beats training from scratch — higher accuracy, less data, far less compute. It's the
  default professional starting point, not an advanced trick.

**Links forward (out of Group 2):** the network now trains deep, fast, generalizes, and can learn
from small datasets. But everything so far assumes inputs are *independent* (one image → one label).
It has no notion of **order or sequence** — of the past influencing the present. That's the problem
Group 3 (RNNs, LSTMs, sequence models) is built to solve.

---

## Group 2 summary — problem → fix map

| Problem (from Group 1) | Fix (Group 2 card) |
|------------------------|--------------------|
| Vanishing gradient (saturating activations) | ReLU / Leaky ReLU / SELU — Card 1 |
| Signal dies at initialization | He / Xavier init — Card 2 |
| Activation distribution drifts during training | Batch Normalization — Card 3 |
| Slow / unstable / oscillating training | Momentum → RMSProp → Adam — Card 4 |
| Overfitting (train ≫ test) | Dropout, L2 weight decay, early stopping — Card 5 |
| Not enough data (overfits small sets) | Data augmentation — Card 6 |
| Can't afford to train from scratch | Transfer learning — Card 7 |

Every one of these is an answer to a failure you *watched happen* in Group 1. That's why they're
tools, not spells.
