# Group 1 — Foundations · Notes (Part 2 of 2)

Format for every card: **Problem → Fix → Why**.
Covers: CNN · Vanishing Gradient (first look).
(Perceptron, Loss, Backprop are in Part 1.)

---

## Card 4 — CNN: why fully-connected nets are wrong for images

### Problem
Take the ANN from Card 1 and feed it an image. To do that you must **flatten** the image —
a 28×28 image becomes a 784-long vector, and the first layer connects every pixel to every neuron.
Three things break:

1. **Parameter explosion.** A modest 200×200 color image is 120,000 numbers. One fully-connected
   layer of 1,000 neurons would need 120 *million* weights — for one layer. It won't scale.
2. **No spatial awareness.** Flattening throws away *where* pixels are. The network has no idea that
   two pixels are neighbours — a cat in the top-left and a cat in the bottom-right look like
   completely unrelated inputs. Position shouldn't change what an object *is*.
3. **No translation invariance.** An edge is an edge wherever it appears. An FC net has to re-learn
   "edge" separately for every location, wasting capacity.

### Fix
A **Convolutional Neural Network (CNN)** replaces "every pixel to every neuron" with two ideas:

- **Convolution.** A small filter (e.g. 3×3 weights) slides across the image, computing a weighted
  sum at each position, producing a **feature map**. The *same* filter is reused everywhere
  (**weight sharing**). One filter might detect vertical edges; a layer has many filters, each
  learning a different feature.
- **Pooling** (e.g. 2×2 max-pool). Downsamples each feature map — keeps the strongest response in
  each little region, shrinking the spatial size.

Stack these: early layers learn edges/textures, deeper layers combine those into shapes, then objects.
A few fully-connected layers at the very end turn the final features into class scores.

### Why
- **Why weight sharing fixes parameters.** A 3×3 filter is 9 weights, reused across the whole image,
  instead of a unique weight per pixel-pair. Millions of parameters collapse to thousands.
  This is *the* reason CNNs scale to real images.
- **Why convolution fixes spatial awareness.** The filter operates on a local neighbourhood, so
  "which pixels are next to each other" is baked into the operation. Structure is preserved, not
  flattened away.
- **Why it gives translation invariance.** Because the same filter slides everywhere, a feature
  learned in one spot is automatically detected in every other spot — learn "edge" once, find it
  anywhere.
- **Why pooling.** It shrinks computation, and by keeping the *strongest* activation it makes the
  network care that a feature is *present* more than exactly *where* — adding robustness to small
  shifts.

**Links back:** it's still Card 1's neurons trained by Card 3's backprop with Card 2's loss — a CNN
just wires the neurons smarter for images. Nothing about learning changed; only the architecture did.

**Links forward:** CNNs let us go *deep* (many layers) to build up complex features. But depth is
exactly what triggers the problem the next card exposes.

---

## Card 5 — Vanishing Gradient (first look): the problem Group 1 ends on

> This is the card we deliberately **leave unsolved**. Group 1's job is to make this pain *visible*;
> Group 2 is the toolkit that fixes it. Understanding it clearly here is what makes Group 2 feel
> necessary instead of a random list of tricks.

### Problem
Backprop (Card 3) computes gradients by multiplying terms layer by layer, from the output back to
the input — the chain rule is a **long chain of multiplications**. In a deep network, the gradient
reaching the *early* layers is the product of many small numbers.

With classic activations like **sigmoid** or **tanh**, the derivative is at most 0.25 (sigmoid) and
is near zero whenever the neuron is saturated (very high or very low input). Multiply a dozen numbers
that are all < 1 and the result rushes toward zero:

```
0.2 × 0.2 × 0.2 × ... (×10 layers) ≈ 0.0000001   → effectively no gradient
```

So the early layers receive almost **no learning signal**. They barely update. The network's first
layers — the ones that should learn basic features — stay near their random initial values while
only the last layers learn. Training stalls or the model underperforms for reasons that aren't
obvious from the loss curve alone.

*(The mirror image is the **exploding gradient**: if the multiplied terms are > 1, the product blows
up to huge values and training diverges. Same root cause — a long product — opposite direction.)*

### Fix
**Not yet — on purpose.** In Group 1 we only *expose* it: we'll build a deep network with sigmoid
activations and **log the gradient norm at each layer**, and you'll watch the early-layer gradients
sit near zero while late-layer gradients are healthy. Seeing that plot is the deliverable.

The fixes are the whole point of Group 2, previewed here so you know where each is going:
- **Better activations** (ReLU/SELU) — derivative doesn't shrink the signal the way sigmoid does.
- **Weight initialization** (Xavier/He) — keeps signal variance stable across layers from the start.
- **Batch normalization** — keeps activations in a healthy range so neurons don't saturate.
- **Residual connections** (later) — give gradients a shortcut path back.

### Why
- **Why it happens specifically in deep nets.** Shallow nets have short chains — few multiplications,
  little shrinkage. Depth is what turns "each term slightly less than 1" into "product ≈ 0." The
  problem is a *direct consequence of depth plus saturating activations.*
- **Why sigmoid/tanh are the culprits.** Their gradients are bounded well below 1 and collapse to ~0
  when saturated, so they're the worst offenders in a long product. This single fact is why the field
  moved to ReLU as the default — the first thing Group 2 will show.
- **Why we expose before fixing.** If you've *watched* early layers refuse to learn, then ReLU, He
  init, and BatchNorm aren't abstract tricks — they're the specific answers to a failure you saw with
  your own eyes. That's the entire teaching strategy of this plan.

**Links forward:** this is the bridge out of Group 1. Every tool in Group 2 exists to keep this
gradient product from collapsing (or exploding) — so the network can actually be trained deep.
