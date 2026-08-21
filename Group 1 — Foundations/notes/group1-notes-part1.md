# Group 1 — Foundations · Notes (Part 1 of 2)

Format for every card: **Problem → Fix → Why**.
Covers: Perceptron/ANN · Loss functions · Backprop + Gradient Descent.
(CNN and Vanishing Gradient are in Part 2.)

---

## Card 1 — Perceptron / ANN: the learnable unit and the forward pass

### Problem
We want a machine that maps inputs to outputs — image → label, features → price —
*without us hand-writing the rules*. Classical code needs explicit `if/else` logic.
For messy real-world patterns (what makes a 7 look like a 7), nobody can write those rules by hand.

### Fix
A **neuron**: take inputs `x`, multiply each by a **weight** `w`, add a **bias** `b`,
sum them, and pass the sum through a non-linear **activation** `σ`.

```
z = w·x + b          (weighted sum — a linear step)
a = σ(z)             (activation — a non-linear squashing)
```

Stack many neurons into **layers**, stack layers into a network → an **Artificial Neural Network (ANN)**.
Running data forward through this to get a prediction is the **forward pass**.
The weights and biases are the **learnable parameters** — training is just finding good values for them.

### Why
- The **weighted sum** lets each input contribute a different amount — the network can learn that
  "color" matters more than "size" by giving it a bigger weight.
- The **bias** shifts the decision boundary so it doesn't have to pass through the origin
  (same role as the intercept `c` in `y = mx + c`).
- The **activation** is the crucial part: without a non-linearity, stacking layers is pointless —
  a stack of linear steps collapses into one linear step (matrix math: `W₂(W₁x) = (W₂W₁)x`, still linear).
  The non-linearity is what lets deep networks represent complex, curved decision boundaries.

**Links forward:** a single neuron with a step activation is the perceptron (manual Exp 1).
Replace the step with a smooth activation and you can do calculus on it — which is what makes
backprop (Card 3) possible.

---

## Card 2 — Loss functions: measuring how wrong the network is

### Problem
The forward pass gives a prediction. But is it any good? We need a **single number** that says
"how wrong was that?" — and it has to be something we can *optimize*. Without a precise, differentiable
measure of wrongness, there's no signal to improve on.

### Fix
A **loss function** `L(prediction, truth)` outputs a scalar: low = good, high = bad.
Training = adjusting weights to make this number small. The right loss depends on the task:

- **Regression (predicting a number):**
  - **MSE — Mean Squared Error** `mean((ŷ - y)²)`. The default.
  - *(addition)* **MAE — Mean Absolute Error** `mean(|ŷ - y|)`. Use when data has outliers.

- **Classification (predicting a category):**
  - **Cross-Entropy Loss.** Compares the predicted probability distribution to the true label.
  - *(addition — worth knowing the two forms):*
    - **Binary** cross-entropy → 2 classes (spam / not-spam).
    - **Categorical** cross-entropy → many classes (digit 0–9). This is what the MNIST labs use.

> Note (as we agreed): MSE and cross-entropy were the two on your list; I've added MAE and the
> binary/categorical split because they come up the instant you touch real data. All optional to
> go deep on now — the two you need for Group 1 are **MSE (regression)** and **cross-entropy (classification).**

### Why
- **Why squared in MSE?** Squaring makes all errors positive (so they don't cancel) and punishes
  big errors far more than small ones — a prediction that's off by 10 costs 100, off by 2 costs 4.
  The model is pushed hardest to fix its worst mistakes.
- **Why MAE for outliers?** Because MSE's squaring makes one wild outlier dominate the whole loss;
  MAE treats errors proportionally, so a few bad data points don't hijack training.
- **Why cross-entropy and not MSE for classification?** Cross-entropy measures *how confidently wrong*
  you were about a probability. If the true class is "cat" and you said 1% cat, cross-entropy blows up
  (rightly — you were confidently wrong). MSE on class labels gives weak, flat gradients that train
  slowly. Cross-entropy gives strong gradients exactly when the model is confidently wrong, which is
  when you most want a correction.

**Links forward:** the loss is the thing backprop differentiates. A loss you can't differentiate
is a loss you can't learn from — which is why every loss here is smooth.

---

## Card 3 — Backpropagation + Gradient Descent: the engine of learning

### Problem
We have a prediction (Card 1) and a number saying how wrong it is (Card 2).
Now the real question: a network can have *millions* of weights — **which ones do we nudge, in which
direction, and by how much** to make the loss go down? Guessing is hopeless at that scale.

### Fix — two pieces working together:

**1. Gradient Descent — the *strategy*.**
The gradient of the loss w.r.t. a weight tells you the direction that *increases* loss fastest.
So step the opposite way. Repeat.

```
w_new = w_old − learning_rate × (∂L/∂w)
```

The **learning rate** controls step size. That's the entire optimization loop in one line.

**2. Backpropagation — the *mechanism* that computes every `∂L/∂w` efficiently.**
The network is a chain of operations, so the loss depends on an early weight *through* all the
layers after it. Backprop applies the **chain rule** of calculus, working backward from the loss:
compute the gradient at the output, then propagate it layer by layer toward the input, reusing
the work at each step.

The full loop per batch:
```
forward pass → compute loss → backward pass (backprop the gradients) → update weights → repeat
```

### Why
- **Why the gradient?** It's the multivariable "slope." At any point on the loss surface it points
  straight uphill; negating it points straight downhill toward lower loss. Following it is the
  mathematically fastest local way down.
- **Why backprop instead of computing each gradient separately?** Naively, each of a million weights
  would need its own pass — impossibly slow. Backprop computes *all* gradients in a single backward
  sweep by reusing shared sub-results (the chain rule lets a layer's gradient be built from the
  next layer's). This efficiency is the reason deep learning is computationally possible at all.
- **Why does the learning rate matter so much?** Too large → you overshoot the minimum and diverge.
  Too small → training crawls. It's the first knob you'll tune, and Group 2 is largely about
  smarter versions of this update rule (momentum, Adam, etc.).

**In PyTorch this is nearly free:** `autograd` builds the computation graph during the forward pass,
`loss.backward()` runs backprop to fill in every `.grad`, and the optimizer does the update step.
Our labs write this loop by hand (not hidden inside `.fit()`) precisely so you can *see* each stage —
and so we can log the gradients to watch them vanish.

**Links forward:** backprop is exactly where the **vanishing gradient** problem lives — as gradients
propagate backward through many layers, they can shrink toward zero, and the earliest layers stop
learning. That's Card 5, and it's the note Group 1 ends on.
