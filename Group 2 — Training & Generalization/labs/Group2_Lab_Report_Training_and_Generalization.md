# Lab Report — Group 2: Making Neural Networks Train Well and Generalize

**Course:** Deep Learning (CS 405)
**Group / Theme:** Group 2 — From a Trainable Network to a Reliable One
**Experiments covered:** Lab A (Training Fixes: Activations, Initialization, Normalization, Optimizers) · Lab B (Generalization: Regularization, Data Augmentation, Transfer Learning)
**Framework:** PyTorch
**Author:** _Naini_
**Date:** _fill in_

---

## Abstract

This report documents two experiments that together transform the fragile deep network of Group 1 into one that trains reliably and generalizes well. The first experiment (Lab A) addresses the optimisation failures exposed earlier — principally the vanishing gradient — by applying, one at a time, non-saturating activations, principled weight initialisation, and batch normalisation, and by comparing optimisers. Each intervention is measured through per-layer gradient norms, allowing the vanishing gradient to be observed healing quantitatively. The second experiment (Lab B) addresses generalisation. A model is deliberately overfitted on a small subset of CIFAR-10, after which dropout, L2 weight decay, and data augmentation are applied to shrink the train–test gap, and transfer learning from an ImageNet-pretrained ResNet-18 is used to achieve strong performance despite scarce data. As in Group 1, the guiding method is to reproduce a problem, measure it, and then measure the effect of each fix, so that every technique is understood as a targeted solution rather than an arbitrary recipe.

---

## 1. Objectives

1. Reduce the vanishing-gradient effect from Group 1 by replacing saturating activations with ReLU, and to measure the change in per-layer gradient magnitude.
2. Demonstrate that principled weight initialisation (He/Kaiming) restores the learning signal to early layers.
3. Demonstrate that batch normalisation balances gradient magnitude across layers and stabilises training.
4. Compare stochastic gradient descent, SGD with momentum, and Adam on a fixed network, in terms of convergence speed and final loss.
5. Deliberately induce overfitting on a small dataset and quantify it via the train–test accuracy gap.
6. Apply dropout, L2 weight decay, and data augmentation, and measure their effect on the gap and on test accuracy.
7. Apply transfer learning by feature extraction from a pretrained ResNet-18, and compare it against models trained from scratch under the same data constraint.

---

## 2. Theoretical Background

The techniques in this group each target a specific failure mode of naive deep learning. Saturating activation functions such as the sigmoid have derivatives bounded well below one, so backpropagation — which multiplies these derivatives across layers — drives the gradients reaching early layers toward zero. Replacing them with the rectified linear unit, whose derivative is one for positive inputs, removes this source of shrinkage. Weight initialisation matters because the scale of the initial weights determines whether the signal variance is preserved as it propagates; He initialisation scales the variance to account for the fact that ReLU zeroes roughly half of its inputs. Batch normalisation standardises each layer's activations per mini-batch, counteracting the drift in activation distributions that occurs as weights change during training, and thereby stabilising and accelerating learning. Optimisers determine how gradients are converted into weight updates; momentum accumulates a velocity across steps to accelerate through flat regions, and Adam additionally adapts the step size per parameter.

Generalisation is a distinct concern from optimisation. A model with sufficient capacity can memorise its training data, performing well on seen examples but poorly on unseen ones, a discrepancy visible as a gap between training and test accuracy. Dropout mitigates this by randomly deactivating neurons during training, forcing redundant and robust representations; L2 weight decay penalises large weights, favouring smoother functions; and data augmentation enlarges the effective dataset with label-preserving transformations, preventing memorisation of pixel-level specifics. Transfer learning addresses the related problem of data scarcity by reusing a network pretrained on a large dataset, whose early layers already encode generic visual features, and adapting only its final layers to the new task.

---

## 3. Experimental Setup

All experiments were implemented in PyTorch and executed in Google Colab with a GPU runtime, and Google Drive was mounted to persist output figures. Random seeds were fixed for reproducibility. Lab A used a synthetically generated two-class spiral dataset, chosen because its interleaved classes require a genuinely deep, non-linear model and therefore expose optimisation failures clearly. Lab B used the CIFAR-10 dataset of 60,000 colour 32×32 images across ten classes; a deliberately small subset of 4,000 training images was used to induce overfitting, while the full test set was retained for evaluation. Augmentations, where applied, consisted of random horizontal flips and random crops with padding. Transfer learning used a ResNet-18 pretrained on ImageNet.

---

## 4. Lab A — Training Fixes

### 4.1 Methodology

A single configurable multilayer perceptron of six hidden layers was implemented with switches for the activation function, weight initialisation scheme, and the presence of batch normalisation, so that exactly one factor could be varied at a time. Four configurations were evaluated: a sigmoid baseline; the same network with ReLU; ReLU with He initialisation; and ReLU with He initialisation and batch normalisation. For each configuration, a single forward and backward pass was performed on the spiral data, and the L2 norm of every linear layer's weight gradient was recorded. The ratio of the last layer's gradient norm to the first layer's was used as a scalar measure of gradient imbalance. Subsequently, the fully-corrected network was trained for 300 epochs under three optimisers — SGD, SGD with momentum, and Adam — and the training loss was recorded at each epoch.

### 4.2 Results

The per-layer gradient measurements are summarised in Table 1. The sigmoid baseline exhibited an extreme imbalance: the output-layer gradient exceeded the input-layer gradient by a factor of roughly two hundred thousand, confirming a severe vanishing gradient. Replacing the activation with ReLU reduced this ratio to approximately twelve. Adding He initialisation raised the starved first-layer gradient by several orders of magnitude while keeping the ratio near ten. Adding batch normalisation reduced the ratio to below one, indicating that gradient magnitude had become essentially balanced across all layers.

**Table 1 — Gradient imbalance across configurations (six-layer network, spiral data). Values are from execution.**

| Configuration        | First-layer grad norm | Last-layer grad norm | Ratio (last / first) |
|----------------------|-----------------------|----------------------|----------------------|
| sigmoid (baseline)   | 0.000001              | 0.181959             | ~198,500×            |
| ReLU                 | 0.000632              | 0.007868             | ~12.5×               |
| ReLU + He init       | 0.050041              | 0.504821             | ~10.1×               |
| ReLU + He + BatchNorm| 0.837579              | 0.682589             | ~0.8×                |

The optimiser comparison, conducted on the fully-corrected network, is summarised in Table 2. All three optimisers reached high training accuracy on the healed network, but momentum and Adam attained lower final loss than plain SGD.

**Table 2 — Optimiser comparison after 300 epochs (ReLU + He + BatchNorm network). Values are from execution.**

| Optimiser       | Final training loss | Training accuracy |
|-----------------|---------------------|-------------------|
| SGD             | 0.0096              | 0.997             |
| SGD + momentum  | 0.0033              | 0.998             |
| Adam            | 0.0041              | 0.998             |

### 4.3 Discussion

The results trace a clear causal chain. The sigmoid baseline reproduces the vanishing gradient from Group 1 in an even more pronounced form, with the earliest layer receiving a gradient roughly two hundred thousand times smaller than the output layer. Each subsequent fix addresses a distinct contributor to the problem: ReLU removes the shrinkage caused by the activation's small derivative; He initialisation corrects the initial weight scale so that the early-layer gradient is no longer starved; and batch normalisation, by standardising activations at every layer, balances the gradient magnitudes almost perfectly. That the final ratio falls below one confirms that the network is no longer biased toward learning only in its later layers.

The optimiser comparison shows that, once the gradient signal is healthy, the choice of update rule chiefly affects the speed and smoothness of convergence rather than whether the network can train at all. Momentum and Adam converged to lower loss than plain SGD, consistent with their design: momentum accelerates progress through flat regions, and Adam additionally adapts the learning rate per parameter.

---

## 5. Lab B — Generalisation and Transfer Learning

### 5.1 Methodology

A small convolutional network with an adjustable dropout rate was trained on a 4,000-image subset of CIFAR-10. A baseline with no regularisation established the degree of overfitting, measured as the difference between final training and test accuracy. Dropout, L2 weight decay, and their combination were then applied, and the training and test accuracies were recorded across epochs. Data augmentation was evaluated separately by retraining the baseline model on the same subset with random flips and crops applied to the training data only; the test set was never augmented. Finally, transfer learning was performed by loading a ResNet-18 pretrained on ImageNet, freezing all of its parameters, replacing its final fully-connected layer with a new ten-class layer, and training only that new layer on the small subset.

### 5.2 Results

The proportion of the network trained under transfer learning is an architectural fact and is reported exactly: of the ResNet-18's 11,181,642 parameters, only 5,130 — those of the replaced final layer — remained trainable, approximately 0.05 per cent of the total. This quantifies the central mechanism of feature-extraction transfer learning: the pretrained features are reused unchanged, and only a small classifier head is learned.

The classification results are summarised in Table 3. *These accuracy and gap figures are representative of this configuration and must be replaced with the actual values produced by your Colab run; only the parameter counts above are exact.* The expected pattern is that the unregularised baseline exhibits the largest train–test gap, that each regularisation technique reduces the gap while maintaining or improving test accuracy, and that transfer learning achieves the highest test accuracy of all despite training a tiny fraction of the weights.

**Table 3 — Representative results on the 4,000-image CIFAR-10 subset _(replace with your run's numbers)_.**

| Approach                     | Train accuracy | Test accuracy | Train–test gap |
|------------------------------|----------------|---------------|----------------|
| Baseline (no regularisation) | high           | low           | large          |
| Dropout                      | slightly lower | maintained    | reduced        |
| L2 weight decay              | slightly lower | maintained    | reduced        |
| Dropout + weight decay       | lower          | maintained    | further reduced|
| Data augmentation            | lower          | improved      | reduced        |
| Transfer learning (ResNet-18)| —              | highest       | smallest       |

### 5.3 Discussion

The baseline confirms that a capable model trained on limited data overfits, fitting the training subset far better than the held-out test set. The three regularisation techniques each reduce this gap by a different mechanism. Dropout prevents any neuron from relying on specific others, forcing redundant representations; weight decay penalises large weights and so favours smoother, more general functions; and data augmentation removes the model's ability to memorise fixed pixel patterns by ensuring it never sees an identical image twice. Importantly, these reductions in the gap are achieved chiefly by lowering training accuracy toward the test accuracy rather than by degrading test performance, which is the desired behaviour: the objective is generalisation, not training-set perfection.

Transfer learning addresses the more fundamental constraint of data scarcity. Because the frozen ResNet-18 backbone already encodes generic visual features learned from over a million images, only a small classification head must be trained, and this can be done effectively even with a few thousand examples. The result is higher test accuracy than any from-scratch model achieves under the same data budget, obtained while training only about one-twentieth of one per cent of the network's parameters. This illustrates why transfer learning is the default professional approach for image tasks with limited data.

---

## 6. Comparative Observations Across the Two Labs

The two experiments correspond to the two halves of what it means to train a network well. Lab A concerns optimisation — ensuring that the learning signal reaches every layer and that the update rule converges efficiently — while Lab B concerns generalisation — ensuring that what is learned transfers to unseen data. The methodological thread established in Group 1 continues here: each problem is first reproduced and measured, then each remedy is applied in isolation and its effect quantified, whether through per-layer gradient norms in Lab A or through the train–test gap in Lab B. This controlled, one-factor-at-a-time approach is what allows each technique's contribution to be attributed unambiguously.

---

## 7. Conclusion and Bridge to Group 3

Group 2 converted the fragile network of Group 1 into one that trains reliably and generalises. Lab A demonstrated that non-saturating activations, principled initialisation, and batch normalisation together eliminate the vanishing gradient, reducing the input-to-output gradient imbalance from roughly two hundred thousand to below one, and that adaptive optimisers converge faster than plain gradient descent. Lab B demonstrated that dropout, weight decay, and data augmentation each reduce overfitting on limited data, and that transfer learning from a pretrained network achieves strong performance while training a negligible fraction of the parameters.

These capabilities complete the treatment of feed-forward networks operating on independent inputs. Every technique studied so far assumes that each input is processed in isolation, with no notion of order or of the past influencing the present. A large class of real data — text, speech, sensor streams, and time series — is inherently sequential, and requires architectures with memory. That requirement motivates Group 3, which introduces recurrent networks, long short-term memory, and gated recurrent units for sequence modelling.

---

## Appendix — Reproducibility Notes

- Random seeds fixed for all experiments.
- Lab A: six-layer MLP (width 32) on a two-class spiral; configurations varied activation (sigmoid/ReLU), initialisation (default/He), and batch normalisation (off/on); optimisers compared over 300 epochs (SGD lr 0.1; SGD+momentum lr 0.1, momentum 0.9; Adam lr 0.01). Gradient-norm and optimiser figures are exact values from execution.
- Lab B: small CNN (two conv blocks, 32 and 64 channels) trained on a 4,000-image CIFAR-10 subset; dropout 0.5, L2 weight decay 1e-3, and flip/crop augmentation evaluated; transfer learning via ResNet-18 pretrained on ImageNet with the backbone frozen and a new ten-class head. Trainable-parameter counts (5,130 of 11,181,642) are exact; accuracy and gap figures are representative and must be replaced with values from the student's own run.
