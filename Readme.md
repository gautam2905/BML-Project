# Project Report: Scalable Uncertainty Quantification via Hybrid Variational Latent Derivative Bayesian Last Layers

**Author:** Gautam Gupta
**Course:** Bayesian Machine Learning
**Date:** December 12, 2025

## 1. Abstract
This project investigates the limitations of standard Bayesian Last Layer (BLL) networks regarding out-of-distribution (OOD) uncertainty and computational scalability. While "Latent Derivative BLL" (LDBLL) improves uncertainty calibration via functional priors on model derivatives, it relies on exact inference, which scales poorly with dataset size due to matrix inversions. Conversely, "Variational BLL" (VBLL) offers a scalable, mini-batch-friendly alternative but suffers from feature overfitting similar to standard BLLs.

I propose and implement a novel **Hybrid Variational LDBLL**, integrating the derivative-based regularization of LDBLL into the scalable VBLL framework. Our experiments on "gap" regression datasets demonstrate that this hybrid approach successfully combines the computational benefits of variational inference with the robust OOD uncertainty quantification of latent derivative priors.

## 2. Introduction
Modern neural networks often yield overconfident predictions on OOD data. The BLL framework addresses this by performing Bayesian inference on the last layer while learning deterministic features. However, two major issues persist:
1.  **Feature Overfitting:** The deterministic feature extractor learns representations that collapse OOD, leading to low uncertainty regions where data does not exist.
2.  **Scalability:** Standard BLL (and LDBLL) often requires full-batch processing to update posterior statistics (matrix inversion), making it computationally expensive for large datasets.

While the original proposal aimed to explore "Adaptive Priors," our initial reproduction phase revealed that scalability was a more critical bottleneck. Consequently, the project focus shifted to merging the **Latent Derivative Prior** (for robustness) with **Variational Inference** (for scalability).

## 3. Methodology

### 3.1 Baseline: BLL and LDBLL
I first implemented the standard BLL and LDBLL as described by Watson et al..
* **BLL:** Maximizes the marginal likelihood.
* **LDBLL:** Adds a functional regularization term $D_{KL}(\pi(z)||p(z))$, where $z$ is the Jacobian of the network output w.r.t. the input. This forces the function to maintain "smoothness" and uncertainty OOD.

### 3.2 Transition to Variational Inference (VBLL)
To address the BLL's requirement for exact matrix inversion (which prohibits standard mini-batch training), I implemented Variational BLL. Instead of calculating the exact posterior, VBLL approximates it using a variational distribution $q(w|\eta)$. The objective becomes maximizing the Evidence Lower Bound (ELBO):
$$\mathcal{L}_{ELBO} = \mathbb{E}_{q(w)}[\log p(y|x, w)] - D_{KL}(q(w)||p(w))$$
This allows for standard backpropagation updates using AdamW, enabling batching.

### 3.3 Novel Contribution: Hybrid Variational LDBLL
I observed that while VBLL converges faster, it retains the BLL's weakness of OOD overconfidence (see Section 4). To resolve this, I integrated the Latent Derivative prior into the VBLL objective.

The new loss function is:
$$\mathcal{L}_{Total} = -\mathcal{L}_{ELBO} + \beta \cdot \text{Reg}_{derivative}$$

**The Derivative Regularization (Forward KL):**
Crucially, I implemented the regularization using **Forward KL** divergence $D_{KL}(P_{\text{prior}} || Q_{\text{posterior}})$, unlike the Reverse KL used in standard VI.
* **Reverse KL ($D_{KL}(Q||P)$):** Forces the posterior $Q$ to fit "inside" the prior $P$. This often leads to mode-seeking behavior, resulting in narrow distributions and underestimated variance.
* **Forward KL ($D_{KL}(P||Q)$):** Forces the posterior $Q$ to "cover" the prior $P$. By setting a high-variance prior on the derivatives (implying I don't know the rate of change OOD), the Forward KL forces the model's posterior predictive variance to expand in regions without data, preventing uncertainty collapse.

## 4. Experiments and Observations

Experiments were conducted using a "SinGap" dataset (synthetic sine wave with a missing data region between $x \in [-2, 2]$) to visualize OOD behavior.

### 4.1 Phase 1: BLL vs. LDBLL (Reproduction)
I successfully reproduced the results of Watson et al..
* **Observation:** The standard BLL fits the training data well but predicts with high confidence (low variance) inside the gap.
* **Observation:** The LDBLL successfully "inflates" uncertainty inside the gap (see *Figure 1* below).
* **Training Dynamics:** I observed that **LDBLL requires significantly more epochs** to converge compared to BLL (e.g., 4500 vs 2000). This is likely because the derivative prior acts as a strong constraint, creating a stiffer optimization landscape that requires more iterations to navigate while balancing data fit and derivative smoothness.

*(Reference: `plots/bll_vs_ldbll_compare.png` generated from code)*

### 4.2 Phase 2: Variational BLL vs. Hybrid Model
I compared the standard Variational BLL against our novel Hybrid Variational LDBLL.

* **Variational BLL:** As hypothesized, the VBLL enabled batch training and stable convergence but failed to capture epistemic uncertainty in the data gap. The variance remained tight around the mean prediction in OOD regions.
* **Hybrid Variational LDBLL:** By injecting the derivative noise (using $\gamma=0.1$) and penalizing the Forward KL of the Jacobian, the hybrid model successfully recovered the "bell-shaped" uncertainty profile in the gap, similar to the exact LDBLL but trained via mini-batches.

*(Reference: `plots/hybrid_compare.png` generated from code)*

### 4.3 Key Implementation Insights

1.  **Computational Cost:**
    * *Exact BLL:* Requires computing $S_N = (\Sigma_0^{-1} + \frac{1}{\sigma^2}\Phi^T\Phi)^{-1}$. This inversion is $O(D^3)$ and must be done on the full dataset or via complex rank-1 updates.
    * *Hybrid VBLL:* The complexity shifts to the forward pass. Computing the Jacobian (via `torch.func.vmap` and `jacrev` in `src/ldbll.py`) adds overhead per batch, but avoids the global matrix inversion, making it practically scalable to larger datasets.

2.  **Forward KL Dynamics:**
    Our experiments confirm the theoretical intuition regarding KL divergence. Standard VI (Reverse KL) minimizes the "exclusive" divergence, causing the approximate posterior to hide within a single mode of the true posterior. In contrast, the LDBLL regularization uses Forward KL on the latent derivatives. This "inclusive" divergence forces the model's gradient distribution to be broad (matching the uninformative prior), which directly translates to higher predictive uncertainty in function space.

## 5. Conclusion
This project successfully bridged the gap between the robust uncertainty quantification of Latent Derivative BLLs and the scalability of Variational BLLs. The resulting **Hybrid Variational LDBLL** demonstrates that:
1.  Latent derivative priors can be effectively optimized via stochastic gradient descent within a variational framework.
2.  The use of Forward KL on derivatives is essential for preventing variance collapse in OOD regions.
3.  While the hybrid method is computationally heavier per-pass than standard VBLL (due to Jacobian computation), it eliminates the $O(N)$ or $O(D^3)$ bottlenecks of exact BLLs, making it a viable strategy for large-scale safety-critical regression tasks.

## 6. References
1.  Watson, J. et al. "Latent Derivative Bayesian Last Layer Networks." AISTATS 2021.
2.  Harrison, J. et al. "Variational Bayesian Last Layers." ICLR 2024.
3.  Weber, N. et al. "Optimizing over a Bayesian Last Layer." NeurIPS 2018.
4.  Rasmussen, C. E., Williams, C. K. I. "Gaussian Processes for Machine Learning." MIT Press, 2006.