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

## 3. Theoretical Background and Methodology

To contextualize the proposed Hybrid model, I must first rigorously define the two frameworks it bridges: the regularization mechanics of Latent Derivative BLLs and the inference mechanics of Variational BLLs.

### 3.1 Latent Derivative Bayesian Last Layers (LDBLL)
The LDBLL framework, introduced by Watson et al. (2021), addresses the pathology where Bayesian Last Layers (BLL) learn feature maps $\phi(x)$ that collapse onto a deterministic mean function in out-of-distribution (OOD) regions. While the last layer $w$ is Bayesian, the features $\phi(x)$ are trained via Type-II Maximum Likelihood (optimizing the marginal likelihood), which tends to result in over-confident feature representations.

#### 3.1.1 The Intuition: Controlling the Jacobian
The core insight of LDBLL is that epistemic uncertainty OOD is inextricably linked to the sensitivity of the function $f(x)$ to its inputs. Consider a first-order Taylor expansion of the network around an input $x$:
$$f(x + \delta) \approx f(x) + \nabla_x f(x)^T \delta$$
the above Taylor approximation illustrates how z influences the predictive uncertainty as the perturbation δ grows. As typical regression problems only consider directly corresponding pairs (y,  ̄x,0), this latent variable perspective is irrelevant for the training data as δ = 0. However, by characterizing prediction between and outside the training data as δ != 0, one can appreciate how controlling the distribution of z influences the epistemic uncertainty in the predictions. 
For a linear Bayesian model $f(x) = w^T \phi(x)$, the gradient with respect to the input is:
$$z(x) := \nabla_x f(x) = \nabla_x (w^T \phi(x)) = (\nabla_x \phi(x))^T w = J_{\phi}(x)^T w$$
where $J_{\phi}(x)$ is the Jacobian of the features. Because $w$ follows a Gaussian posterior $p(w|\mathcal{D}) = \mathcal{N}(\mu_N, \Sigma_N)$, the latent derivative $z(x)$ is also Gaussian distributed (as it is a linear transformation of Gaussian weights):
$$p(z | x, \mathcal{D}) = \mathcal{N}(z \mid J_{\phi}^T \mu_N, \, J_{\phi}^T \Sigma_N J_{\phi})$$

If the model learns features such that $J_{\phi}(x) \to 0$ in OOD regions (a common failure mode), the variance of $z$ collapses to zero. This implies the function becomes flat and deterministic OOD.

#### 3.1.2 Functional Regularization via Index Sets
To prevent this collapse, LDBLL places a **Functional Prior** $\pi(z)$ on the derivatives. A typical choice is a zero-mean Gaussian process with high variance, $\pi(z) = \mathcal{N}(0, \gamma I)$, which encodes the belief that "I do not know how the function changes OOD, so the gradient could be anything."

The training objective augments the standard Marginal Likelihood with a Functional KL divergence ($D_{KL}$) term. Since calculating $D_{KL}$ over the entire input domain is intractable, Watson et al. approximate it using a finite **Index Set** $\mathcal{T}$ (points sampled near the training data with additive noise):

$$\mathcal{L}_{LDBLL} = \log p(\mathcal{D} | \theta) - \frac{\beta}{|\mathcal{T}|} \sum_{x \in \mathcal{T}} D_{KL}\Big( \pi(z|x) \,||\, p(z|x, \mathcal{D}) \Big)$$

Crucially, Watson et al. utilize the **Forward KL** (Prior || Posterior). This encourages the posterior distribution of the Jacobian to "cover" the high-variance prior, forcing the model to maintain high predictive uncertainty in function space where data is sparse.

### 3.2 Variational Bayesian Last Layers (VBLL)
The standard BLL and LDBLL rely on exact inference to compute the posterior covariance $\Sigma_N = (\Sigma_0^{-1} + \sigma^{-2}\Phi^T\Phi)^{-1}$. This operation is $O(D^3)$ or requires $O(N)$ updates, making it computationally intractable for large datasets or mini-batch training.

Harrison et al. (2024) propose VBLL to solve this by approximating the exact posterior $p(w|\mathcal{D})$ with a variational distribution $q(w|\eta) = \mathcal{N}(\mu_q, S_q)$.

#### 3.2.1 The Objective (Regression)
I maximize the Evidence Lower Bound (ELBO):
$$\mathcal{L}_{ELBO} = \mathbb{E}_{q(w)} [\log p(y|x, w)] - D_{KL}(q(w) || p(w))$$

For the regression case with likelihood $y \sim \mathcal{N}(w^T\phi(x), \sigma^2)$, the expected log-likelihood term has a closed-form solution that enables efficient computation. Harrison et al. derive this as:

$$\mathbb{E}_{q(w)} [\log p(y|x, w)] = \log \mathcal{N}(y \mid \mu_q^T \phi(x), \sigma^2) - \frac{1}{2\sigma^2} \text{Tr}(\phi(x)^T S_q \phi(x))$$

**Interpretation:**
1.  **Mean Fit:** The first term maximizes the fit of the mean prediction $\mu_q^T \phi(x)$ to the data.
2.  **Uncertainty Penalty:** The trace term $\text{Tr}(\phi^T S_q \phi)$ penalizes the model if the predictive variance is high where the prediction error is high.

This formulation reduces the computational complexity to $O(D^2)$ (due to matrix-vector multiplications with $S_q$) and allows $S_q$ and $\mu_q$ to be learned via standard backpropagation on mini-batches, solving the scalability bottleneck.

### 3.3 The Proposed Approach: Hybrid Variational LDBLL
My contribution merges the robustness of Section 3.1 with the scalability of Section 3.2.

The standard VBLL efficiently fits the data but, like the BLL, does not inherently prevent feature collapse OOD (the trace penalty in VBLL only acts on *observed* data). By injecting the Latent Derivative Regularization from LDBLL into the VBLL optimization loop, I force the variational posterior $S_q$ to support a broad distribution of gradients.

The resulting Hybrid loss function is:
$$\mathcal{L}_{Hybrid} = \underbrace{-\left( \log \mathcal{N}(y | \mu_q^T \phi, \sigma^2) - \frac{1}{2\sigma^2} \text{Tr}(\phi^T S_q \phi) \right) + D_{KL}(q(w)||p(w))}_{\text{VBLL Negative ELBO}} + \underbrace{\beta \cdot D_{KL}(\pi(z) || q(z))}_{\text{LDBLL Derivative Reg}}$$

This enables a scalable, batch-friendly BNN that maintains calibrated "bell-shaped" uncertainty in data gaps, effectively mimicking a Gaussian Process without the cubic scaling costs.

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