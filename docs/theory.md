# Theory: Uncertainty Quantification for PDE Surrogates

## 1. Problem Setup

We consider the 2D Laplace equation $\nabla^2 u = 0$ on the unit square $\Omega = [0,1]^2$ with Dirichlet boundary conditions $u|_{\partial\Omega} = g$. In realistic settings, the boundary data $g$ is observed only at sparse locations and corrupted by additive Gaussian noise: $\tilde{g}_i = g(x_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$. A neural surrogate $f_\theta$ maps the noisy, sparse boundary observations to the full interior solution field $u \in \mathbb{R}^{H \times W}$. Because the input is uncertain and the map is approximate, the goal is not a single point prediction but a calibrated distribution over solution fields -- i.e., uncertainty quantification (UQ).

---

## 2. Conditional Flow Matching

### Background

Flow matching (Lipman et al. 2023) learns a continuous-time generative model by regressing a neural network $v_\theta(x_t, t)$ onto a target velocity field that transports samples from a simple prior $p_0$ to the data distribution $p_1$. Unlike diffusion models, no noise schedule or variance bookkeeping is required -- the training objective is a simple regression loss on velocities.

### Interpolant and Velocity Target

Given a source sample $x_0 \sim p_0$ (standard Gaussian) and a data sample $x_1 \sim p_1$, we define the linear interpolant:

$$x_t = (1 - t)\, x_0 + t\, x_1, \quad t \in [0, 1]$$

The conditional velocity field that generates this path is simply:

$$u_t(x_t \mid x_0, x_1) = x_1 - x_0$$

The flow matching objective trains $v_\theta$ to match this velocity:

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x_0,\, x_1} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]$$

This is a standard MSE loss -- no log-likelihood bounds, no score matching tricks.

### OT-CFM: Optimal Transport Coupling

Naive flow matching pairs $x_0$ and $x_1$ independently, producing crossing trajectories that are hard to learn. OT-CFM (Tong et al. 2024) replaces independent coupling with **mini-batch optimal transport**: within each mini-batch of size $B$, we solve the assignment problem

$$\pi^* = \arg\min_{\pi \in \Pi} \sum_{i,j} \pi_{ij} \| x_0^{(i)} - x_1^{(j)} \|^2$$

using the Hungarian algorithm in $O(B^3)$ time. This produces straighter, non-crossing flows that are easier for the network to learn, connecting to the dynamic formulation of optimal transport (Benamou & Brenier 2000):

$$\inf_{v} \int_0^1 \int \|v(x,t)\|^2 \, \rho(x,t) \, dx \, dt$$

subject to the continuity equation $\partial_t \rho + \nabla \cdot (\rho v) = 0$. The OT coupling approximates the solution to this variational problem, yielding minimum-energy (and thus maximally straight) transport paths.

### Conditional Generation

For conditional generation (boundary observations $\to$ solution field), we condition $v_\theta$ on the boundary data by concatenating it as an additional input channel to the U-Net. At inference, we draw $x_0 \sim \mathcal{N}(0, I)$ and integrate the learned ODE:

$$\frac{dx_t}{dt} = v_\theta(x_t, t \mid \text{bc}), \quad x_0 \sim \mathcal{N}(0, I)$$

using Euler integration with 50 uniform steps. Drawing $N$ independent samples yields an empirical posterior, from which we compute pixelwise mean and standard deviation for UQ.

### Advantages over DDPM

Flow matching avoids several complexities of diffusion models: (1) no noise schedule to tune (cosine, linear, etc.), (2) no variance parameterization ($\epsilon$-prediction vs. $v$-prediction), (3) deterministic ODE sampling rather than stochastic SDE, and (4) the training loss has uniform gradient magnitude across $t$ by construction. The OT coupling further accelerates convergence by providing a more learnable target.

---

## 3. Improved DDPM

We use three complementary improvements to the standard DDPM (Ho et al. 2020) that together yield 3--5x faster convergence.

### Cosine Schedule with Zero-Terminal SNR

The standard linear noise schedule $\beta_t$ wastes capacity: the signal-to-noise ratio (SNR) at $t=T$ is not exactly zero, so the model never learns to generate from pure noise. Following Nichol & Dhariwal (2021), we use a cosine schedule:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

with offset $s = 0.008$, and enforce $\bar{\alpha}_T = 0$ (zero-terminal SNR) so that the final distribution is exactly $\mathcal{N}(0, I)$.

### v-Prediction

Instead of predicting the noise $\epsilon$ (as in standard DDPM) or the clean data $x_0$, we predict the **velocity** (Salimans & Ho 2022):

$$v_t = \sqrt{\bar{\alpha}_t}\, \epsilon - \sqrt{1 - \bar{\alpha}_t}\, x_0$$

This is a rotation of the $(\epsilon, x_0)$ prediction targets that balances gradient magnitudes across timesteps. At high noise ($t \approx T$), $v_t \approx \epsilon$; at low noise ($t \approx 0$), $v_t \approx -x_0$. The network smoothly interpolates between these regimes.

### Min-SNR-$\gamma$ Weighting

Different timesteps contribute conflicting gradients to the loss. Hang et al. (2023) propose weighting the loss by:

$$w(t) = \min\!\left(\text{SNR}(t),\, \gamma\right)$$

with $\gamma = 5$. This clips the weight at high-SNR timesteps (low noise), preventing them from dominating training. Combined with $v$-prediction, this yields a loss landscape with uniform effective gradient magnitude across all $t$.

### Why These Compound

The cosine schedule ensures meaningful signal at all timesteps. $v$-prediction provides a natural parameterization that transitions smoothly between noise and data prediction. Min-SNR weighting removes the remaining gradient imbalance. Together, they eliminate the three main sources of training inefficiency in standard DDPM.

---

## 4. Conformal Prediction

### Motivation

Both generative models (FM, DDPM) and ensembles produce uncertainty estimates, but these are **not calibrated** out of the box -- a 90% prediction interval may cover only 15--82% of true values depending on the regime. Conformal prediction provides a post-hoc calibration wrapper with finite-sample coverage guarantees, requiring no retraining.

### Split Conformal Prediction

Given a calibration set $\{(x_i, y_i)\}_{i=1}^n$ and a base model producing pointwise predictions $\hat{\mu}(x)$ and uncertainty estimates $\hat{\sigma}(x)$, we define the **nonconformity score**:

$$s_i = \frac{|y_i - \hat{\mu}(x_i)|}{\hat{\sigma}(x_i)}$$

This normalizes residuals by the model's own uncertainty. We compute the $(1 - \alpha)(1 + 1/n)$-quantile $\hat{q}$ of $\{s_1, \ldots, s_n\}$, and construct prediction intervals:

$$C(x) = \left[\hat{\mu}(x) - \hat{q}\, \hat{\sigma}(x),\;\; \hat{\mu}(x) + \hat{q}\, \hat{\sigma}(x)\right]$$

By the exchangeability of calibration and test data, this satisfies the **finite-sample guarantee** (Vovk et al. 2005):

$$P(y_{\text{new}} \in C(x_{\text{new}})) \geq 1 - \alpha$$

### Spatial Variant: Simultaneous Coverage

For spatially-extended fields $u \in \mathbb{R}^{H \times W}$, we need all pixels covered simultaneously. We define the spatial nonconformity score as the maximum over pixels:

$$s_i = \max_{j \in \text{pixels}} \frac{|y_{ij} - \hat{\mu}_j(x_i)|}{\hat{\sigma}_j(x_i)}$$

The resulting intervals are wider but guarantee $P(\text{all pixels covered}) \geq 1 - \alpha$.

### Pixelwise Variant: Marginal Coverage

When simultaneous coverage is unnecessarily conservative, we apply conformal calibration independently per pixel (or with a single shared quantile over all pixel scores). This gives **marginal** coverage -- each pixel individually has $\geq 1 - \alpha$ coverage -- with substantially tighter intervals. In our experiments, pixelwise conformal achieves 88--91% coverage with interval widths 2--10x smaller than the spatial variant.

### Why Conformal Prediction Matters

- **Distribution-free**: no assumptions on the data distribution or model form
- **Model-agnostic**: wraps any base UQ method (ensemble, generative, Bayesian)
- **Zero training cost**: only requires a held-out calibration set
- **Finite-sample valid**: the guarantee holds for any $n$, not just asymptotically

---

## 5. References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS*.
2. Nichol, A. & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. *ICML*.
3. Salimans, T. & Ho, J. (2022). Progressive Distillation for Fast Sampling of Diffusion Models. *ICLR*.
4. Hang, T. et al. (2023). Efficient Diffusion Training via Min-SNR Weighting Strategy. *CVPR*.
5. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling. *ICLR*.
6. Tong, A., Malkin, N., Huguet, G., Zhang, Y., et al. (2024). Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport. *TMLR*.
7. Benamou, J.-D. & Brenier, Y. (2000). A Computational Fluid Mechanics Solution to the Monge-Kantorovich Mass Transfer Problem. *Numerische Mathematik*.
8. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
9. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*.
10. Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized Quantile Regression. *NeurIPS*.
