# Mathematical Derivations: MSE vs Cross-Entropy for Softmax Classification

## 1. Mean Squared Error (MSE)

For one sample with logits $\mathbf{z} \in \mathbb{R}^K$, softmax probabilities $\mathbf{p}$, and one-hot target $\mathbf{y}$:

$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}
$$

Batch MSE loss over $N$ samples:

$$
\mathcal{L}_{\mathrm{MSE}} = \frac{1}{N} \sum_{n=1}^{N} \sum_{k=1}^{K} \left(y_{nk} - p_{nk}\right)^2
$$

For one sample (dropping index $n$ for readability):

$$
\ell_{\mathrm{MSE}} = \sum_{i=1}^{K} (p_i - y_i)^2
$$

Gradient wrt probability $p_i$:

$$
\frac{\partial \ell_{\mathrm{MSE}}}{\partial p_i} = 2(p_i - y_i)
$$

Softmax Jacobian:

$$
\frac{\partial p_i}{\partial z_k} = p_i(\delta_{ik} - p_k)
$$

Chain rule to logits:

$$
\frac{\partial \ell_{\mathrm{MSE}}}{\partial z_k}
= \sum_{i=1}^{K} \frac{\partial \ell_{\mathrm{MSE}}}{\partial p_i}
\frac{\partial p_i}{\partial z_k}
= \sum_{i=1}^{K} 2(p_i-y_i)\,p_i(\delta_{ik}-p_k)
$$

Including batch averaging:

$$
\frac{\partial \mathcal{L}_{\mathrm{MSE}}}{\partial z_k}
= \frac{2}{N}\sum_{n=1}^{N}\sum_{i=1}^{K}(p_{ni}-y_{ni})\,p_{ni}(\delta_{ik}-p_{nk})
$$

Equivalent index form (as often written in notes):

$$
\frac{\partial \mathcal{L}_{\mathrm{MSE}}}{\partial z_k}
= \frac{2}{N}\sum_{i}(p_i-y_i)\,p_k(\delta_{ik}-p_i)
$$

### Saturation intuition

When a wrong class becomes highly confident (e.g., $p_k \to 1$ while $y_k=0$), the error term $(p_k-y_k)$ is large, but softmax sensitivity contains factors like $p_k(1-p_k)$ or $p_i p_k$, which go toward zero. So the chain-rule product can shrink despite large prediction error, causing gradient saturation.

---

## 2. Cross-Entropy (CE)

Softmax:

$$
p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

Batch CE loss:

$$
\mathcal{L}_{\mathrm{CE}} = -\frac{1}{N}\sum_{n=1}^{N}\sum_{k=1}^{K} y_{nk}\log p_{nk}
$$

For one sample:

$$
\ell_{\mathrm{CE}} = -\sum_{i=1}^{K} y_i \log p_i
$$

Differentiate wrt $z_k$:

$$
\frac{\partial \ell_{\mathrm{CE}}}{\partial z_k}
= -\sum_{i=1}^{K} y_i \frac{1}{p_i}\frac{\partial p_i}{\partial z_k}
= -\sum_{i=1}^{K} y_i\frac{1}{p_i}p_i(\delta_{ik}-p_k)
$$

$$
\frac{\partial \ell_{\mathrm{CE}}}{\partial z_k}
= -\sum_{i=1}^{K} y_i(\delta_{ik}-p_k)
= -y_k + p_k\sum_{i=1}^{K}y_i
$$

Because one-hot labels satisfy $\sum_i y_i = 1$:

$$
\frac{\partial \ell_{\mathrm{CE}}}{\partial z_k} = p_k - y_k
$$

Including batch averaging:

$$
\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial z_k}
= \frac{1}{N}(p_k-y_k)
= -\frac{1}{N}(y_k-p_k)
$$

This is the key cancellation: $1/p_i$ from $\log$ derivative cancels $p_i$ from the softmax Jacobian.

---

## 3. Analytical Comparison

| Property | MSE | Cross-Entropy |
|---|---|---|
| Gradient wrt logits | $\frac{2}{N}\sum_i (p_i-y_i)p_i(\delta_{ik}-p_k)$ | $\frac{1}{N}(p_k-y_k)$ |
| Dependence on softmax Jacobian | Explicit multiplicative factor $p(1-p)$ / $p_ip_k$ | Cancels out in final form |
| Error signal form | Nonlinear and damped near simplex boundaries | Linear in error $(p-y)$ |
| Saturation risk | High near $p\to0$ or $p\to1$ | Much lower for badly wrong predictions |

### Proof sketch: CE gradient is linear in prediction error

From derivation above,

$$
\nabla_{\mathbf{z}}\mathcal{L}_{\mathrm{CE}} = \frac{1}{N}(\mathbf{p}-\mathbf{y})
$$

which is affine/linear in per-class error components.

### Proof sketch: MSE includes saturating softmax sensitivity

MSE gradient contains Jacobian terms $\partial p_i/\partial z_k = p_i(\delta_{ik}-p_k)$, so each component is scaled by probabilities themselves. As probabilities approach boundaries ($0$ or $1$), those factors can vanish, suppressing updates.

### Numerical example (binary-style intuition on correct-class logit)

Let $p_{\text{correct}}$ denote the correct-class probability and $y_{\text{correct}}=1$.

1. Very wrong prediction: $p_{\text{correct}}=0.01$

- CE magnitude on correct logit:

$$
|p-y| = |0.01-1| = 0.99
$$

- MSE-like damped magnitude (error $\times$ softmax slope):

$$
|p-y|\,p(1-p) \approx 0.99\times0.01\times0.99 \approx 0.0098 \approx 0.0099
$$

2. Mostly correct prediction: $p_{\text{correct}}=0.99$

- CE magnitude:

$$
|p-y| = |0.99-1| = 0.01
$$

- MSE-like damped magnitude:

$$
|p-y|\,p(1-p) \approx 0.01\times0.99\times0.01 \approx 0.000099
$$

Interpretation: CE remains strong when predictions are badly wrong, while MSE can become tiny due to probability-dependent damping.
