# Differentiable Volume Rendering for Multi‑Modal NIfTI

---

## Notation

- $x \in \mathbb{R}^3$ — world coordinate.  
- $i \in \mathbb{Z}^3$ — voxel index coordinate.  
- $A \in \mathbb{R}^{4\times 4}$ — NIfTI affine mapping indices $\to$ world.  
  $$x = A \begin{bmatrix} i \\ 1 \end{bmatrix}.$$
- $s(x) \in \mathbb{R}^M$ — multi‑modal scalar vector at $x$ (e.g., FLAIR, T1, T2, SWI, FA, MD).  
- Ray: $r(t) = o + t d$, $o\in\mathbb{R}^3$ origin, $d\in\mathbb{R}^3$ unit direction, $t\in[t_0,t_1]$.  
- $\sigma(x)\ge 0$ — extinction (attenuation) coefficient at $x$.  
- $c(x)\in[0,1]^3$ — emitted color/radiance at $x$.  
- Transmittance:
  $$T(t) = \exp\!\Big(-\int_{t_0}^{t} \sigma(r(u))\,du\Big).$$
- Discrete step $\Delta t$; per-step opacity:
  $$\alpha = 1 - \exp(-\sigma \Delta t).$$
- POIs: $P=\{p_k\}_{k=1}^K$, optionally with radii $r_k$ and amplitudes $a_k$.  
- POI influence field:
  $$g(x;P) = \sum_{k=1}^K a_k \, K\!\left(\frac{\|x-p_k\|}{r_k}\right),$$
  where $K(\cdot)$ is a smooth kernel (e.g. Gaussian).

---

## 1 — Multi‑modal sampling (NIfTI → world → trilinear)

Map world $x$ to voxel coordinates:
$$i(x) = A^{-1}\begin{bmatrix} x \\ 1 \end{bmatrix}.$$
Sample vector from $M$ modality volumes $V^m$ using trilinear interpolation:
$$s(x) = \text{trilinear}\big(V^1,\dots,V^M;\,i(x)\big) \in \mathbb{R}^M.$$
Trilinear sampling is linear in voxel values, so gradients w.r.t. the voxel grid are simple weighted sums.

---

## 2 — Learned per-location networks (color, sigma, importance)

Parameterize optical fields with differentiable networks (MLPs / small conv nets):

$$
\sigma(x) = \sigma_{\theta_\sigma}\big(s(x),\,x,\,g(x;P)\big), \qquad \sigma\ge 0\quad(\text{use softplus/exp})
$$
$$
c(x) = c_{\theta_c}\big(s(x),\,x,\,g(x;P)\big)\in\mathbb{R}^3
$$
$$
m(x) = m_{\theta_I}\big(s(x),\,x,\,g(x;P)\big)\ge 0\quad(\text{importance})
$$

All networks are differentiable; $g$ conditions them on POIs.

---

## 3 — Continuous volume-rendering integral (emission–absorption)

Exact continuous model (no scattering):
$$C = \int_{t_0}^{t_1} T(t)\, c\big(r(t)\big)\, \sigma\big(r(t)\big)\, dt, \qquad
T(t)=\exp\!\Big(-\int_{t_0}^{t}\sigma(r(u))\,du\Big).$$

Derivative w.r.t. parameter $\theta$:
$$\frac{dC}{d\theta}
= \int_{t_0}^{t_1} \Big[ \big(\partial_\theta T\big) c \sigma + T (\partial_\theta c)\sigma + T c(\partial_\theta \sigma) \Big] dt,$$
with
$$\partial_\theta T(t) = - T(t) \int_{t_0}^{t} \partial_\theta \sigma(r(u))\, du.$$

---

## 4 — Discrete ray marching (front‑to‑back)

Uniform samples $t_i = t_0 + (i-1)\Delta t$, $x_i = r(t_i)$. Let $\sigma_i=\sigma(x_i)$, $c_i=c(x_i)$, $\tau_i=\sigma_i\Delta t$, $\alpha_i = 1-e^{-\tau_i}$.

Initialize $T_0=1$. For $i=1\ldots N$:
$$
C \mathrel{+}= T_{i-1}\, c_i\, \alpha_i,
$$
$$
T_i = T_{i-1}\, (1-\alpha_i).
$$

Equivalently:
$$C=\sum_{i=1}^N \Big(\prod_{j<i} (1-\alpha_j)\Big) \, (c_i \,\alpha_i).$$

Alpha gradient:
$$\frac{\partial \alpha_i}{\partial \sigma_i} = \Delta t\, e^{-\sigma_i \Delta t} = \Delta t (1-\alpha_i).$$

---

## 5 — Efficient reverse‑mode backprop (discrete)

Let the upstream gradient be $G=\partial L/\partial C$ (vector for RGB). Store forward arrays $T_{i-1}, c_i, \alpha_i$.

Backward (O(N)):
Initialize adjoints $\tilde T_N=0$, and zero arrays $\tilde c_i,\tilde\alpha_i$.
For $i=N\ldots 1$:
$$
\tilde c_i \mathrel{+}= G \odot (T_{i-1}\alpha_i) \quad\text{(RGB-vector)}\\
$$
$$\tilde \alpha_i \mathrel{+}= G\cdot (T_{i-1} c_i) \quad\text{(dot over RGB)}$$
$$\tilde T_{i-1} \mathrel{+}= G\cdot (c_i \alpha_i)$$

Propagate $\tilde T$ through recurrence $T_{k}=T_{k-1}(1-\alpha_k)$:
for $k=i-1$ downwards:
$$
\tilde \alpha_k \mathrel{+}= -\tilde T_k \, T_{k-1}
$$
$$
\tilde T_{k-1} \mathrel{+}= \tilde T_k \, (1-\alpha_k)
$$

Finally chain $\tilde\alpha_i \to \tilde\sigma_i$:
$$\tilde\sigma_i = \tilde\alpha_i \cdot \Delta t\, e^{-\sigma_i\Delta t}.$$
Then backprop $\tilde\sigma_i,\tilde c_i$ through $\sigma_{\theta_\sigma}, c_{\theta_c}$ and the sampling operation.

---

## 6 — Differentiating trilinear sampling & gradients to voxels

Trilinear interpolation:
$$s(x) = \sum_{n=1}^8 w_n(x)\, v_n,\qquad \sum_n w_n(x)=1.$$
Then
$$
\frac{\partial s}{\partial v_n} = w_n(x),
$$
$$
\frac{\partial s}{\partial x} = \sum_{n=1}^8 v_n\, \frac{\partial w_n}{\partial x}.
$$
Backprop to voxel grid: accumulate $\tilde v_n += \tilde s(x)\, w_n(x)$. Gradient to sample position $\tilde x = (\partial s/\partial x)^\top \tilde s$ flows to ray parameters.

---

## 7 — Differentiable adaptive sampling (coarse→fine via inverse‑CDF)

Coarse stage: sample $\{t^c_k\}_{k=1}^K$ uniformly; compute importance weights
$$w_k = m\big(r(t^c_k)\big)\ge0.$$
Construct piecewise linear cumulative:
$$W_0=0,\quad W_k=\sum_{\ell=1}^k w_\ell,\quad F(t)=\frac{W(t)}{W_K}.$$

Fine sampling: choose deterministic quantiles $u_j\in(0,1)$ (e.g. $(j-0.5)/J$); find
$$t^f_j = F^{-1}(u_j)$$
by linear interpolation inside the coarse bins.

Implicit differentiation for fixed $u$:
$$F\big(t(u;w);w\big)=u \quad\Rightarrow\quad
\frac{\partial t}{\partial\theta} = -\frac{\partial F/\partial\theta}{\partial F/\partial t}.$$
Inside a coarse bin, $\partial F/\partial t = w(t)/W_K$ (local density over total), and $\partial F/\partial\theta$ depends only on the neighboring $w_k$ coefficients — so $\partial t^f_j/\partial w_k$ is sparse and computable. Then chain via $w_k=m(\cdot)$ into network parameters and POIs.

When $t^f_j$ changes, sample position changes:
$$x^f_j = o + t^f_j d,\quad \frac{\partial x^f_j}{\partial t^f_j} = d.$$

---

## 8 — Differentiating POI positions $p_k$

Given
$$g(x;P) = \sum_k a_k K\!\left(\frac{\|x-p_k\|}{r_k}\right),$$
we have
$$\frac{\partial g}{\partial p_k}
= a_k\, K'\!\left(\frac{\|x-p_k\|}{r_k}\right)\cdot
\left(-\frac{x-p_k}{r_k\|x-p_k\|}\right).$$

Backprop contribution to $\partial L/\partial p_k$ collects per-sample terms:
$$\frac{\partial L}{\partial p_k}
= \sum_{i\in\text{samples}} \Big(
\tilde\sigma_i \frac{\partial\sigma_i}{\partial g}
+ \tilde c_i\!\cdot\!\frac{\partial c_i}{\partial g}
+ \tilde m_i \frac{\partial m_i}{\partial g}
\Big)\,\frac{\partial g(x_i;P)}{\partial p_k},$$
plus terms from inverse‑CDF dependence $\partial t^f/\partial p_k$ if applicable.

---

## 9 — Derivatives w.r.t. ray origin $o$ and direction $d$

If $x_i=o+t_i d$ with fixed $t_i$,
$$
\frac{\partial x_i}{\partial o} = I_3,
$$
$$
\frac{\partial x_i}{\partial d} = t_i I_3.
$$
So
$$
\frac{\partial L}{\partial o} = \sum_i \frac{\partial L}{\partial x_i},
$$
$$
\frac{\partial L}{\partial d} = \sum_i t_i\,\frac{\partial L}{\partial x_i}.
$$
If $t_i$ depend on $o,d$ (e.g. intersection times or inverse‑CDF), include $\partial t_i/\partial(\cdot)$ via implicit differentiation.

---

## 10 — Isosurface / root finding (differentiable intersection)

For isovalue $s_0$, solve for $t^*$ such that $f(t)=s\big(r(t)\big)$ and $f(t^*)=s_0$.
By implicit function theorem:
$$\frac{dt^*}{d\theta} = -\frac{\partial f/\partial\theta (t^*)}{\partial f/\partial t (t^*)}
= -\frac{\partial_\theta s(x(t^*); \theta)}{\nabla s(x(t^*))\cdot d}.$$
In practice, backprop through Newton iterations either by unrolling iterations or by using this implicit formula.

---

## 11 — Smoothing non‑differentiable ops

Replace hard thresholds with smooth approximations:
- Hard step $\mathbf{1}_{s>s_0}$ → $\sigma\big(\beta(s-s_0)\big)$ (sigmoid).
- Hard empty‑space skipping → continuous occupancy $o(x)\in[0,1]$ learned and used multiplicatively.

---

## 12 — Practical tips (brief)

- Use $\alpha = 1 - \exp(-\sigma\Delta t)$ for numerical stability. For small $\tau=\sigma\Delta t$, use series expansion $\alpha\approx \tau - \tau^2/2$ if needed.
- Account for voxel spacing (anisotropy) when converting $\Delta t$ in world units to voxel indices.
- Memory: naive reverse-mode stores O(N) per ray; use checkpointing to trade compute↔memory.
- Regularizers: smoothness on $\sigma$, sparsity on $m(x)$, coverage loss to avoid collapsed importance.
- Preprocess NIfTI: register modalities to same grid, skull strip, correct bias field, normalize intensity per modality.

---

## 13 — Key formula summary

Continuous:
$$C=\int_{t_0}^{t_1}\exp\!\Big(-\int_{t_0}^{t}\sigma(r(u))\,du\Big)\,c(r(t))\,\sigma(r(t))\,dt.$$

Discrete:
$$
\alpha_i = 1-e^{-\sigma_i \delta t_i},\quad T_{0=1}
$$
$$
C = \sum_{i=1}^N T_{i-1} c_i \alpha_i,\quad T_i=T_{i-1}(1-\alpha_i).
$$

Inverse‑CDF derivative (implicit):
$$\frac{\partial t}{\partial\theta} = -\frac{\partial F/\partial\theta}{\partial F/\partial t}.$$