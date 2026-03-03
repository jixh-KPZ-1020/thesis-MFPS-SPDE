import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Geometric Langevin on S^2
# ----------------------------
beta = 6.0
dt = 5e-4
n_steps = 800_000
burn_in = 100_000
thin = 20

rng = np.random.default_rng(0)

def normalize(v):
    return v / np.linalg.norm(v)

def P(x):
    # Tangent projector at x on S^2
    return np.eye(3) - np.outer(x, x)

# Potential on the sphere: two wells near ±m (a "dipole"-like energy)
m = normalize(np.array([0.0, 0.0, 1.0]))
def F(x):
    # minima at ±m; constant shift irrelevant
    c = np.dot(m, x)
    return 0.5*(1.0 - c**2)

def gradF_ambient(x):
    # Extend F to R^3 via same formula and take Euclidean gradient
    c = np.dot(m, x)
    return -(c) * m  # gradient of 0.5*(1 - (m·x)^2) is -(m·x) m

# Stratonovich SDE on S^2 (extrinsic form):
# dX = - P_X gradF dt + sqrt(2 beta^{-1}) P_X ∘ dB
# Invariant measure: ∝ exp(-beta F(x)) dσ(x) on the sphere.

x = np.zeros((n_steps+1, 3))
x[0] = normalize(np.array([1.0, 0.2, 0.1]))

sqrt_dt = np.sqrt(dt)
noise_scale = np.sqrt(2.0/beta)

for k in range(n_steps):
    xk = x[k]
    drift = - P(xk) @ gradF_ambient(xk)
    dB = rng.normal(size=3)
    # Stratonovich-consistent for embedded manifolds: project noise in tangent, then renormalize
    xkp = xk + drift*dt + noise_scale * (P(xk) @ (sqrt_dt*dB))
    x[k+1] = normalize(xkp)

samples = x[burn_in::thin]

# ----------------------------
# Diagnostics: histogram of z = m·x should be bimodal near ±1
# ----------------------------
z = samples @ m

plt.figure(figsize=(7.2, 3.8))
plt.hist(z, bins=120, density=True, alpha=0.65)
plt.xlabel(r"$z = m\cdot x$")
plt.ylabel("density")
plt.title(r"Geometric Langevin on $S^2$: equilibrium in the coordinate $z$")
plt.tight_layout()
plt.show()

print("E[z] =", float(np.mean(z)))
print("E[z^2] =", float(np.mean(z**2)))