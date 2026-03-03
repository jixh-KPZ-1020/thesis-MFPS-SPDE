import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fuBlue = (0/255, 51/255, 102/255)
# ----------------------------
# Thesis-like PGF export setup
# ----------------------------

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.unicode_minus": False,
    "figure.figsize": (5.8, 3.3),  # adjust to your thesis textwidth
    "pgf.preamble": r"""
\usepackage[T1]{fontenc}
\usepackage{libertine}
\usepackage{amsmath,amssymb,dsfont,mathrsfs,mathtools}
"""
})

# ----------------------------
# Model: double well + mobility
# ----------------------------
beta = 6.0  # increase beta if you want clearer bimodality

def F(x):
    # Double well: minima around x=±1
    return 0.25*x**4 - 0.5*x**2

def dF(x):
    return x**3 - x

def M(x):
    # positive mobility
    return 1.0 + 0.6*np.sin(x)  # min = 0.4

def dM(x):
    return 0.6*np.cos(x)

# Stratonovich SDE:
# dX = -M dF dt + sqrt(2 beta^{-1} M) ∘ dB
# Equivalent Ito SDE:
# dX = [-M dF + beta^{-1} dM] dt + sqrt(2 beta^{-1} M) dB
def drift_ito(x):
    return -M(x) * dF(x) + (1.0 / beta) * dM(x)

def sigma(x):
    return np.sqrt(2.0 * (1.0 / beta) * M(x))

# ----------------------------
# Simulation: Euler–Maruyama
# ----------------------------
dt = 5e-4
n_steps = 2_000_000
burn_in = 300_000
thin = 50

x = np.zeros(n_steps + 1)
x[0] = 1.5  # start near right well

rng = np.random.default_rng(0)
sqrt_dt = np.sqrt(dt)

for k in range(n_steps):
    xk = x[k]
    x[k + 1] = xk + drift_ito(xk) * dt + sigma(xk) * sqrt_dt * rng.normal()

samples = x[burn_in::thin]

# ----------------------------
# Target Gibbs density
# ----------------------------
grid = np.linspace(-2.8, 2.8, 2000)
unnorm = np.exp(-beta * F(grid))
Z = np.trapz(unnorm, grid)
target = unnorm / Z

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots()
ax.hist(samples, color=fuBlue, bins=160, density=True, alpha=0.55, label=r"samples")
ax.plot(grid, target, color=fuBlue, lw=1.5, label=r"target $\propto e^{-\beta \mathcal{F}}$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"density")
ax.legend(frameon=False)
fig.tight_layout()

fig.savefig("double_well_fdr.pgf")
fig.savefig("double_well_fdr.pdf")