import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

rng = np.random.default_rng(0)

def normalize(v):
    return v / np.linalg.norm(v)

def P(x):
    return np.eye(3) - np.outer(x, x)  # tangent projector

# longitude-latitude from x on S^2
def lon_lat(x):
    lam = np.arctan2(x[1], x[0])          # longitude in [-pi,pi)
    phi = np.arcsin(np.clip(x[2], -1, 1)) # latitude in [-pi/2,pi/2]
    return lam, phi

# A speed field written in Mollweide coordinates (longitude/latitude).
# Example: faster near the equator + a longitude modulation; keep it strictly positive.
a_eq = 0.8
a_lon = 0.4
m0 = 1.0

def m_of_x(x):
    lam, phi = lon_lat(x)
    return m0 * (1.0 + a_eq * np.cos(phi)**2 + a_lon * np.cos(2.0*lam))

# Ambient gradient of m(x); for simplicity use finite differences (robust and easy).
# You can replace by an analytic gradient if you want speed.
def grad_m_ambient(x, eps=1e-6):
    g = np.zeros(3)
    fx = m_of_x(x)
    for i in range(3):
        e = np.zeros(3); e[i] = 1.0
        xp = normalize(x + eps*e)
        xm = normalize(x - eps*e)
        g[i] = (m_of_x(xp) - m_of_x(xm)) / (2*eps)
    return g

# Example double-well potential on S^2: minima near +/- north pole direction mvec.
mvec = normalize(np.array([0.0, 0.0, 1.0]))
def F(x):
    c = np.dot(mvec, x)
    return 0.5*(1.0 - c**2)

def gradF_ambient(x):
    c = np.dot(mvec, x)
    return -(c) * mvec

# Parameters
beta = 6.0
dt = 5e-4
n_steps = 600_000
burn_in = 100_000
thin = 20

x = np.zeros((n_steps+1, 3))
x[0] = normalize(np.array([1.0, 0.2, 0.1]))
sqrt_dt = np.sqrt(dt)

for k in range(n_steps):
    xk = x[k]
    mk = m_of_x(xk)

    # Tangential gradients
    gradF_tan = P(xk) @ gradF_ambient(xk)
    gradm_tan = P(xk) @ grad_m_ambient(xk)

    # Ito drift corresponding to generator: -m gradF + beta^{-1} grad m (all tangential)
    drift = -mk * gradF_tan + (1.0/beta) * gradm_tan

    dB = rng.normal(size=3)
    noise = np.sqrt(2.0*mk/beta) * (P(xk) @ (sqrt_dt*dB))

    x[k+1] = normalize(xk + drift*dt + noise)

samples = x[burn_in::thin]

# Mollweide map of the empirical density on S^2
x_s, y_s, z_s = samples.T
theta = np.arccos(np.clip(z_s, -1, 1))   # polar
phi_lon = np.arctan2(y_s, x_s)           # longitude
lat = np.pi/2 - theta                    # latitude in [-pi/2, pi/2]

H, xedges, yedges = np.histogram2d(phi_lon, lat, bins=180, density=True)
Xc = 0.5*(xedges[:-1]+xedges[1:])
Yc = 0.5*(yedges[:-1]+yedges[1:])
Xm, Ym = np.meshgrid(Xc, Yc)

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111, projection="mollweide")
ax.pcolormesh(Xm, Ym, H.T, cmap="viridis")
ax.set_title("Empirical invariant density on $S^2$ (Mollweide)")
plt.show()

# Mollweide map of the speed field m(lon,lat)
lon = np.linspace(-np.pi, np.pi, 361)
latg = np.linspace(-np.pi/2, np.pi/2, 181)
LON, LAT = np.meshgrid(lon, latg)

# Convert (lon,lat) -> point on S^2
X = np.cos(LAT)*np.cos(LON)
Y = np.cos(LAT)*np.sin(LON)
Z = np.sin(LAT)

Mgrid = np.empty_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Mgrid[i,j] = m_of_x(np.array([X[i,j], Y[i,j], Z[i,j]]))

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111, projection="mollweide")
ax.pcolormesh(LON, LAT, Mgrid, cmap="viridis")
ax.set_title("Speed field $m$ on $S^2$ (Mollweide)")
plt.show()