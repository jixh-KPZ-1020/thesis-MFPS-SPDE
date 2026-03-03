import numpy as np
import matplotlib.pyplot as plt

def mollweide_density_surface(samples, bins_lon=180, bins_lat=90, eps=1e-12):
    """
    samples: (N,3) points on unit sphere.
    Produces a 3D surface z = empirical density in (lon,lat) Mollweide coordinates.
    """
    x, y, z = samples.T
    lon = np.arctan2(y, x)                          # [-pi, pi]
    lat = np.arcsin(np.clip(z, -1.0, 1.0))          # [-pi/2, pi/2]

    # Histogram in lon-lat coordinates (this is a density w.r.t. dlon dlat)
    H, lon_edges, lat_edges = np.histogram2d(lon, lat, bins=[bins_lon, bins_lat], density=True)

    lonc = 0.5*(lon_edges[:-1] + lon_edges[1:])
    latc = 0.5*(lat_edges[:-1] + lat_edges[1:])
    LON, LAT = np.meshgrid(lonc, latc, indexing="ij")

    # Mollweide forward map: (lon,lat) -> (X,Y) in the plane.
    # Uses the standard implicit auxiliary angle theta: 2θ + sin(2θ) = π sin(lat).
    TH = np.zeros_like(LAT)
    rhs = np.pi * np.sin(LAT)
    for _ in range(12):  # Newton iterations
        f  = 2*TH + np.sin(2*TH) - rhs
        fp = 2 + 2*np.cos(2*TH)
        TH = TH - f/(fp + eps)

    X = 2*np.sqrt(2)/np.pi * LON * np.cos(TH)
    Y = np.sqrt(2) * np.sin(TH)

    # Mask outside the Mollweide ellipse: (X/(2sqrt2))^2 + (Y/sqrt2)^2 <= 1
    mask = (X/(2*np.sqrt(2)))**2 + (Y/np.sqrt(2))**2 <= 1.0 + 1e-9

    Zsurf = H.copy()
    Zsurf[~mask] = np.nan

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111, projection="3d")

    # Surface wants arrays shaped (n_lon, n_lat)
    ax.plot_surface(X, Y, Zsurf, rstride=1, cstride=1, linewidth=0, antialiased=True)

    ax.set_xlabel("Mollweide $x$")
    ax.set_ylabel("Mollweide $y$")
    ax.set_zlabel("density")
    ax.set_title("Mollweide density surface: $z = \\hat\\rho(\\lambda,\\varphi)$")
    plt.tight_layout()
    plt.show()

# Usage:
# mollweide_density_surface(samples)