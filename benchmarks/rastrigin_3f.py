#rastrigin_3f.py
#Three-fidelity shifted-rotated Rastrigin benchmark (Mainini et al. 2022)
#Domain: -0.1 <= x_i <= 0.2 for i = 1,...,D

import numpy as np

#-------------------------------
#Helpers: shift and rotation
#-------------------------------

def _compute_z(X, theta=0.2):
    """
    Compute shifted-rotated coordinates z = R(theta) (x - x*).

    X: array-like, shape (N, D)
    theta: rotation angle (radians), default 0.2
    """
    X = np.asarray(X, dtype=float)
    X = np.atleast_2d(X)
    N, D = X.shape

    #shift: x* = 0.1 * 1_D
    x_star = 0.1 * np.ones(D)
    Y = X - x_star

    #rotation:
    #for D >= 2, rotate the first two coordinates with 2D rotation matrix
    #for D == 1, this just returns the shifted coordinate
    if D >= 2:
        c = np.cos(theta)
        s = np.sin(theta)
        R2 = np.array([[c, -s],
                       [s,  c]])
        Z = Y.copy()
        Z[:, :2] = Y[:, :2] @ R2.T
    else:
        Z = Y

    return Z


#-------------------------------
#Base Rastrigin and resolution error
#-------------------------------

def rastrigin_f1(X, theta=0.2):
    """Highest-fidelity base Rastrigin f1(z)."""
    z = _compute_z(X, theta)
    return np.sum(z**2 + 1.0 - np.cos(10.0 * np.pi * z), axis=1)


def rastrigin_error(X, phi, theta=0.2):
    """Resolution error e_r(z, phi)."""
    z = _compute_z(X, theta)

    Theta = 1.0 - 0.0001 * phi
    a = Theta
    w = 10.0 * np.pi * Theta
    b = 0.5 * np.pi * Theta

    arg = w * z + b + np.pi
    return a * np.sum(np.cos(arg)**2, axis=1)


def rastrigin_f_with_phi(X, phi, theta=0.2):
    """General fidelity definition f(x; phi) = f1(z) + e_r(z, phi)."""
    return rastrigin_f1(X, theta) + rastrigin_error(X, phi, theta)


#-------------------------------
#Three fidelities (f1 high → f3 low)
#-------------------------------

#phi values per Mainini/Rumpfkeil
_PHI_HIGH = 10000.0   #high-fidelity
_PHI_MED  = 5000.0    #medium-fidelity
_PHI_LOW  = 2500.0    #low-fidelity


def rastrigin_f1_high(X, theta=0.2):
    """High-fidelity Rastrigin f1 (phi = 10000)."""
    return rastrigin_f_with_phi(X, _PHI_HIGH, theta)


def rastrigin_f2_med(X, theta=0.2):
    """Medium-fidelity Rastrigin f2 (phi = 5000)."""
    return rastrigin_f_with_phi(X, _PHI_MED, theta)


def rastrigin_f3_low(X, theta=0.2):
    """Low-fidelity Rastrigin f3 (phi = 2500)."""
    return rastrigin_f_with_phi(X, _PHI_LOW, theta)


# -------------------------------
# Interface with methods
# -------------------------------

def rastrigin_3f_funcs(D, theta=0.2):
    """
    Return low/med/high fidelity callables and bounds for D-dimensional
    shifted-rotated Rastrigin.

    low(X)  -> f3 (lowest fidelity, phi=2500)
    med(X)  -> f2 (medium fidelity, phi=5000)
    high(X) -> f1 (highest fidelity, phi=10000)

    X is expected to be shape (N, D).
    """
    def low(X):
        return rastrigin_f3_low(X, theta)

    def med(X):
        return rastrigin_f2_med(X, theta)

    def high(X):
        return rastrigin_f1_high(X, theta)

    #D-dimensional box bounds: each x_i in [-0.1, 0.2]
    bounds = np.array([[-0.1, 0.2]] * D)

    return low, med, high, bounds


#-------------------------------
#Manual test + 3D surface plots (D=2)
#-------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  #needed for 3D projection import

    D = 2
    low, med, high, bounds = rastrigin_3f_funcs(D)

    xs = np.linspace(-0.1, 0.2, 200)
    X1, X2 = np.meshgrid(xs, xs)
    grid = np.stack([X1.ravel(), X2.ravel()], axis=1)  #shape (N, 2)

    Z_high = high(grid).reshape(X1.shape)
    Z_med  = med(grid).reshape(X1.shape)
    Z_low  = low(grid).reshape(X1.shape)

    print("bounds:\n", bounds)

    fig = plt.figure(figsize=(12, 4))

    #f1: highest fidelity
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(X1, X2, Z_high, linewidth=0, antialiased=True)
    ax1.set_title("Rastrigin f1 (high)")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")

    #f2: medium fidelity
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_surface(X1, X2, Z_med, linewidth=0, antialiased=True)
    ax2.set_title("Rastrigin f2 (medium)")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")

    #f3: lowest fidelity
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot_surface(X1, X2, Z_low, linewidth=0, antialiased=True)
    ax3.set_title("Rastrigin f3 (low)")
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")

    plt.tight_layout()
    plt.show()
