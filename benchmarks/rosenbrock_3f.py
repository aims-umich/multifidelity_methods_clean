#rosenbrock_3f.py
#Three-fidelity D-dimensional Rosenbrock benchmark (Mainini et al. 2022)
#Domain: -2 <= x_i <= 2 for i = 1,...,D

import numpy as np

#-------------------------------
#Three fidelities (f1 high → f3 low)
#-------------------------------

def rosenbrock_f1(X):
    """Highest-fidelity Rosenbrock f1(x)."""
    X = np.asarray(X, dtype=float)
    X = np.atleast_2d(X)          #shape (N, D)
    x = X
    x_prev = x[:, :-1]            #x_i
    x_next = x[:, 1:]             #x_{i+1}
    term1 = 100.0 * (x_next - x_prev**2)**2
    term2 = (1.0 - x_prev)**2
    return np.sum(term1 + term2, axis=1)


def rosenbrock_f2(X):
    """Medium-fidelity Rosenbrock f2(x)."""
    X = np.asarray(X, dtype=float)
    X = np.atleast_2d(X)
    x = X
    x_prev = x[:, :-1]
    x_next = x[:, 1:]
    term1 = 50.0 * (x_next - x_prev**2)**2
    term2 = (-2.0 - x_prev)**2          #matches (-2 - x_i)^2 in the paper
    base = np.sum(term1 + term2, axis=1)
    linear = 0.5 * np.sum(x, axis=1)    #- sum_i 0.5 x_i
    return base - linear


def rosenbrock_f3(X):
    """Lowest-fidelity Rosenbrock f3(x)."""
    X = np.asarray(X, dtype=float)
    X = np.atleast_2d(X)
    x = X
    f1_vals = rosenbrock_f1(X)
    s = np.sum(x, axis=1)
    num = f1_vals - 4.0 - 0.5 * s       #f1(x) - 4 - sum_i 0.5 x_i
    #denominator sum is over all x_i (paper typo uses x1)
    den = 10.0 + 0.25 * s               #10 + sum_i 0.25 x_i
    return num / den


# -------------------------------
# Interface with methods
# -------------------------------

def rosenbrock_3f_funcs(D):
    """
    Return low/med/high fidelity callables and bounds for D-dimensional Rosenbrock.

    low(X)  -> f3 (lowest fidelity)
    med(X)  -> f2 (medium fidelity)
    high(X) -> f1 (highest fidelity)

    X is expected to be shape (N, D).
    """
    def low(X):
        return rosenbrock_f3(X)

    def med(X):
        return rosenbrock_f2(X)

    def high(X):
        return rosenbrock_f1(X)

    #D-dimensional box bounds: each x_i in [-2, 2]
    bounds = np.array([[-2.0, 2.0]] * D)

    return low, med, high, bounds


#-------------------------------
#Manual test + 3D surface plots (D=2)
#-------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  #needed for 3D projection import

    D = 2
    low, med, high, bounds = rosenbrock_3f_funcs(D)

    xs = np.linspace(-2.0, 2.0, 200)
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
    ax1.set_title("Rosenbrock f1 (high)")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")

    #f2: medium fidelity
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_surface(X1, X2, Z_med, linewidth=0, antialiased=True)
    ax2.set_title("Rosenbrock f2 (medium)")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")

    #f3: lowest fidelity
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot_surface(X1, X2, Z_low, linewidth=0, antialiased=True)
    ax3.set_title("Rosenbrock f3 (low)")
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")

    plt.tight_layout()
    plt.show()
