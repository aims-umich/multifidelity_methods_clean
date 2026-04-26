#forrester_3f.py
#Three-fidelity Forrester benchmark using f4 (low), f3 (med), f2 (high)
#Equations from Mainini et al. (2022)
import numpy as np

# -------------------------------
# Four fidelities (f1 highest → f4 lowest)
# For now this only uses three fidelities but can be extended to 4
# Low fidelity = f4, medium fidelity = f3, and high fidelity = f2
# -------------------------------

def forrester_f1(X):
    """Highest fidelity (not used in 3f version)."""
    X = np.asarray(X).reshape(-1, 1)
    x = X[:, 0]
    return (6*x - 2)**2 * np.sin(12*x - 4)

def forrester_f2(X):
    """Second-highest fidelity (used as HIGH in 3f)."""
    X = np.asarray(X).reshape(-1, 1)
    x = X[:, 0]
    return (5.5*x - 2.5)**2 * np.sin(12*x - 4)

def forrester_f3(X):
    """Medium fidelity for 3f."""
    X = np.asarray(X).reshape(-1, 1)
    x = X[:, 0]
    return 0.75 * forrester_f1(X) + 5.0*(x - 0.5) - 2.0

def forrester_f4(X):
    """Lowest fidelity (used as LOW in 3f)."""
    X = np.asarray(X).reshape(-1, 1)
    x = X[:, 0]
    return 0.5 * forrester_f1(X) + 10.0*(x - 0.5) - 5.0


# -------------------------------
# Interface with methods
# -------------------------------

def forrester_3f_funcs():
    """
    Returns low, med, high fidelity functions and bounds.
    Using:
        low  = f4
        med  = f3
        high = f2
    """
    low  = lambda X: forrester_f4(X)
    med  = lambda X: forrester_f3(X)
    high = lambda X: forrester_f2(X)

    bounds = np.array([[0.0, 1.0]])  # domain for x

    return low, med, high, bounds


# -------------------------------
#   Manual test + plot
# -------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    low, med, high, bounds = forrester_3f_funcs()

    X = np.linspace(0, 1, 400).reshape(-1, 1)

    y_low  = low(X)
    y_med  = med(X)
    y_high = high(X)

    print("bounds:", bounds)

    plt.figure(figsize=(8,5))
    plt.plot(X, y_high, label="HIGH = f2", linewidth=2)
    plt.plot(X, y_med,  label="MED = f3", linestyle="--")
    plt.plot(X, y_low,  label="LOW = f4", linestyle="dotted")
    plt.title("Three-Fidelity Forrester Benchmark (f4, f3, f2)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



