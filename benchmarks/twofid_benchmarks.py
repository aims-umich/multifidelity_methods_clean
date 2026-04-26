#Two fidelity benchmarks
import mf2
import numpy as np

# -----------------------
# mf2 benchmarks (low/high)
# -----------------------

#This script is getting the bi-fidelity benchmarks from the mf2 package ready to be used in the multifidelity machine learning methods

def branin_funcs():
    """Return low/high fidelity callables and bounds for Branin (2D)."""
    # mf2.branin has .low and .high (vectorized over N x 2)
    #note: Lambda function is a handy function to do a simple forward process without a self-defined function
    #this line below

    #low = lambda X: mf2.branin.low(X)

    #is equivalent to:

    #def low(X):
      #return mf2.branin.low(X)

    low = lambda X: mf2.branin.low(X)  # Low-fidelity Branin function (expects N×2).
    high = lambda X: mf2.branin.high(X)  # High-fidelity Branin function.
    # Typical Branin domain: x1 ∈ [-5, 10], x2 ∈ [0, 15]
    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])  # 2D box bounds.
    return low, high, bounds  # Tuple of callables and domain.

def forrester_funcs():
    """Return low/high fidelity callables and bounds for Forrester (1D)."""
    # 1D; domain [0,1]
    low = lambda X: mf2.forrester.low(X)  # Low-fidelity Forrester (expects N×1).
    high = lambda X: mf2.forrester.high(X)  # High-fidelity Forrester.
    bounds = np.array([[0.0, 1.0]])  # 1D box bounds.
    return low, high, bounds  # Tuple of callables and domain.

def hartmann6_funcs():
    """Return low/high fidelity callables and bounds for Hartmann-6 (6D)."""
    # 6D; domain [0,1]^6  (using mf2.hartmann6 per your request)
    low = lambda X: mf2.hartmann6.low(X)  # Low-fidelity Hartmann-6 (expects N×6).
    high = lambda X: mf2.hartmann6.high(X)  # High-fidelity Hartmann-6.
    bounds = np.array([[0.0, 1.0]] * 6, dtype=float)  # 6D unit hypercube bounds.
    return low, high, bounds  # Tuple of callables and domain.

def booth_funcs():
    """Return low/high fidelity callables and bounds for Booth (2D)."""
    low = lambda X: mf2.booth.low(X)  # Low-fidelity Booth function (expects N×2).
    high = lambda X: mf2.booth.high(X)  # High-fidelity Booth function.
    # Typical booth domain: x1 ∈ [-10, 10], x2 ∈ [-10, 10]
    bounds = np.array([[-10.0, 10.0], [-10.0, 10.0]])  # 2D box bounds.
    return low, high, bounds  # Tuple of callables and domain.

def park91a_funcs():
    bounds = np.array([
        [1e-08, 1.0],  # x1
        [0.0,    1.0], # x2
        [0.0,    1.0], # x3
        [0.0,    1.0], # x4
    ], dtype=float)
    low = lambda X: mf2.park91a.low(X)
    high = lambda X: mf2.park91a.high(X)
    return low, high, bounds

def borehole_funcs():
    bounds = np.array([
        [0.05,   0.15],
        [100.0,  50000.0],
        [63070., 115600.],
        [990.,   1110.],
        [63.1,   116.0],
        [700.,   820.],
        [1120.,  1680.],
        [9855.,  12045.]
    ], dtype=float)
    low = lambda X: mf2.borehole.low(X)
    high = lambda X: mf2.borehole.high(X)
    return low, high, bounds
