#MF-GP: simple auto-regressive co-kriging:
# y_H(x) ≈ ρ * μ_L(x) + δ(x), where μ_L is GP fit to low-fid;
# δ is another GP fit on residuals at high-fid locations.
# This implements a two-stage GP with a learned scalar ρ and a residual GP.

#Imports
from sklearn.gaussian_process import GaussianProcessRegressor  # GP regressor implementation.
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, WhiteKernel  # Common GP kernels.
from sklearn.preprocessing import StandardScaler  # Feature scaling/standardization.
import numpy as np  # Numerical computing (arrays, random numbers, vectorized ops).

class MFGP:
    def __init__(self, kernel_low=None, kernel_res=None, alpha_low=1e-10, alpha_res=1e-10):
        """Initialize two GPs (low-level and residual) and input scalers."""
        if kernel_low is None:
            # Broadened, more permissive bounds to reduce "close to bound" warnings;
            # include a small WhiteKernel noise floor with relaxed bounds.
            kernel_low = (
                C(1.0, (1e-5, 1e5))
                * Matern(length_scale=np.ones(1), length_scale_bounds=(1e-2, 1e3), nu=2.5)
                + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
            )
        if kernel_res is None:
            # Same strategy for the residual GP kernel.
            kernel_res = (
                C(1.0, (1e-5, 1e5))
                * RBF(length_scale=np.ones(1), length_scale_bounds=(1e-2, 1e3))
                + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
            )
        # Increase optimizer restarts for more robust convergence; keep normalize_y=True.
        self.gp_low = GaussianProcessRegressor(
            kernel=kernel_low, alpha=alpha_low, normalize_y=True, n_restarts_optimizer=10
        )  # Low-fid GP.
        self.gp_res = GaussianProcessRegressor(
            kernel=kernel_res, alpha=alpha_res, normalize_y=True, n_restarts_optimizer=10
        )  # Residual GP.
        self.rho_ = None  # Scalar correlation/scale between low and high fidelities (learned).
        self.scaler_x_low = StandardScaler()  # Standardize inputs for low-fid GP.
        self.scaler_x_res = StandardScaler()  # Standardize inputs for residual GP.
        self.loss_history = None

    def fit(self, Xl, yl, Xh, yh):
        """Fit low-fidelity GP, estimate rho on high-fid sites, then fit residual GP."""
        # Fit GP to low fidelity
        Xl_s = self.scaler_x_low.fit_transform(Xl)  # Fit scaler on Xl and transform.
        self.gp_low.fit(Xl_s, yl.ravel())  # Train low-fid GP to predict yl.

        # Predict low-fid mean at high-fid points
        Xh_s_for_low = self.scaler_x_low.transform(Xh)  # Transform Xh with low-fid scaler.
        mu_L = self.gp_low.predict(Xh_s_for_low).reshape(-1, 1)  # Predict low-fid mean at high-fid inputs.

        # Estimate rho by linear regression: yh ≈ rho*mu_L
        # Use .item() to extract scalars (avoids NumPy deprecation warnings for (1,1) arrays).
        rho_num = (mu_L.T @ yh).item()  # Numerator of LS estimate (μ_L^T y_H).
        rho_den = (mu_L.T @ mu_L).item() + 1e-12  # Denominator with small epsilon for stability.
        self.rho_ = rho_num / rho_den  # Closed-form least squares estimate for scalar ρ.

        # Fit residual GP on high-fid residuals
        r = yh - self.rho_ * mu_L  # Residuals: y_H - ρ * μ_L.
        Xh_s = self.scaler_x_res.fit_transform(Xh)  # Fit residual scaler on Xh and transform.
        self.gp_res.fit(Xh_s, r.ravel())  # Train residual GP to predict residuals.
        return self  # Enable method chaining.

    def predict(self, X, return_std=False):
        """Predict high-fidelity mean (and optional std) at new inputs X."""
        # μ_H(x) = rho * μ_L(x) + μ_res(x)
        X_s_for_low = self.scaler_x_low.transform(X)  # Standardize with low-fid scaler.
        mu_L = self.gp_low.predict(X_s_for_low).reshape(-1, 1)  # Low-fid mean at X.

        X_s_for_res = self.scaler_x_res.transform(X)  # Standardize with residual scaler.
        if return_std:
            mu_res, std_res = self.gp_res.predict(X_s_for_res, return_std=True)  # Residual mean and std.
            mu = self.rho_ * mu_L.ravel() + mu_res  # Combine scaled low-fid mean + residual mean.
            return mu.reshape(-1, 1), std_res.reshape(-1, 1)  # Return mean and std (column vectors).
        else:
            mu_res = self.gp_res.predict(X_s_for_res).reshape(-1, 1)  # Residual mean only.
            mu = self.rho_ * mu_L + mu_res  # Combined prediction.
            return mu  # High-fidelity prediction.