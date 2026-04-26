#MF-NN (delta learning): y_H = y_L_pred + NN_delta(x, y_L_pred)
#Two networks: one maps x -> y_L_pred, second maps (x, y_L_pred) -> delta.

#Imports
from sklearn.preprocessing import StandardScaler  # Feature scaling/standardization.
import torch  # PyTorch main package (tensors, device management).
from helpers_2f import MLP, train_torch_regressor, get_device
import numpy as np  # Numerical computing (arrays, random numbers, vectorized ops).

class MFNN_Delta:
    """Delta-learning multi-fidelity NN: predicts low-fid, then learns high-fid correction."""
    def __init__(self, x_dim, hidden=(128,128,128), lr=0.0005, epochs=500, wd=0.0, verbose=False):
        self.device = get_device()  # single source of truth: GPU if available, else CPU
        self.low = MLP(in_dim=x_dim, hidden=hidden, out_dim=1)
        self.delta = MLP(in_dim=x_dim+1, hidden=hidden, out_dim=1)
        self.lr, self.epochs, self.wd, self.verbose = lr, epochs, wd, verbose
        self.scaler_x_low = StandardScaler()
        self.scaler_x_delta = StandardScaler()

    def fit(self, Xl, yl, Xh, yh):
        """Train low-fid network, then delta network on high-fid residuals."""
        device = self.device
        # Train low-fid predictor
        Xl_s = self.scaler_x_low.fit_transform(Xl)
        self.low, self.low_loss_history = train_torch_regressor(
            self.low, Xl_s, yl, lr=self.lr, epochs=self.epochs,
            weight_decay=self.wd, verbose=self.verbose, device=device)

        # Prepare delta training — model is already on device after train_torch_regressor
        with torch.no_grad():
            mu_Lh = self.low(
                torch.tensor(self.scaler_x_low.transform(Xh), dtype=torch.float32, device=device)
            ).cpu().numpy()
        r = yh - mu_Lh  # residuals
        Xh_aug = np.hstack([Xh, mu_Lh])

        Xh_aug_s = self.scaler_x_delta.fit_transform(Xh_aug)
        self.delta, self.delta_loss_history = train_torch_regressor(
            self.delta, Xh_aug_s, r, lr=self.lr, epochs=self.epochs,
            weight_decay=self.wd, verbose=self.verbose, device=device)
        return self

    def predict(self, X):
        """Predict high-fidelity by low-fid prediction + learned delta."""
        device = self.device
        with torch.no_grad():
            mu_L = self.low(
                torch.tensor(self.scaler_x_low.transform(X), dtype=torch.float32, device=device)
            ).cpu().numpy()
            X_aug = np.hstack([X, mu_L])
            mu_delta = self.delta(
                torch.tensor(self.scaler_x_delta.transform(X_aug), dtype=torch.float32, device=device)
            ).cpu().numpy()
        return mu_L + mu_delta