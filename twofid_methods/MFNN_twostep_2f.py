#2 step MFNN
from sklearn.preprocessing import StandardScaler
import torch
from helpers_2f import MLP, train_torch_regressor, get_device
import numpy as np

class MFNN_2step:
    def __init__(self, x_dim, out_dim=1, hidden=(128,128,128),
                 lr=0.001, epochs=500, wd=0.0, verbose=False):
        self.out_dim = out_dim
        self.device  = get_device()
        self.low     = MLP(in_dim=x_dim,           hidden=hidden, out_dim=out_dim)
        self.delta   = MLP(in_dim=x_dim + out_dim, hidden=(64,),  out_dim=out_dim)
        self.lr, self.epochs, self.wd, self.verbose = lr, epochs, wd, verbose
        self.scaler_x_low   = StandardScaler()
        self.scaler_x_delta = StandardScaler()

    def fit(self, Xl, yl, Xh, yh):
        device = self.device
        Xl_s = self.scaler_x_low.fit_transform(Xl)
        self.low, self.low_loss_history = train_torch_regressor(
            self.low, Xl_s, yl, lr=self.lr, epochs=self.epochs,
            weight_decay=self.wd, verbose=self.verbose, device=device)
        with torch.no_grad():
            mu_Lh = self.low(
                torch.tensor(self.scaler_x_low.transform(Xh), dtype=torch.float32, device=device)
            ).cpu().numpy()
        Xh_aug   = np.hstack([Xh, mu_Lh])
        Xh_aug_s = self.scaler_x_delta.fit_transform(Xh_aug)
        self.delta, self.delta_loss_history = train_torch_regressor(
            self.delta, Xh_aug_s, yh, lr=self.lr, epochs=self.epochs,
            weight_decay=self.wd, verbose=self.verbose, device=device)
        return self

    def predict(self, X):
        device = self.device
        with torch.no_grad():
            mu_L  = self.low(
                torch.tensor(self.scaler_x_low.transform(X), dtype=torch.float32, device=device)
            ).cpu().numpy()
            X_aug = np.hstack([X, mu_L])
            y_mf  = self.delta(
                torch.tensor(self.scaler_x_delta.transform(X_aug), dtype=torch.float32, device=device)
            ).cpu().numpy()
        return y_mf
