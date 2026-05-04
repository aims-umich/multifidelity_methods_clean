#3 step MFNN
from sklearn.preprocessing import StandardScaler
import torch
from helpers_2f import MLP, MLP_lin, train_torch_regressor, get_device
import numpy as np

class MFNN_3step:
    def __init__(self, x_dim, out_dim=1,
                 hidden_low=(128,128,128),
                 hidden_lin=(128,),
                 hidden_high=(128,),
                 lr=0.001, epochs=500, wd=0.0, verbose=False):
        self.out_dim = out_dim
        self.low     = MLP(    in_dim=x_dim,               hidden=hidden_low,  out_dim=out_dim)
        self.lin     = MLP_lin(in_dim=x_dim + out_dim,     hidden=hidden_lin,  out_dim=out_dim)
        self.high    = MLP(    in_dim=x_dim + out_dim * 2, hidden=hidden_high, out_dim=out_dim)
        self.lr, self.epochs, self.wd, self.verbose = lr, epochs, wd, verbose
        self.device       = get_device()
        self.scaler_x_low = StandardScaler()
        self.scaler_x_lin = StandardScaler()
        self.scaler_x_mf  = StandardScaler()

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
        Xh_aug_s = self.scaler_x_lin.fit_transform(Xh_aug)
        self.lin, self.lin_loss_history = train_torch_regressor(
            self.lin, Xh_aug_s, yh, lr=self.lr, epochs=self.epochs,
            weight_decay=self.wd, verbose=self.verbose, device=device)
        with torch.no_grad():
            mu_lin_h = self.lin(
                torch.tensor(Xh_aug_s, dtype=torch.float32, device=device)
            ).cpu().numpy()
        Xh_mf   = np.hstack([Xh, mu_Lh, mu_lin_h])
        Xh_mf_s = self.scaler_x_mf.fit_transform(Xh_mf)
        self.high, self.high_loss_history = train_torch_regressor(
            self.high, Xh_mf_s, yh, lr=self.lr, epochs=self.epochs,
            weight_decay=self.wd, verbose=self.verbose, device=device)
        return self

    def predict(self, X):
        device = self.device
        with torch.no_grad():
            mu_L    = self.low(
                torch.tensor(self.scaler_x_low.transform(X), dtype=torch.float32, device=device)
            ).cpu().numpy()
            X_aug   = np.hstack([X, mu_L])
            mu_lin  = self.lin(
                torch.tensor(self.scaler_x_lin.transform(X_aug), dtype=torch.float32, device=device)
            ).cpu().numpy()
            X_mf    = np.hstack([X, mu_L, mu_lin])
            y_mf    = self.high(
                torch.tensor(self.scaler_x_mf.transform(X_mf), dtype=torch.float32, device=device)
            ).cpu().numpy()
        return y_mf
