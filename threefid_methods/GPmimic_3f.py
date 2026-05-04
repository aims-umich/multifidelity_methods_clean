#3 fidelity GPmimic
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import torch.nn as nn
from tqdm import trange
from helpers_2f import MLP, get_device

class GPmimic3f(nn.Module):
    def __init__(self, x_dim, out_dim=1, hidden=(128,128,128,128),
                 lr=0.001, epochs=500, w_h=0.6, w_m=0.3, w_l=0.1,
                 lam=1e-4, verbose=False):
        super().__init__()
        self.out_dim  = out_dim
        self.lr       = lr
        self.epochs   = epochs
        self.w_h, self.w_m, self.w_l = w_h, w_m, w_l
        self.lam      = lam
        self.verbose  = verbose
        self.device   = get_device()
        latent = out_dim * 3
        self.shared   = MLP(in_dim=x_dim, hidden=hidden, out_dim=latent)
        self.mixing   = nn.Linear(latent, latent, bias=True)
        self.scaler_x = StandardScaler()
        self.loss_history = []
        self.to(self.device)

    def forward(self, x):
        u    = self.shared(x)
        y    = self.mixing(u)
        y_hf = y[:, :self.out_dim]
        y_mf = y[:, self.out_dim:self.out_dim*2]
        y_lf = y[:, self.out_dim*2:]
        return y_hf, y_mf, y_lf

    def fit(self, Xl, yl, Xm, ym, Xh, yh):
        device = self.device
        self.loss_history = []
        X_all = np.vstack([Xl, Xm, Xh])
        self.scaler_x.fit(X_all)
        Xl_s = torch.tensor(self.scaler_x.transform(Xl), dtype=torch.float32, device=device)
        Xm_s = torch.tensor(self.scaler_x.transform(Xm), dtype=torch.float32, device=device)
        Xh_s = torch.tensor(self.scaler_x.transform(Xh), dtype=torch.float32, device=device)
        yl_t = torch.tensor(yl, dtype=torch.float32, device=device)
        ym_t = torch.tensor(ym, dtype=torch.float32, device=device)
        yh_t = torch.tensor(yh, dtype=torch.float32, device=device)
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.lr)
        mse = nn.MSELoss()
        for ep in trange(self.epochs, disable=not self.verbose):
            self.train()
            optimizer.zero_grad()
            _, _, pred_l = self.forward(Xl_s)
            _, pred_m, _ = self.forward(Xm_s)
            pred_h, _, _ = self.forward(Xh_s)
            W   = self.mixing.weight
            reg = self.lam * torch.sum(W * W)
            loss = (self.w_h * mse(pred_h, yh_t) +
                    self.w_m * mse(pred_m, ym_t) +
                    self.w_l * mse(pred_l, yl_t) + reg)
            loss.backward()
            optimizer.step()
            self.loss_history.append(float(loss.item()))
        return self

    def predict(self, X):
        self.eval()
        device = self.device
        X_s = torch.tensor(self.scaler_x.transform(X), dtype=torch.float32, device=device)
        with torch.no_grad():
            y_hf, _, _ = self.forward(X_s)
        return y_hf.cpu().numpy()

    def predict_all(self, X):
        self.eval()
        device = self.device
        X_s = torch.tensor(self.scaler_x.transform(X), dtype=torch.float32, device=device)
        with torch.no_grad():
            y_hf, y_mf, y_lf = self.forward(X_s)
        return y_hf.cpu().numpy(), y_mf.cpu().numpy(), y_lf.cpu().numpy()
