#GPmimic 2 fidelity method
from sklearn.preprocessing import StandardScaler
import torch
from helpers_2f import MLP, get_device
import numpy as np
import torch.nn as nn
from tqdm import trange

class GPmimic(nn.Module):
    def __init__(self, x_dim, out_dim=1, hidden=(128,128,128,128),
                 lr=0.001, epochs=500, alpha=3e-2, lam=1e-4, verbose=False):
        super().__init__()
        self.out_dim = out_dim
        self.lr      = lr
        self.epochs  = epochs
        self.alpha   = alpha
        self.lam     = lam
        self.verbose = verbose
        self.device  = get_device()
        latent = out_dim * 2
        self.shared   = MLP(in_dim=x_dim, hidden=hidden, out_dim=latent)
        self.mixing   = nn.Linear(latent, latent, bias=True)
        self.scaler_x = StandardScaler()
        self.to(self.device)

    def forward(self, x):
        u    = self.shared(x)
        y    = self.mixing(u)
        y_hf = y[:, :self.out_dim]
        y_lf = y[:, self.out_dim:]
        return y_hf, y_lf

    def fit(self, Xl, yl, Xh, yh):
        device = self.device
        X_all = np.vstack([Xl, Xh])
        self.scaler_x.fit_transform(X_all)
        Xl_s = torch.tensor(self.scaler_x.transform(Xl), dtype=torch.float32, device=device)
        yl_t = torch.tensor(yl, dtype=torch.float32, device=device)
        Xh_s = torch.tensor(self.scaler_x.transform(Xh), dtype=torch.float32, device=device)
        yh_t = torch.tensor(yh, dtype=torch.float32, device=device)
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.lr)
        mse = nn.MSELoss()
        self.loss_history = []
        for ep in trange(self.epochs, disable=not self.verbose):
            self.train()
            optimizer.zero_grad()
            pred_hf_l, pred_lf_l = self.forward(Xl_s)
            loss_lf = mse(pred_lf_l, yl_t)
            pred_hf_h, pred_lf_h = self.forward(Xh_s)
            loss_hf = mse(pred_hf_h, yh_t)
            W   = self.mixing.weight
            reg = self.lam * torch.sum(W * W)
            loss = self.alpha * loss_hf + (1 - self.alpha) * loss_lf + reg
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())
        return self

    def predict(self, X):
        self.eval()
        device = next(self.parameters()).device
        X_s = torch.tensor(self.scaler_x.transform(X), dtype=torch.float32, device=device)
        with torch.no_grad():
            y_hf, _ = self.forward(X_s)
        return y_hf.cpu().numpy()
