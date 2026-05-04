#MFNN Intermediate 3f
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
from helpers_2f import get_device

class IntermediateNet3f(nn.Module):
    def __init__(self, x_dim, out_dim=1, hidden=(64,64,64)):
        super().__init__()
        self.out_dim = out_dim
        trunk = []
        d = x_dim
        for w in hidden:
            trunk.append(nn.Linear(d, w))
            trunk.append(nn.ReLU())
            d = w
        self.trunk     = nn.Sequential(*trunk)
        self.lf_head   = nn.Linear(d, out_dim)
        self.mf_hidden = nn.Linear(d + out_dim, d)
        self.mf_act    = nn.ReLU()
        self.mf_head   = nn.Linear(d, out_dim)
        self.hf_hidden = nn.Linear(d + out_dim * 2, d)
        self.hf_act    = nn.ReLU()
        self.hf_head   = nn.Linear(d, out_dim)

    def forward(self, x):
        h    = self.trunk(x)
        y_lf = self.lf_head(h)
        y_mf = self.mf_head(self.mf_act(self.mf_hidden(torch.cat([h, y_lf], dim=-1))))
        y_hf = self.hf_head(self.hf_act(self.hf_hidden(torch.cat([h, y_lf, y_mf], dim=-1))))
        return y_hf, y_mf, y_lf

class MFNN_Intermediate3f:
    def __init__(self, x_dim, out_dim=1, hidden=(64,64,64),
                 lr=1e-3, epochs=2000, wd=0.0,
                 w_hf=0.6, w_mf=0.3, w_lf=0.1, lam=1e-4, verbose=False):
        self.out_dim  = out_dim
        self.lr       = lr
        self.epochs   = epochs
        self.wd       = wd
        self.w_hf, self.w_mf, self.w_lf = w_hf, w_mf, w_lf
        self.lam      = lam
        self.verbose  = verbose
        self.net      = IntermediateNet3f(x_dim=x_dim, out_dim=out_dim, hidden=hidden)
        self.scaler_x = StandardScaler()
        self.device   = get_device()
        self.net.to(self.device)

    def fit(self, Xl, yl, Xm, ym, Xh, yh):
        self.loss_history = []
        device = self.device
        X_all  = np.vstack([Xl, Xm, Xh])
        self.scaler_x.fit(X_all)
        Xl_t = torch.tensor(self.scaler_x.transform(Xl), dtype=torch.float32, device=device)
        Xm_t = torch.tensor(self.scaler_x.transform(Xm), dtype=torch.float32, device=device)
        Xh_t = torch.tensor(self.scaler_x.transform(Xh), dtype=torch.float32, device=device)
        yl_t = torch.tensor(yl, dtype=torch.float32, device=device)
        ym_t = torch.tensor(ym, dtype=torch.float32, device=device)
        yh_t = torch.tensor(yh, dtype=torch.float32, device=device)
        mse  = nn.MSELoss()
        opt  = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        for ep in trange(self.epochs, disable=not self.verbose):
            self.net.train()
            opt.zero_grad()
            _, _, y_lf_l = self.net(Xl_t); loss_lf = mse(y_lf_l, yl_t)
            _, y_mf_m, _ = self.net(Xm_t); loss_mf = mse(y_mf_m, ym_t)
            y_hf_h, _, _ = self.net(Xh_t); loss_hf = mse(y_hf_h, yh_t)
            reg  = self.lam * sum((p*p).sum() for p in self.net.parameters() if p.requires_grad)
            loss = self.w_hf*loss_hf + self.w_mf*loss_mf + self.w_lf*loss_lf + reg
            loss.backward()
            opt.step()
            self.loss_history.append(loss.item())
        self.net.eval()
        return self

    def predict(self, X):
        device = self.device
        self.net.eval()
        X_t = torch.tensor(self.scaler_x.transform(X), dtype=torch.float32, device=device)
        with torch.no_grad():
            y_hf, _, _ = self.net(X_t)
        return y_hf.cpu().numpy()

    def predict_all(self, X):
        device = self.device
        self.net.eval()
        X_t = torch.tensor(self.scaler_x.transform(X), dtype=torch.float32, device=device)
        with torch.no_grad():
            y_hf, y_mf, y_lf = self.net(X_t)
        return y_hf.cpu().numpy(), y_mf.cpu().numpy(), y_lf.cpu().numpy()
