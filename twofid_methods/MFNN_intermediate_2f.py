#MFNN Intermediate 2f
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import torch.nn as nn
from tqdm import trange
from helpers_2f import get_device

class IntermediateNet(nn.Module):
    def __init__(self, x_dim, out_dim=1, hidden=(64,64,64), hf_hidden=(64,)):
        super().__init__()
        trunk = []
        d = x_dim
        for w in hidden:
            trunk.append(nn.Linear(d, w))
            trunk.append(nn.ReLU())
            d = w
        self.trunk    = nn.Sequential(*trunk)
        self.lf_head  = nn.Linear(d, out_dim)
        hf_in_dim     = d + out_dim
        layers = []
        d2 = hf_in_dim
        for h in hf_hidden:
            layers.append(nn.Linear(d2, h))
            layers.append(nn.ReLU())
            d2 = h
        self.hf_trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.hf_head  = nn.Linear(d2, out_dim)

    def forward(self, x):
        h     = self.trunk(x)
        y_lf  = self.lf_head(h)
        z     = self.hf_trunk(torch.cat([h, y_lf], dim=-1))
        y_hf  = self.hf_head(z)
        return y_hf, y_lf

class MFNN_Intermediate:
    def __init__(self, x_dim, out_dim=1, hidden=(64,64,64), hf_hidden=(64,),
                 lr=1e-3, epochs=500, wd=0.0, alpha=3e-2, lam=1e-4, verbose=False):
        self.out_dim  = out_dim
        self.lr       = lr
        self.epochs   = epochs
        self.wd       = wd
        self.alpha    = alpha
        self.lam      = lam
        self.verbose  = verbose
        self.net      = IntermediateNet(x_dim, out_dim=out_dim, hidden=hidden, hf_hidden=hf_hidden)
        self.scaler_x = StandardScaler()
        self.device   = get_device()
        self.net.to(self.device)

    def fit(self, Xl, yl, Xh, yh):
        device = self.device
        X_all  = np.vstack([Xl, Xh])
        self.scaler_x.fit(X_all)
        Xl_t = torch.tensor(self.scaler_x.transform(Xl), dtype=torch.float32, device=device)
        Xh_t = torch.tensor(self.scaler_x.transform(Xh), dtype=torch.float32, device=device)
        yl_t = torch.tensor(yl, dtype=torch.float32, device=device)
        yh_t = torch.tensor(yh, dtype=torch.float32, device=device)
        mse  = nn.MSELoss()
        opt  = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        self.loss_history = []
        for ep in trange(self.epochs, disable=not self.verbose):
            self.net.train()
            opt.zero_grad()
            y_hf_l, y_lf_l = self.net(Xl_t)
            loss_lf = mse(y_lf_l, yl_t)
            y_hf_h, y_lf_h = self.net(Xh_t)
            loss_hf = mse(y_hf_h, yh_t)
            reg  = self.lam * sum((p*p).sum() for p in self.net.parameters() if p.requires_grad)
            loss = self.alpha * loss_hf + (1.0 - self.alpha) * loss_lf + reg
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
            y_hf, _ = self.net(X_t)
        return y_hf.cpu().numpy()
