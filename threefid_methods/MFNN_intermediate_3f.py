#MFNN_intermediate_3f.py

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
from helpers_2f import get_device  # shared device utility

class IntermediateNet3f(nn.Module):
    """
    3-fidelity Intermediate network (configurable):

      trunk: x -> trunk_layers × (Linear+ReLU, width=hidden_width) -> h_shared
                               |
                               v
                            y_LF head
                               |
                               v
             [h_shared, y_LF] -> one MF hidden layer (hidden_width) -> y_MF
                               |
                               v
        [h_shared, y_LF, y_MF] -> one HF hidden layer (hidden_width) -> y_HF
    """
    def __init__(self, x_dim, hidden=(64,64,64)):
        super().__init__()

        #build shared trunk from tuple, like (64,64,64)
        trunk = []
        d = x_dim
        for w in hidden:
            trunk.append(nn.Linear(d, w))
            trunk.append(nn.ReLU())
            d = w
        self.trunk = nn.Sequential(*trunk)
        trunk_out_dim = d 

        #LF head from trunk output
        self.lf_head = nn.Linear(trunk_out_dim, 1)

        #MF hidden+head from [h_shared, y_LF]
        mf_in_dim = trunk_out_dim + 1
        self.mf_hidden = nn.Linear(mf_in_dim, trunk_out_dim)
        self.mf_activation = nn.ReLU()
        self.mf_head = nn.Linear(trunk_out_dim, 1)

        #HF hidden+head from [h_shared, y_LF, y_MF]
        hf_in_dim = trunk_out_dim + 2
        self.hf_hidden = nn.Linear(hf_in_dim, trunk_out_dim)
        self.hf_activation = nn.ReLU()
        self.hf_head = nn.Linear(trunk_out_dim, 1)

    def forward(self, x):
        #shared trunk
        h = self.trunk(x)                 #shape (N, hidden_width)

        #LF from shared representation
        y_lf = self.lf_head(h)            #shape (N, 1)

        #MF from [h, y_LF]
        mf_in = torch.cat([h, y_lf], dim=-1)   #shape (N, hidden_width+1)
        z_mf = self.mf_activation(self.mf_hidden(mf_in))
        y_mf = self.mf_head(z_mf)         #shape (N, 1)

        #HF from [h, y_LF, y_MF]
        hf_in = torch.cat([h, y_lf, y_mf], dim=-1)  #shape (N, hidden_width+2)
        z_hf = self.hf_activation(self.hf_hidden(hf_in))
        y_hf = self.hf_head(z_hf)         #shape (N, 1)

        return y_hf, y_mf, y_lf


class MFNN_Intermediate3f:
    """
    3-fidelity Intermediate MFNN wrapper:

      Data:
        Xl, yl : low-fidelity   (N_l, D), (N_l, 1)
        Xm, ym : mid-fidelity   (N_m, D), (N_m, 1)
        Xh, yh : high-fidelity  (N_h, D), (N_h, 1)

      Loss:
        L = w_hf*MSE_HF + w_mf*MSE_MF + w_lf*MSE_LF + lam * ||theta||^2
    """
    def __init__(
        self,
        x_dim,
        hidden=(64,64,64),
        lr=1e-3,
        epochs=2000,
        wd=0.0,
        w_hf=0.6,
        w_mf=0.3,
        w_lf=0.1,
        lam=1e-4,
        verbose=False,
    ):
        self.x_dim = x_dim
        self.hidden = hidden

        self.lr = lr
        self.epochs = epochs
        self.wd = wd
        self.w_hf = w_hf
        self.w_mf = w_mf
        self.w_lf = w_lf
        self.lam = lam
        self.verbose = verbose

        self.net = IntermediateNet3f(x_dim=x_dim, hidden=hidden)
        self.scaler_x = StandardScaler()
        self.device = get_device()  # single source of truth: GPU if available, else CPU
        self.net.to(self.device)    # move weights to device once at construction

    def fit(self, Xl, yl, Xm, ym, Xh, yh):
        """
        Fit on 3-fidelity data.
        """
        self.loss_history = []
        device = self.device  # use stored device, don't re-query

        #scale inputs using all fidelities
        X_all = np.vstack([Xl, Xm, Xh])
        self.scaler_x.fit(X_all)
        Xl_s = self.scaler_x.transform(Xl)
        Xm_s = self.scaler_x.transform(Xm)
        Xh_s = self.scaler_x.transform(Xh)

        Xl_t = torch.tensor(Xl_s, dtype=torch.float32, device=device)
        Xm_t = torch.tensor(Xm_s, dtype=torch.float32, device=device)
        Xh_t = torch.tensor(Xh_s, dtype=torch.float32, device=device)

        yl_t = torch.tensor(yl, dtype=torch.float32, device=device)
        ym_t = torch.tensor(ym, dtype=torch.float32, device=device)
        yh_t = torch.tensor(yh, dtype=torch.float32, device=device)

        mse = nn.MSELoss()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        for ep in trange(self.epochs, disable=not self.verbose):
            self.net.train()
            opt.zero_grad()

            #low-fidelity batch: use only y_LF
            y_hf_l, y_mf_l, y_lf_l = self.net(Xl_t)
            loss_lf = mse(y_lf_l, yl_t)

            #mid-fidelity batch: use only y_MF
            y_hf_m, y_mf_m, y_lf_m = self.net(Xm_t)
            loss_mf = mse(y_mf_m, ym_t)

            #high-fidelity batch: use only y_HF
            y_hf_h, y_mf_h, y_lf_h = self.net(Xh_t)
            loss_hf = mse(y_hf_h, yh_t)

            #manual L2 regularization on all parameters
            reg = self.lam * sum((p * p).sum() for p in self.net.parameters() if p.requires_grad)

            loss = self.w_hf * loss_hf + self.w_mf * loss_mf + self.w_lf * loss_lf + reg
            loss.backward()
            opt.step()
            self.loss_history.append(loss.item())


        self.net.eval()
        return self

    def predict(self, X):
        device = self.device  # use stored device, don't re-query
        self.net.eval()

        X_s = self.scaler_x.transform(X)
        X_t = torch.tensor(X_s, dtype=torch.float32, device=device)

        with torch.no_grad():
            y_hf, y_mf, y_lf = self.net(X_t)

        return y_hf.cpu().numpy()

    def predict_all(self, X):
        """Return all three fidelities (HF, MF, LF) for analysis."""
        device = self.device  # use stored device, don't re-query
        self.net.eval()

        X_s = self.scaler_x.transform(X)
        X_t = torch.tensor(X_s, dtype=torch.float32, device=device)

        with torch.no_grad():
            y_hf, y_mf, y_lf = self.net(X_t)

        return (
            y_hf.cpu().numpy(),
            y_mf.cpu().numpy(),
            y_lf.cpu().numpy(),
        )
