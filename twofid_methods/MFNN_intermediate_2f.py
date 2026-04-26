#Imports
from sklearn.preprocessing import StandardScaler  # Feature scaling/standardization.
import torch  # PyTorch main package (tensors, device management).
import numpy as np  # Numerical computing (arrays, random numbers, vectorized ops).
import torch.nn as nn  # Neural network layers and modules.
from helpers_2f import get_device

class IntermediateNet(nn.Module):
    """
    Core Intermediate network:

      x -> 3×(Linear+Tanh, 64) -> h3
                   |
                   v
                y_LF head
                   |
                   v
       [h3, y_LF] -> HF trunk -> y_HF
    """
    def __init__(self, x_dim, hidden=(64,64,64), hf_hidden=(64,)):
        super().__init__()
        #shared trunk: 3 hidden layers of size 64 with relu
        trunk = []
        d = x_dim
        for w in hidden:
            trunk.append(nn.Linear(d, w))
            trunk.append(nn.ReLU())
            d = w
        self.trunk = nn.Sequential(*trunk)
        trunk_out_dim = d

        #LF head from h3
        self.lf_head = nn.Linear(trunk_out_dim, 1)

        #HF trunk from [h3, y_LF]
        hf_in_dim = trunk_out_dim + 1
        layers = []
        d = hf_in_dim
        for h in hf_hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            d = h
        self.hf_trunk = nn.Sequential(*layers) if layers else nn.Identity()

        #HF head
        self.hf_head = nn.Linear(d, 1)

    def forward(self, x):
        h3 = self.trunk(x)

        y_lf = self.lf_head(h3)              # branch: LF output from h3

        hf_in = torch.cat([h3, y_lf], dim=-1)  # continue to HF from [h3, y_LF]
        z = self.hf_trunk(hf_in)
        y_hf = self.hf_head(z)

        return y_hf, y_lf

class MFNN_Intermediate:
    """
    Wrapper around IntermediateNet with custom multi-fidelity loss:

      L = alpha*MSE_HF + (1-alpha)*MSE_LF + lam * ||W||^2
    """
    def __init__(
        self,
        x_dim,
        hidden=(64,64,64),
        hf_hidden=(64,),
        lr=1e-3,
        epochs=500,
        wd=0.0,
        alpha=3e-2,
        lam=1e-4,
        verbose=False
    ):
        self.x_dim = x_dim
        self.lr = lr
        self.epochs = epochs
        self.wd = wd
        self.alpha = alpha
        self.lam = lam
        self.verbose = verbose

        self.net = IntermediateNet(x_dim, hidden=hidden, hf_hidden=hf_hidden)
        self.scaler_x = StandardScaler()
        self.device = get_device()  # single source of truth: GPU if available, else CPU
        self.net.to(self.device)   # move weights to device once at construction

    def fit(self, Xl, yl, Xh, yh):
        device = self.device  # use stored device, don't re-query

        #scale inputs using both fidelities
        X_all = np.vstack([Xl, Xh])
        self.scaler_x.fit(X_all)
        Xl_s = self.scaler_x.transform(Xl)
        Xh_s = self.scaler_x.transform(Xh)

        Xl_t = torch.tensor(Xl_s, dtype=torch.float32, device=device)
        Xh_t = torch.tensor(Xh_s, dtype=torch.float32, device=device)
        yl_t = torch.tensor(yl, dtype=torch.float32, device=device)
        yh_t = torch.tensor(yh, dtype=torch.float32, device=device)

        mse = nn.MSELoss()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        self.loss_history = []
        
        for ep in trange(self.epochs, disable=not self.verbose):
            self.net.train()
            opt.zero_grad()

            #low-fidelity batch
            y_hf_l, y_lf_l = self.net(Xl_t)
            loss_lf = mse(y_lf_l, yl_t)

            #high-fidelity batch
            y_hf_h, y_lf_h = self.net(Xh_t)
            loss_hf = mse(y_hf_h, yh_t)

            #L2 regularization
            reg = self.lam * sum((p * p).sum() for p in self.net.parameters() if p.requires_grad)

            loss = self.alpha * loss_hf + (1.0 - self.alpha) * loss_lf + reg
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
            y_hf, y_lf = self.net(X_t)

        return y_hf.cpu().numpy()   # high-fidelity prediction only
