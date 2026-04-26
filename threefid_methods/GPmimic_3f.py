#3 fidelity GPmimic

#Imports
from sklearn.preprocessing import StandardScaler  # Feature scaling/standardization.
import torch  # PyTorch main package (tensors, device management).
import numpy as np  # Numerical computing (arrays, random numbers, vectorized ops).
import torch.nn as nn  # Neural network layers and modules.
from tqdm import trange  # Progress-bar iterator for training loops.
from helpers_2f import MLP, get_device  # Shared MLP architecture and device utility.

class GPmimic3f(nn.Module):
    """3-fidelity GPmimic-style model:
       shared MLP trunk -> 3-dim latent u(x) -> 3x3 linear mixing -> [y_HF, y_MF, y_LF]
       Training loss:
         L = w_h*MSE_HF + w_m*MSE_MF + w_l*MSE_LF + lam*||W||^2
    """

    def __init__(
        self,
        x_dim,
        hidden=(128,128,128,128),
        lr=0.001,
        epochs=500,
        w_h=0.6,
        w_m=0.3,
        w_l=0.1,
        lam=1e-4,
        verbose=False,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.lr = lr
        self.epochs = epochs
        self.w_h = w_h
        self.w_m = w_m
        self.w_l = w_l
        self.lam = lam
        self.verbose = verbose
        self.device = get_device()  # single source of truth: GPU if available, else CPU

        #shared trunk
        self.shared = MLP(in_dim=x_dim, hidden=hidden, out_dim=3)

        #mixing layer
        self.mixing = nn.Linear(3, 3, bias=True)

        #input scaler
        self.scaler_x = StandardScaler()

        #store training loss per epoch
        self.loss_history = []

        # move all submodules to device once at construction
        self.to(self.device)

    def forward(self, x):
        u = self.shared(x)
        y = self.mixing(u)
        y_hf = y[:, 0:1]
        y_mf = y[:, 1:2]
        y_lf = y[:, 2:3]
        return y_hf, y_mf, y_lf

    def fit(self, Xl, yl, Xm, ym, Xh, yh):
        device = self.device

        #reset loss history
        self.loss_history = []

        #scale inputs using all fidelities together
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

            #low-fidelity batch
            pred_h_l, pred_m_l, pred_l_l = self.forward(Xl_s)
            loss_l = mse(pred_l_l, yl_t)

            #medium-fidelity batch
            pred_h_m, pred_m_m, pred_l_m = self.forward(Xm_s)
            loss_m = mse(pred_m_m, ym_t)

            #high-fidelity batch
            pred_h_h, pred_m_h, pred_l_h = self.forward(Xh_s)
            loss_h = mse(pred_h_h, yh_t)

            #regularization on mixing matrix (3x3)
            W = self.mixing.weight
            reg = self.lam * torch.sum(W * W)

            loss = self.w_h * loss_h + self.w_m * loss_m + self.w_l * loss_l + reg

            loss.backward()
            optimizer.step()

            self.loss_history.append(float(loss.item()))

        return self

    def predict(self, X):
        self.eval()
        device = self.device
        X_s = torch.tensor(self.scaler_x.transform(X), dtype=torch.float32, device=device)
        with torch.no_grad():
            y_hf, y_mf, y_lf = self.forward(X_s)
        return y_hf.cpu().numpy()

    def predict_all(self, X):
        self.eval()
        device = self.device
        X_s = torch.tensor(self.scaler_x.transform(X), dtype=torch.float32, device=device)
        with torch.no_grad():
            y_hf, y_mf, y_lf = self.forward(X_s)
        return (
            y_hf.cpu().numpy(),
            y_mf.cpu().numpy(),
            y_lf.cpu().numpy(),
        )

