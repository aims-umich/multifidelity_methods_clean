#GPmimic 2 fidelity method
#Imports
from sklearn.preprocessing import StandardScaler  # Feature scaling/standardization.
import torch  # PyTorch main package (tensors, device management).
from helpers_2f import MLP, get_device
import numpy as np  # Numerical computing (arrays, random numbers, vectorized ops).
import torch.nn as nn  # Neural network layers and modules.
from tqdm import trange  # Progress-bar iterator for training loops.

class GPmimic(nn.Module):
    """GPmimic model: shared MLP trunk -> 2-dim latent u(x) -> 2x2 linear mixing -> [y_HF, y_LF]
    Training loss: L = alpha*MSE_HF + (1-alpha)*MSE_LF + lam * ||W||^2"""

    def __init__(self,x_dim,hidden=(128,128,128,128),lr=0.001,epochs=500,alpha=3e-2,lam=1e-4,verbose=False):
        super().__init__()
        self.x_dim = x_dim
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.lam = lam
        self.verbose = verbose
        self.device = get_device()  # single source of truth: GPU if available, else CPU

        #Shared neural network trunk attaching to both high fidelity and low fidelity
        self.shared = MLP(in_dim=x_dim, hidden=hidden, out_dim=2)

        #Add mixing layer before the outputs
        self.mixing = nn.Linear(2, 2, bias=True)

        #Input scaler for high fidelity and low fidelity
        self.scaler_x = StandardScaler()

        # Move all submodules to the chosen device now, once
        self.to(self.device)

    def forward(self, x):
        u = self.shared(x)
        y = self.mixing(u)
        y_hf = y[:, 0:1]
        y_lf = y[:, 1:]
        return y_hf, y_lf

    def fit(self, Xl, yl, Xh, yh):
        device = self.device
        # --- Scale inputs ---
        X_all = np.vstack([Xl, Xh])
        X_all_s = self.scaler_x.fit_transform(X_all)

        # Convert all arrays to tensors
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

            # Forward low-fidelity batch
            pred_hf_l, pred_lf_l = self.forward(Xl_s)
            # Low-fidelity loss is based only on LF prediction
            loss_lf = mse(pred_lf_l, yl_t)

            # Forward high-fidelity batch
            pred_hf_h, pred_lf_h = self.forward(Xh_s)
            # High-fidelity loss uses HF prediction
            loss_hf = mse(pred_hf_h, yh_t)

            # Regularization on mixing matrix (2x2)
            W = self.mixing.weight
            reg = self.lam * torch.sum(W * W)

            # Total loss
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
            y_hf, y_lf = self.forward(X_s)
        return y_hf.cpu().numpy()