#Fidelity-flag NN: train on union of low+high with f∈{0,1}
#Add a binary flag feature indicating fidelity level; single network learns both.

#Imports
from sklearn.preprocessing import StandardScaler  # Feature scaling/standardization.
import torch  # PyTorch main package (tensors, device management).
from helpers_2f import MLP, train_torch_regressor, get_device
import numpy as np  # Numerical computing (arrays, random numbers, vectorized ops).

class MFNN_Flag:
    """Single NN that takes (x, f) with f∈{0,1} and learns to map to y; infer with f=1."""
    def __init__(self, x_dim, hidden=(128,128,128,128), lr=0.001, epochs=500, wd=0.0, verbose=False):
        self.device = get_device()  # single source of truth: GPU if available, else CPU
        self.net = MLP(in_dim=x_dim+1, hidden=hidden, out_dim=1)
        self.lr, self.epochs, self.wd, self.verbose = lr, epochs, wd, verbose
        self.scaler = StandardScaler()

    def fit(self, Xl, yl, Xh, yh):
        """Concatenate low+high data with fidelity flag and train one network."""
        X_l = np.hstack([Xl, np.zeros((Xl.shape[0], 1))])  # flag f=0 for low-fid
        X_h = np.hstack([Xh, np.ones((Xh.shape[0], 1))])   # flag f=1 for high-fid
        X_all = np.vstack([X_l, X_h])
        y_all = np.vstack([yl, yh])

        X_all_s = self.scaler.fit_transform(X_all)
        self.net, self.loss_history = train_torch_regressor(
            self.net, X_all_s, y_all, lr=self.lr, epochs=self.epochs,
            weight_decay=self.wd, verbose=self.verbose, device=self.device)
        return self

    def predict(self, X, fidelity_flag=1):
        """Infer outputs for inputs X at desired fidelity (default high: f=1)."""
        device = self.device
        f = np.full((X.shape[0], 1), fidelity_flag, dtype=float)
        Xf = np.hstack([X, f])
        with torch.no_grad():
            yhat = self.net(
                torch.tensor(self.scaler.transform(Xf), dtype=torch.float32, device=device)
            ).cpu().numpy()
        return yhat