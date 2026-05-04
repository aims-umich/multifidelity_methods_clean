#Fidelity-flag NN: train on union of low+high with f∈{0,1}
from sklearn.preprocessing import StandardScaler
import torch
from helpers_2f import MLP, train_torch_regressor, get_device
import numpy as np

class MFNN_Flag:
    def __init__(self, x_dim, out_dim=1, hidden=(128,128,128,128), lr=0.001, epochs=500, wd=0.0, verbose=False):
        self.out_dim = out_dim
        self.device  = get_device()
        self.net     = MLP(in_dim=x_dim + 1, hidden=hidden, out_dim=out_dim)
        self.lr, self.epochs, self.wd, self.verbose = lr, epochs, wd, verbose
        self.scaler  = StandardScaler()

    def fit(self, Xl, yl, Xh, yh):
        X_l   = np.hstack([Xl, np.zeros((Xl.shape[0], 1))])
        X_h   = np.hstack([Xh, np.ones((Xh.shape[0], 1))])
        X_all = np.vstack([X_l, X_h])
        y_all = np.vstack([yl, yh])
        X_all_s = self.scaler.fit_transform(X_all)
        self.net, self.loss_history = train_torch_regressor(
            self.net, X_all_s, y_all, lr=self.lr, epochs=self.epochs,
            weight_decay=self.wd, verbose=self.verbose, device=self.device)
        return self

    def predict(self, X, fidelity_flag=1):
        device = self.device
        f  = np.full((X.shape[0], 1), fidelity_flag, dtype=float)
        Xf = np.hstack([X, f])
        with torch.no_grad():
            yhat = self.net(
                torch.tensor(self.scaler.transform(Xf), dtype=torch.float32, device=device)
            ).cpu().numpy()
        return yhat
