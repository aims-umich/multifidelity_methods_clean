#3 fidelity flag NN
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from helpers_2f import MLP, train_torch_regressor, get_device

class MFNN_Flag3f:
    def __init__(self, x_dim, out_dim=1, hidden=(128,128,128,128),
                 lr=0.001, epochs=500, wd=0.0, verbose=False):
        self.out_dim = out_dim
        self.device  = get_device()
        self.net     = MLP(in_dim=x_dim + 1, hidden=hidden, out_dim=out_dim)
        self.lr, self.epochs, self.wd, self.verbose = lr, epochs, wd, verbose
        self.scaler  = StandardScaler()

    def fit(self, Xl, yl, Xm, ym, Xh, yh):
        yl = np.asarray(yl).reshape(-1, self.out_dim)
        ym = np.asarray(ym).reshape(-1, self.out_dim)
        yh = np.asarray(yh).reshape(-1, self.out_dim)
        f_l = np.zeros((Xl.shape[0], 1), dtype=float)
        f_m = np.ones((Xm.shape[0], 1), dtype=float)
        f_h = np.full((Xh.shape[0], 1), 2.0, dtype=float)
        X_all = np.vstack([np.hstack([Xl, f_l]),
                           np.hstack([Xm, f_m]),
                           np.hstack([Xh, f_h])])
        y_all = np.vstack([yl, ym, yh])
        X_all_s = self.scaler.fit_transform(X_all)
        self.net, self.loss_history = train_torch_regressor(
            self.net, X_all_s, y_all, lr=self.lr, epochs=self.epochs,
            weight_decay=self.wd, verbose=self.verbose, device=self.device)
        return self

    def predict(self, X, fidelity_flag=2):
        device = self.device
        f  = np.full((X.shape[0], 1), float(fidelity_flag))
        Xf = np.hstack([X, f])
        with torch.no_grad():
            yhat = self.net(
                torch.tensor(self.scaler.transform(Xf), dtype=torch.float32, device=device)
            ).cpu().numpy()
        return yhat
