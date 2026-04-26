#Imports
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from helpers_2f import MLP, train_torch_regressor, get_device

class MFNN_Flag3f:
    """
    3-fidelity flag network
    
    Single MLP takes (x, f) where
      0 -> low-fidelity
      1 -> medium-fidelity
      2 -> high-fidelity
    """
    def __init__(self, x_dim, hidden=(128,128,128,128), lr=0.001, epochs=500, wd=0.0, verbose=False):
        self.device = get_device()  # single source of truth: GPU if available, else CPU
        self.net = MLP(in_dim=x_dim + 1, hidden=hidden, out_dim=1)
        self.lr = lr
        self.epochs = epochs
        self.wd = wd
        self.verbose = verbose
        self.scaler = StandardScaler()

    def fit(self, Xl, yl, Xm, ym, Xh, yh):
        """
        Train on low, medium, and high fidelity data

        Xl, Xm, Xh: shape (N_fid, D)
        yl, ym, yh: shape (N_fid, 1) or (N_fid,)
        """

        #ensure column vectors for targets
        yl = np.asarray(yl).reshape(-1, 1)
        ym = np.asarray(ym).reshape(-1, 1)
        yh = np.asarray(yh).reshape(-1, 1)

        #build flagged inputs
        f_l = np.zeros((Xl.shape[0], 1), dtype=float)          #f=0 for low
        f_m = np.ones((Xm.shape[0], 1), dtype=float)           #f=1 for medium
        f_h = np.full((Xh.shape[0], 1), 2.0, dtype=float)      #f=2 for high

        X_l = np.hstack([Xl, f_l])
        X_m = np.hstack([Xm, f_m])
        X_h = np.hstack([Xh, f_h])

        #stack all fidelities
        X_all = np.vstack([X_l, X_m, X_h])
        y_all = np.vstack([yl, ym, yh])

        #scale and train
        X_all_s = self.scaler.fit_transform(X_all)
        self.net, self.loss_history = train_torch_regressor(
            self.net,
            X_all_s,
            y_all,
            lr=self.lr,
            epochs=self.epochs,
            weight_decay=self.wd,
            verbose=self.verbose,
            device=self.device,
        )
        return self

    def predict(self, X, fidelity_flag=2):
        """
        Predict at a desired fidelity

        fidelity_flag:
            0 -> low-fidelity prediction
            1 -> medium-fidelity prediction
            2 -> high-fidelity prediction (default)
        """
        device = self.device
        f = np.full((X.shape[0], 1), float(fidelity_flag))
        Xf = np.hstack([X, f])
        Xf_s = self.scaler.transform(Xf)

        with torch.no_grad():
            X_tensor = torch.tensor(Xf_s, dtype=torch.float32, device=device)
            yhat = self.net(X_tensor).cpu().numpy()
        return yhat
