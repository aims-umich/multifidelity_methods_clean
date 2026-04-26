#Imports
from sklearn.preprocessing import StandardScaler  # Feature scaling/standardization.
import torch  # PyTorch main package (tensors, device management).
from helpers_2f import MLP, MLP_lin, train_torch_regressor, get_device
import numpy as np  # Numerical computing (arrays, random numbers, vectorized ops).


class MFNN_3step:
    """
    3-step multi-fidelity NN:
      1) low:  x -> y_LF
      2) lin:  [x, y_LF] -> y_lin  (linear correlations only)
      3) high: [x, y_LF, y_lin] -> y_HF
    """
    def __init__(
        self,
        x_dim,
        hidden_low=(128,128,128),   #for NN_LF
        hidden_lin=(128,),     #for NN_lin (linear, usually single layer)
        hidden_high=(128,),    #for NN_HF (single hidden layer per paper)
        lr=0.001,
        epochs=500,
        wd=0.0,
        verbose=False
    ):
        #1) x -> y_LF
        self.low = MLP(in_dim=x_dim, hidden=hidden_low, out_dim=1)
        #2) [x, y_LF] -> y_lin (linear network)
        self.lin = MLP_lin(in_dim=x_dim+1, hidden=hidden_lin, out_dim=1)
        #3) [x, y_LF, y_lin] -> y_HF (nonlinear, shallow)
        self.high = MLP(in_dim=x_dim+2, hidden=hidden_high, out_dim=1)

        self.lr = lr
        self.epochs = epochs
        self.wd = wd
        self.verbose = verbose
        self.device = get_device()  # single source of truth: GPU if available, else CPU

        #scalers for each input space
        self.scaler_x_low = StandardScaler()  #for x into low
        self.scaler_x_lin = StandardScaler()  #for [x, y_LF] into lin
        self.scaler_x_mf = StandardScaler()   #for [x, y_LF, y_lin] into high

    def fit(self, Xl, yl, Xh, yh):
        """
        Xl, yl: low-fidelity data
        Xh, yh: high-fidelity data
        """
        device = self.device
        #----- step 1: train low on (Xl, yl) -----
        Xl_s = self.scaler_x_low.fit_transform(Xl)
        self.low, self.low_loss_history = train_torch_regressor(
            self.low, Xl_s, yl,
            lr=self.lr, epochs=self.epochs, weight_decay=self.wd,
            verbose=self.verbose, device=device)

        #----- step 2: train lin on ([Xh, y_LF(Xh)], yh) -----
        with torch.no_grad():
            Xh_low_s = self.scaler_x_low.transform(Xh)
            mu_Lh = self.low(
                torch.tensor(Xh_low_s, dtype=torch.float32, device=device)
            ).cpu().numpy()

        Xh_aug = np.hstack([Xh, mu_Lh])
        Xh_aug_s = self.scaler_x_lin.fit_transform(Xh_aug)
        self.lin, self.lin_loss_history = train_torch_regressor(
            self.lin, Xh_aug_s, yh,
            lr=self.lr, epochs=self.epochs, weight_decay=self.wd,
            verbose=self.verbose, device=device)

        #----- step 3: train high on ([Xh, y_LF(Xh), y_lin(Xh)], yh) -----
        with torch.no_grad():
            mu_lin_h = self.lin(
                torch.tensor(Xh_aug_s, dtype=torch.float32, device=device)
            ).cpu().numpy()

        Xh_mf = np.hstack([Xh, mu_Lh, mu_lin_h])
        Xh_mf_s = self.scaler_x_mf.fit_transform(Xh_mf)
        self.high, self.high_loss_history = train_torch_regressor(
            self.high, Xh_mf_s, yh,
            lr=self.lr, epochs=self.epochs, weight_decay=self.wd,
            verbose=self.verbose, device=device)

        return self

    def predict(self, X):
        """
        Full 3-step prediction:
          X -> y_LF(X) -> y_lin(X, y_LF) -> y_HF(X, y_LF, y_lin)
        Returns y_HF.
        """
        device = self.device
        with torch.no_grad():
            #step1: y_LF(X)
            X_low_s = self.scaler_x_low.transform(X)
            mu_L = self.low(
                torch.tensor(X_low_s, dtype=torch.float32, device=device)
            ).cpu().numpy()

            #step2: y_lin(X, y_LF(X))
            X_aug = np.hstack([X, mu_L])
            X_aug_s = self.scaler_x_lin.transform(X_aug)
            mu_lin = self.lin(
                torch.tensor(X_aug_s, dtype=torch.float32, device=device)
            ).cpu().numpy()

            #step3: y_HF(X, y_LF(X), y_lin(X))
            X_mf = np.hstack([X, mu_L, mu_lin])
            X_mf_s = self.scaler_x_mf.transform(X_mf)
            y_mf = self.high(
                torch.tensor(X_mf_s, dtype=torch.float32, device=device)
            ).cpu().numpy()

        return y_mf