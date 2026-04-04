# src/symmetry/modeling/suff_stats.py

import numpy as np
import torch
from scipy import stats


class SuffStatsGPU:
    """
    Accumulates sufficient statistics using PyTorch tensors on the GPU.
    Supports both reduced model (1 predictor) and full model (2 predictors).
    """
    def __init__(self, C, device):
        self.C = C
        self.device = device
        self.n = 0

        # Scalar sums kept as float64 tensors on GPU
        self.Sx1   = torch.tensor(0.0, dtype=torch.float64, device=device)
        self.Sx2   = torch.tensor(0.0, dtype=torch.float64, device=device)
        self.Sx1x1 = torch.tensor(0.0, dtype=torch.float64, device=device)
        self.Sx2x2 = torch.tensor(0.0, dtype=torch.float64, device=device)
        self.Sx1x2 = torch.tensor(0.0, dtype=torch.float64, device=device)

        # Vector sums (C,) on GPU
        self.Sy   = torch.zeros(C, dtype=torch.float64, device=device)
        self.Sx1y = torch.zeros(C, dtype=torch.float64, device=device)
        self.Sx2y = torch.zeros(C, dtype=torch.float64, device=device)
        self.Syy  = torch.zeros(C, dtype=torch.float64, device=device)

    def update1(self, x1, y):
        """Reduced model update. x1: (N,), y: (N, C) tensors on GPU."""
        self.n    += x1.shape[0]
        self.Sx1  += torch.sum(x1)
        self.Sx1x1 += torch.dot(x1, x1)
        self.Sy   += torch.sum(y, dim=0)
        self.Sx1y += torch.matmul(x1, y)
        self.Syy  += torch.sum(y ** 2, dim=0)

    def update2(self, x1, x2, y):
        """Full model update. x1, x2: (N,), y: (N, C) tensors on GPU."""
        self.n    += x1.shape[0]
        self.Sx1  += torch.sum(x1)
        self.Sx2  += torch.sum(x2)
        self.Sx1x1 += torch.dot(x1, x1)
        self.Sx2x2 += torch.dot(x2, x2)
        self.Sx1x2 += torch.dot(x1, x2)
        self.Sy   += torch.sum(y, dim=0)
        self.Sx1y += torch.matmul(x1, y)
        self.Sx2y += torch.matmul(x2, y)
        self.Syy  += torch.sum(y ** 2, dim=0)

    def compute_reduced(self):
        """Move data to CPU and solve OLS for reduced model (1 predictor).
        Returns beta0, beta1, t, p, R2 — all (C,) numpy arrays.
        """
        n   = self.n
        Sx  = self.Sx1.item()
        Sxx = self.Sx1x1.item()
        Sy  = self.Sy.cpu().numpy()
        Sxy = self.Sx1y.cpu().numpy()
        Syy = self.Syy.cpu().numpy()

        denom = n * Sxx - Sx ** 2
        beta1 = np.where(denom != 0, (n * Sxy - Sx * Sy) / denom, 0.0)
        beta0 = (Sy - beta1 * Sx) / n

        SS_res = Syy - beta1 * Sxy - beta0 * Sy
        SS_tot = Syy - Sy ** 2 / n
        R2     = np.where(SS_tot > 0, np.maximum(0.0, 1.0 - SS_res / SS_tot), 0.0)

        df   = n - 2
        MSE  = np.where(df > 0, np.maximum(0.0, SS_res) / df, 0.0)
        Sxx_centered = Sxx - Sx ** 2 / n
        se_beta1 = np.where(Sxx_centered > 0, np.sqrt(MSE / Sxx_centered), np.inf)
        t = np.where(se_beta1 > 0, beta1 / se_beta1, 0.0)
        p = 2 * (1 - stats.t.cdf(np.abs(t), df=df))

        return beta0, beta1, t, p, R2, SS_res

    def compute_full(self, R2_reduced, SS_res_reduced):
        """Move data to CPU and solve OLS for full model (2 predictors).
        Returns beta0, beta1, beta2, t1, p1, t2, p2, R2_full, delta_R2 — all (C,).
        """
        n = self.n

        Sx1  = self.Sx1.item();  Sx2  = self.Sx2.item()
        Sx1x1 = self.Sx1x1.item(); Sx2x2 = self.Sx2x2.item()
        Sx1x2 = self.Sx1x2.item()

        Sy   = self.Sy.cpu().numpy()
        Sx1y = self.Sx1y.cpu().numpy()
        Sx2y = self.Sx2y.cpu().numpy()
        Syy  = self.Syy.cpu().numpy()

        A = np.array([
            [n,    Sx1,   Sx2  ],
            [Sx1,  Sx1x1, Sx1x2],
            [Sx2,  Sx1x2, Sx2x2],
        ], dtype=np.float64)

        b = np.stack([Sy, Sx1y, Sx2y], axis=0)  # (3, C)

        try:
            coeffs = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(A, b, rcond=None)[0]

        beta0, beta1, beta2 = coeffs[0], coeffs[1], coeffs[2]

        SS_tot = Syy - Sy ** 2 / n
        SS_res = Syy - beta0 * Sy - beta1 * Sx1y - beta2 * Sx2y

        R2_full  = np.where(SS_tot > 0, np.maximum(0.0, 1.0 - SS_res / SS_tot), 0.0)
        delta_R2 = np.maximum(0.0, R2_full - R2_reduced)

        df  = n - 3
        MSE = np.where(df > 0, np.maximum(0.0, SS_res) / df, 0.0)

        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)

        var_beta1 = A_inv[1, 1] * MSE
        var_beta2 = A_inv[2, 2] * MSE

        se1 = np.where(var_beta1 > 0, np.sqrt(var_beta1), np.inf)
        se2 = np.where(var_beta2 > 0, np.sqrt(var_beta2), np.inf)
        t1  = np.where(se1 > 0, beta1 / se1, 0.0)
        t2  = np.where(se2 > 0, beta2 / se2, 0.0)
        p1  = 2 * (1 - stats.t.cdf(np.abs(t1), df=np.maximum(df, 1)))
        p2  = 2 * (1 - stats.t.cdf(np.abs(t2), df=np.maximum(df, 1)))
        F_stat  = np.where(MSE > 0, (SS_res_reduced - SS_res) / MSE, 0.0)
        p_val_F = 1 - stats.f.cdf(F_stat, dfn=1, dfd=np.maximum(df, 1))

        return beta0, beta1, beta2, t1, p1, t2, p2, R2_full, delta_R2, F_stat, p_val_F
