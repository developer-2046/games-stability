import torch

class LinearGame:
    """
    F(x) = A x  with A = S + K,  S=S^T (symmetric), K=-K^T (skew).
    Control skew ratio by alpha in [0,1]:  A = (1-alpha)*S0 + alpha*K0
    """
    def __init__(self, d=2, sym_mu=0.2, skew_omega=1.0, alpha=0.8, device="cpu"):
        self.d = d
        self.device = device
        # Base symmetric (positive definite-ish)
        S0 = sym_mu * torch.eye(d, device=device)
        # Base skew (simple 2x2 rotation block + zeros)
        K0 = torch.zeros(d, d, device=device)
        if d >= 2:
            K0[:2,:2] = torch.tensor([[0.0, -skew_omega],
                                      [skew_omega, 0.0]], device=device)
        self.A = (1-alpha)*S0 + alpha*K0

    def F(self, x):
        return x @ self.A.T  # (n,d) @ (d,d)^T -> (n,d)

def simplex_proj(z):
    # Euclidean projection onto probability simplex (per row)
    # Returns shape like z
    z_sorted, _ = torch.sort(z, dim=1, descending=True)
    cumsum = torch.cumsum(z_sorted, dim=1)
    k = torch.arange(1, z.shape[1]+1, device=z.device).float().unsqueeze(0)
    t = (cumsum - 1) / k
    # Find rho: largest j with z_sorted_j > t_j
    mask = (z_sorted > t).float()
    rho = torch.argmax((mask * torch.arange(1, z.shape[1]+1, device=z.device)).float(), dim=1)
    theta = t[torch.arange(z.shape[0]), rho]
    return torch.clamp(z - theta.unsqueeze(1), min=0.0)
