import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class Salad(nn.Module):
    def __init__(
        self,
        num_channels: int = 1536,
        num_clusters: int = 64,
        cluster_dim: int = 128,
        token_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim),
        )
        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            self.dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1),
        )
        # MLP for score matrix S
        self.score_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            self.dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    def forward(self, x) -> Tensor:
        """
        Args:
            x (tuple): A tuple containing two elements, f and t.
        Returns:
            f (Tensor): The global descriptor [B, m*l + g]
        """
        f, t = x
        p = self.score_features(f).flatten(2)
        f = self.cluster_features(f).flatten(2)
        t = self.token_features(t)

        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)
        p = p[:, :-1, :]

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        f = torch.cat([
            F.normalize(t, p=2, dim=-1),
            F.normalize(f * p.sum(dim=-1), p=2, dim=1).flatten(1)
        ], dim=-1)
        f = F.normalize(f, p=2, dim=-1)
        return f


def log_otp_solver(log_a: Tensor, log_b: Tensor, M: Tensor, num_iters: int = 20, reg: float = 1.0) -> Tensor:
    """
    Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a: Source weights
        log_b: Target weights
        M: metric cost matrix
        num_iters: The number of iterations.
        reg: regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)


def get_matching_probs(S, dustbin_score = 1.0, num_iters = 3, reg = 1.0):
    """
    Sinkhorn algorithm.
    """
    batch_size, m, n = S.size()

    # augment scores matrix
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n-m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(log_a, log_b, S_aug, num_iters, reg)
    return log_P - norm