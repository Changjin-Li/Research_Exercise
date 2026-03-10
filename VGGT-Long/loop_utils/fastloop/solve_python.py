import torch
from torch import Tensor
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

def solve_sparse(A: csc_matrix, b: np.ndarray, freen: int) -> np.ndarray:
    """Solve linear system A * delta = b, supports submatrix solving."""
    if freen < 0:
        return spsolve(A, b)
    else:
        A_sub = A[:freen, :freen].tocsc()
        b_sub = b[:freen]
        delta_sub = spsolve(A_sub, b_sub)
        delta = np.zeros_like(b)
        delta[:freen] = delta_sub
        return delta

def solve_system_py(J_Ginv_i: Tensor, J_Ginv_j: Tensor, ii: Tensor, jj: Tensor, res: Tensor, ep: float, lm: float, freen: int) -> Tensor:
    device = res.device
    J_Ginv_i, J_Ginv_j = J_Ginv_i.cpu(), J_Ginv_j.cpu()
    ii, jj = ii.cpu(), jj.cpu()
    res = res.clone().cpu()

    r = res.size(0)
    n = max(ii.max().item(), jj.max().item()) + 1 # num of nodes

    res_vec = res.view(-1).numpy().astype(np.float64)

    rows, cols, data = [], [], []
    J_Ginv_i_np, J_Ginv_j_np = J_Ginv_i.numpy(), J_Ginv_j.numpy()
    ii_np, jj_np = ii.numpy(), jj.numpy()

    for x in range(r):
        i, j = ii_np[x], jj_np[x]
        if i == j:
            raise ValueError("Self-edges are not allowed.")

        for k in range(7):
            for l in range(7):
                row_index = x * 7 + k

                col_index_i = i * 7 + l
                val_i = J_Ginv_i_np[x, k, l]
                rows.append(row_index)
                cols.append(col_index_i)
                data.append(val_i)

                col_index_j = j * 7 + l
                val_j = J_Ginv_j_np[x, k, l]
                rows.append(row_index)
                cols.append(col_index_j)
                data.append(val_j)

    J = coo_matrix((data, (rows, cols)), shape=(r * 7, n * 7)).tocsc()

    b_vec = -J.T @ res_vec
    A_mat = J.T @ J

    diag = A_mat.diagonal()
    new_diag = diag * (1.0 + lm) + ep
    A_mat.setdiag(new_diag)

    freen_total = freen * 7
    delta = solve_sparse(A_mat.tocsc(), b_vec, freen_total)

    delta_tensor = torch.from_numpy(delta.astype(np.float32)).view(n, 7).to(device)
    return delta_tensor