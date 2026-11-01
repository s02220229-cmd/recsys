import numpy as np
from scipy import sparse, linalg
from tqdm.auto import tqdm

class EASE:
  def __init__(self, reg=100.0, progress=True):
    self.reg = reg
    self.progress = progress
  def fit(self, matrix):
    if self.reg <= 0:
      raise ValueError
    self.history = sparse.csr_matrix(matrix, dtype=np.float64).copy()
    with tqdm(total=3, desc="EASE stages", unit="stage", leave=False,
              mininterval=0.5, disable=not self.progress) as progress:
        progress.set_postfix_str("Gram matrix")
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::gram.shape[0] + 1] += self.reg
        progress.update(1)
        progress.set_postfix_str("Cholesky solve")
        factor = linalg.cho_factor(gram, lower=True, check_finite=False)
        precision = linalg.cho_solve(factor, np.eye(gram.shape[0]), check_finite = False)
        progress.update(1)
        progress.set_postfix_str("Weight normalization")
        self.W = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.W, 0)
        assert np.isfinite(self.W).all()
        progress.update(1)
    return self
  def score_matrix(self):
    return self.history @ self.W
