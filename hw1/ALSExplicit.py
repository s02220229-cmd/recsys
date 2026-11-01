import numpy as np
from scipy import sparse

class ALSExplicit:
  def __init__(self, n_factors=32, n_iters=5, reg=10.0, random_state=42):
    if n_factors < 1 or n_iters < 1 or reg <= 0:
      raise ValueError
    self.n_factors, self.n_iters = n_factors, n_iters
    self.reg, self.random_state = reg, random_state

  def fit(self, matrix, validation=None):
    matrix = sparse.csr_matrix(matrix,  dtype=np.float64)
    n_users_fit, n_items_fit = matrix.shape
    rng = np.random.default_rng(self.random_state)
    self.user_factors = 0.1 * rng.standard_normal((n_users_fit, self.n_factors))
    self.item_factors = 0.1 * rng.standard_normal((n_items_fit, self.n_factors))
    by_item = matrix.tocsc()
    active_users = np.flatnonzero(np.diff(matrix.indptr))
    active_items = np.flatnonzero(np.diff(by_item.indptr))
    self.user_factors[np.diff(matrix.indptr) == 0] = 0
    self.item_factors[np.diff(by_item.indptr) == 0] = 0
    eye = np.eye(self.n_factors)
    observed = matrix.tocoo()
    self.history_ = []

    for iteration in range(1, self.n_iters + 1):
      for u in active_users:
        start, end = matrix.indptr[u:u+2]
        item_idx = matrix.indices[start:end]
        factors = self.item_factors[item_idx]
        self.user_factors[u] = np.linalg.solve(
            factors.T @ factors + self.reg * eye,
            factors.T @ matrix.data[start:end]
        )
      for i in active_items:
        start, end = by_item.indptr[i:i+2]
        user_idx = by_item.indices[start:end]
        factors = self.user_factors[user_idx]
        self.item_factors[i] = np.linalg.solve(
            factors.T @ factors + self.reg * eye,
            factors.T @ by_item.data[start:end]
        )
      train_pred = self.predict_pairs(observed.row, observed.col)
      row = {"iteration": iteration,
             "train_rmse": float(np.sqrt(np.mean((observed.data - train_pred) ** 2)))}
      if validation is not None:
        val_pred = self.predict_pairs(
            validation.user_idx.to_numpy(), validation.item_idx.to_numpy()
        )
        row["valid_rmse"] = float(np.sqrt(np.mean(
            (validation.rating.to_numpy() - val_pred) ** 2
        )))
      self.history_.append(row)
    return self
  def predict_pairs(self, user_idx, item_idx):
    return np.sum(self.user_factors[user_idx] * self.item_factors[item_idx], axis=1)
  def scores_for_user(self, user_idx):
    return self.item_factors @ self.user_factors[user_idx]
  def score_matrix(self):
    return self.user_factors @ np.transpose(self.item_factors)

