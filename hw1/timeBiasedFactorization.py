import numpy as np
from scipy import sparse


class timeBiasedFactorization:
  def __init__(self, n_factors=32, n_epochs=15, use_history=False,
               lr_user=0.08, lr_item=0.08, reg=0.02, t_m = None, 
               n_bns = None, time_array = None, random_state=42):
    self.n_factors, self.n_epochs = n_factors, n_epochs
    self.use_history = use_history
    self.lr_user, self.lr_item, self.reg = lr_user, lr_item, reg
    self.t_m = t_m
    self.n_bns = n_bns
    self.time_array = time_array
    self.random_state = random_state

  def fit(self, matrix):
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    n_users_fit, n_items_fit = matrix.shape
    self.n_users = n_users_fit
    self.n_items = n_items_fit 
    rng = np.random.default_rng(self.random_state)
    self.rng = rng
    self.global_mean = float(matrix.data.mean())
    self.user_factors = 0.05 * rng.standard_normal((n_users_fit, self.n_factors))
    #self.user_factors_t = 0.05 * rng.standard_normal((n_users_fit, self.n_factors, t_m + 1))
    self.user_factors_t = np.zeros((n_users_fit, self.n_factors, self.t_m + 1))
    self.item_factors = 0.05 * rng.standard_normal((n_items_fit, self.n_factors))
    self.history_factors = np.zeros((n_items_fit, self.n_factors))
    self.user_alpha = np.zeros(n_users_fit)
    self.user_alpha_k = np.zeros((n_users_fit, self.n_factors))
    self.user_bias = np.zeros(n_users_fit)
    self.user_bias_t = np.zeros((n_users_fit, self.t_m + 1))
    self.item_bias = np.zeros(n_items_fit)
    self.item_bias_t = np.zeros((n_users_fit, 30))
    counts = np.diff(matrix.indptr)
    history = matrix.copy()
    history.data = np.repeat(
        np.divide(1, np.sqrt(counts), out=np.zeros_like(counts, dtype=float), where=counts > 0),
        counts
    )
    self.history_matrix = history
    self.history_ = []

    for epoch in range(1, self.n_epochs + 1):
      for idx, u in enumerate(rng.permutation(n_users_fit)):
        start, end = matrix.indptr[u:u + 2]
        items = matrix.indices[start:end]
        if len(items) == 0:
          continue
        #print(("len items", len(items)))
        t = self.time_array[start:end]
        t_bin = np.floor(t / self.n_bns)
        dev_u = np.where(t - np.mean(t) > 0, 1, -1) * (np.abs(t - np.mean(t)) ** 0.4)
        ratings_u = matrix.data[start:end]
        old_q = self.item_factors[items].copy()
        history_term = (
            self.history_factors[items].sum(axis=0) / np.sqrt(len(items))
            if self.use_history else 0
        )
        effective_user = self.user_factors[u] + history_term
        t_arr = np.zeros(len(items))
        eff_t = np.repeat(effective_user[:, None], len(items), axis=1)
        t_arr += self.user_alpha[u] * dev_u
        t_arr += self.user_bias_t[u, self.time_array[start:end]]
        t_arr += np.array([self.item_bias_t[int(item_bin[0]), int(item_bin[1])] for item_bin in zip(items, t_bin)])
        eff_t += np.repeat(self.user_alpha_k[u][:, None], len(items), axis=1) * np.repeat(dev_u[None, :], self.n_factors, axis=0)
        eff_t += self.user_factors_t[u][:, self.time_array[start:end]]
        prediction = (self.global_mean + self.user_bias[u] + t_arr
                  + self.item_bias[items] + (old_q @ eff_t).diagonal())
        #prediction = (self.global_mean + self.user_bias[u]
        #              + self.item_bias[items] + old_q @ effective_user)
        error = ratings_u - prediction
        if np.abs((old_q @ eff_t).diagonal().mean()) > 100000:
          print((old_q, eff_t))
          print((idx, error.mean(), (old_q @ eff_t).diagonal().mean(), self.user_bias[u].mean(), self.item_bias[items].mean(),
                 t_arr.mean(), old_q.mean(), eff_t.mean()))
        mean_factor_gradient = error @ old_q / len(items)
        mean_factor_gradient_ = (error * dev_u) @ old_q / len(items)
        mean_factor_gradient_k = error[:, None] * old_q / len(items)

        self.user_bias[u] += self.lr_user * (error.mean() - self.reg * self.user_bias[u])
        self.user_alpha[u] += self.lr_user * ((error * dev_u).mean() - self.reg * self.user_alpha[u]) #.sum()
        self.user_alpha_k[u] += self.lr_user * (
            (mean_factor_gradient_) - self.reg * self.user_alpha_k[u]
        )
        self.user_bias_t[u, self.time_array[start:end]] += self.lr_user * ((error / len(items)) -
                                  self.reg * self.user_bias_t[u, self.time_array[start:end]]) #error / len(items)
        self.user_factors[u] += self.lr_user * (
            mean_factor_gradient - self.reg * self.user_factors[u]
        )
        self.user_factors_t[u][:, self.time_array[start:end]] += self.lr_user * (
            np.transpose(mean_factor_gradient_k) - self.reg * self.user_factors_t[u][:, self.time_array[start:end]]
        )
        self.item_bias[items] += self.lr_item * (
            error - self.reg * self.item_bias[items]
        )
        for idx, item_bin in enumerate(zip(items, t_bin)):
          item = int(item_bin[0])
          bin = int(item_bin[1])
          self.item_bias_t[item, bin] += self.lr_item * (
              error[idx] - self.reg * self.item_bias_t[item, bin]
          )
        self.item_factors[items] += self.lr_item * (
            error[:, None] * effective_user - self.reg * old_q
        )
        if self.use_history:
          self.history_factors[items] += self.lr_user * (
              mean_factor_gradient / np.sqrt(len(items))
              - self.reg * self.history_factors[items]
          )
      observed = matrix.tocoo()
      effective_users = self.user_factors.copy()
      if self.use_history:
        effective_users += self.history_matrix @ self.history_factors

      t_arr = np.zeros(len(observed.data))
      eff_t = effective_users[observed.row].copy()
      for u in rng.permutation(n_users_fit):
        start, end = matrix.indptr[u:u + 2]
        items = matrix.indices[start:end]
        t = self.time_array[start:end]
        t_bin = np.floor(t / self.n_bns)
        dev_u = np.where(t - np.mean(t) > 0, 1, -1) * (np.abs(t - np.mean(t)) ** 0.4)
        t_arr[start:end] += self.user_alpha[u] * dev_u
        t_arr[start:end] += self.user_bias_t[u, self.time_array[start:end]]
        t_arr[start:end] += np.array([self.item_bias_t[int(item_bin[0]), int(item_bin[1])] for item_bin in zip(items, t_bin)])
        eff_t[start:end] += np.transpose(np.repeat(self.user_alpha_k[u][:, None], len(items), axis=1) * np.repeat(dev_u[None, :], self.n_factors, axis=0))
        eff_t[start:end] += np.transpose(self.user_factors_t[u][:, self.time_array[start:end]])
      fitted = (self.global_mean + self.user_bias[observed.row] + t_arr
                + self.item_bias[observed.col]
                + np.sum(eff_t
                         * self.item_factors[observed.col], axis=1))
      rmse = float(np.sqrt(np.mean((observed.data - fitted) ** 2)))
      if not np.isfinite(rmse):
        raise FloatingPointError
      self.history_.append({"epoch": epoch, "train_rmse": rmse})
    return self

  def scores_for_user(self, user_idx):
    return self.score_matrix[user_idx]
    """
    effective_user = self.user_factors[user_idx].copy()
    if self.use_history:
      effective_user += (self.history_matrix[user_idx] @ self.history_factors).ravel()
    return (self.global_mean + self.user_bias[user_idx]
            + self.item_bias + self.item_factors @ effective_user)
    """
  def score_matrix(self):
    observed = self.history_matrix.tocoo()
    effective_users = self.user_factors.copy()
    if self.use_history:
      effective_users += self.history_matrix @ self.history_factors

    t_arr = np.zeros(len(observed.data))
    eff_t = effective_users[observed.row].copy()
    for u in self.rng.permutation(self.history_matrix.shape[0]):
      start, end = self.history_matrix.indptr[u:u + 2]
      items = self.history_matrix.indices[start:end]
      t = self.time_array[start:end]
      t_bin = np.floor(t / self.n_bns)
      dev_u = np.where(t - np.mean(t) > 0, 1, -1) * (np.abs(t - np.mean(t)) ** 0.4)
      t_arr[start:end] += self.user_alpha[u] * dev_u
      t_arr[start:end] += self.user_bias_t[u, self.time_array[start:end]]
      t_arr[start:end] += np.array([self.item_bias_t[int(item_bin[0]), int(item_bin[1])] for item_bin in zip(items, t_bin)])
      eff_t[start:end] += np.transpose(np.repeat(self.user_alpha_k[u][:, None], len(items), axis=1) * np.repeat(dev_u[None, :], self.n_factors, axis=0))
      eff_t[start:end] += np.transpose(self.user_factors_t[u][:, self.time_array[start:end]])
    pred = (self.global_mean + self.user_bias[observed.row] + t_arr
                + self.item_bias[observed.col]
                + np.sum(eff_t
                         * self.item_factors[observed.col], axis=1))
    return sparse.csr_matrix(
      (pred, (observed.row, observed.col)),
      shape=(self.n_users, self.n_items), dtype=np.float64
    ).toarray()
    """
    effective = self.user_factors.copy()
    if self.use_history:
      effective_user += (self.history_matrix @ self.history_factors).ravel()
    return (self.global_mean + self.user_bias
            + self.item_bias + effective_user @ np.transpose(self.item_factors))
    """
  def predict_pairs(self, user_idx, item_idx):
    return self.score_matrix[user_idx, item_idx]
    effective_users = self.user_factors.copy()
    if self.use_history:
      effective_users += self.history_matrix @ self.history_factors
    return (self.global_mean + self.user_bias[user_idx]
            + self.item_bias[item_idx]
            + np.sum(effective_users[user_idx] * self.item_factors[item_idx], axis=1))


