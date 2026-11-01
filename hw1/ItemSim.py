import numpy as np
from scipy.sparse import csr_matrix


class ItemSim:
  def __init__(self, random_state=42):
    self.random_state = random_state
  def fit(self, R_train, X_train, min_common=3, shrinkage=10):
    n_users = X_train.shape[0]
    n_items = X_train.shape[1]
    train_ratings = R_train
    values = train_ratings.data
    rows, cols = train_ratings.nonzero()
    user_count = np.bincount(rows, minlength=n_users)
    user_sum = np.bincount(rows, weights = values, minlength = n_users)
    global_mean = float(values.mean())
    self.user_mean = np.divide(
        user_sum,
        user_count,
        out=np.full(n_users, global_mean),
        where = user_count > 0, )
    residuals = values - self.user_mean[rows]
    centered = csr_matrix((residuals, (rows, cols)), shape=(n_users, n_items))
    self.centered = centered
    observed = X_train
    centered_csc = centered.tocsc()
    matrix = centered
    observation_matrix = observed
    gram = (matrix.T @ matrix).toarray()
    overlap = (observation_matrix.T @ observation_matrix).toarray()
    norms = np.sqrt(np.maximum(np.diag(gram), 0))
    denominator = np.outer(norms, norms)
    cosine = np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator>0)
    similarity = cosine * overlap / (overlap + shrinkage)
    np.fill_diagonal(similarity, 0)
    self.similarity = similarity
    self.overlap = overlap
    return self

  def score_matrix(self):
    weights = self.similarity
    weighted_r = np.dot(self.centered.toarray(), weights)
    sum_weights = np.abs(np.sum(weights, axis = 0))
    estimate = self.user_mean[:, None] + np.divide(weighted_r, sum_weights, out=np.zeros_like(weighted_r), where=sum_weights>0)
    return np.clip(estimate, 1, 10).astype(float)
