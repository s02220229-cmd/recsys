import numpy as np
from scipy.special import expit
from rec_utils import interaction_matrix, sample_negatives, evaluate
from tqdm.auto import tqdm
FM_SAMPLES_PER_EPOCH = None
K = 10

class FactorizationMachine:
  def __init__(self, n_factors=16, epochs=30, lr=0.01, reg=1e-5,
               batch_size=512, random_state=42, progress=True):
    self.n_factors, self.epochs, self.lr = n_factors, epochs, lr
    self.reg, self.batch_size, self.random_state = reg, batch_size, random_state
    self.progress = progress

  def initialize(self, features):
    self.features = features
    rng = np.random.default_rng(self.random_state)
    self.w = np.zeros(features.n_features)
    self.V = 0.05 * rng.standard_normal((features.n_features, self.n_factors))
    self.m_w, self.s_w = np.zeros_like(self.w), np.zeros_like(self.w)
    self.m_v, self.s_v = np.zeros_like(self.V), np.zeros_like(self.V)
    self.step = 0

  def forward(self, X):
    squared = X.multiply(X)
    projected = X @ self.V
    scores = np.asarray(X @ self.w).ravel() + 0.5 * np.sum(
        projected ** 2 - squared @ (self.V ** 2), axis=1
    )
    return scores, projected, squared

  def gradients(self, X, projected, squared, errors):
    grad_w = np.asarray(X.T @ errors).ravel()
    grad_v = (X.T @ (errors[:,None] * projected)
              - np.asarray(squared.T @ errors).ravel()[:,None] * self.V)
    return grad_w, np.asarray(grad_v)

  def bpr_gradients(self, X_pos, X_neg):
    pos, projected_pos, squared_pos = self.forward(X_pos)
    neg, projected_neg, squared_neg = self.forward(X_neg)
    delta = pos - neg
    weight = expit(-delta)
    gw_pos, gv_pos = self.gradients(X_pos, projected_pos, squared_pos, -weight)
    gw_neg, gv_neg = self.gradients(X_neg, projected_neg, squared_neg, weight)
    size = X_pos.shape[0]
    loss = float(np.logaddexp(0, -delta).mean())
    loss += 0.5 * self.reg * (np.sum(self.w ** 2) + np.sum(self.V ** 2))
    return loss, (gw_pos + gw_neg) / size + self.reg * self.w, \
        (gv_pos + gv_neg) / size + self.reg * self.V

  def update(self, grad_w, grad_v):
    self.step += 1
    for parameter, first, second, gradient in [
        (self.w, self.m_w, self.s_w, grad_w),
        (self.V, self.m_v, self.s_v, grad_v),
    ]:
        first *= 0.9
        first += 0.1 * gradient
        second *= 0.999
        second += 0.001 * gradient ** 2
        parameter -= self.lr * (first / (1 - 0.9 ** self.step)) / (
            np.sqrt(second / (1 - 0.999 ** self.step)) + 1e-8
        )

  def score_matrix(self):
    boundary = self.features.user_dim
    U, I = self.features.user_features, self.features.item_features
    Vu, Vi = self.V[:boundary], self.V[boundary:]
    projected_u, projected_i = U @ Vu, I @ Vi
    offset_u = U @ self.w[:boundary] + 0.5 * np.sum(
        projected_u ** 2 - U.multiply(U) @ (Vu ** 2), axis=1
    )
    offset_i = I @ self.w[boundary:] + 0.5 * np.sum(
        projected_i ** 2 - I.multiply(I) @ (Vi ** 2), axis=1
    )
    return offset_u[:, None] + offset_i[None, :] + projected_u @ projected_i.T

  def fit(self, history, features, validation=None, checkpoints=None):
    self.initialize(features)
    positives = history.drop_duplicates(["user_idx", "item_idx"])
    positive_matrix = interaction_matrix(history)
    seen = interaction_matrix(history, positive_only=False).toarray().astype(bool)
    candidates = np.flatnonzero(np.asarray(positive_matrix.sum(axis=0)).ravel() > 0)
    saturated = seen[:, candidates].all(axis=1)
    positives = positives[~saturated[positives.user_idx.to_numpy()]]
    positive_users = positives.user_idx.to_numpy()
    positive_items = positives.item_idx.to_numpy()
    rng = np.random.default_rng(self.random_state)
    self.history_, self.best_epoch_ = [], self.epochs
    best_metric, best_parameters = -np.inf, None
    checkpoints = checkpoints or [self.epochs]
    label = f"FM-BPR {features.label}"
    with tqdm(total=self.epochs, desc = label + " epochs", unit="epoch",
              mininterval=0.5, disable=not self.progress) as epoch_progress, \
         tqdm(total=0, desc="Minibatches", unit="batch", leave=False,
              mininterval=0.5, disable=not self.progress) as batch_progress:
        for epoch in range(1, self.epochs + 1):
          if FM_SAMPLES_PER_EPOCH is None:
            order = rng.permutation(len(positives))
          else:
            sample_count = min(len(positives), FM_SAMPLES_PER_EPOCH)
            order = rng.integers(0, len(positives), size=sample_count)
          total_loss = 0.0
          batch_progress.reset(total=(len(order) + self.batch_size - 1) // self.batch_size)
          batch_progress.set_description(f"{label}, epoch {epoch}")
          for start in range(0, len(order), self.batch_size):
            batch = order[start:start + self.batch_size]
            u, i = positive_users[batch], positive_items[batch]
            j = sample_negatives(rng, u, seen, candidates)
            assert not seen[u, j].any()
            X_pos, X_neg = features.pairs(u, i), features.pairs(u, j)
            loss, grad_w, grad_v = self.bpr_gradients(X_pos, X_neg)
            self.update(grad_w, grad_v)
            total_loss += loss * len(batch)
            batch_progress.update(1)
          record = {"epoch": epoch, "training_bpr": total_loss / len(order)}
          status = {"loss": f"{record['training_bpr']:.4f}"}
          if validation is not None and epoch in checkpoints:
            metrics, _ = evaluate(self, history, validation, progress=False)
            record[f"validation_NDCG@{K}"] = metrics[f"NDCG@{K}"]
            status["val_NNDCG"] = f"{metrics[f'NDCG@{K}']:.4f}"
            if metrics[f"NDCG@{K}"] > best_metric:
              best_metric = metrics[f"NDCG@{K}"]
              self.best_epoch_ = epoch
              best_parameters = (self.w.copy(), self.V.copy())
          self.history_.append(record)
          epoch_progress.set_postfix(status)
          epoch_progress.update(1)
    if best_parameters is not None:
      self.w, self.V = best_parameters
    assert np.isfinite(self.V).all() and np.isfinite(self.w).all()
    return self
