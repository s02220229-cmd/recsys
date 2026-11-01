import numpy as np
import pandas as pd
from time import perf_counter
from tqdm.auto import tqdm
from scipy import sparse
from contextlib import nullcontext

K = 10
try:
  from threadpoolctl import threadpool_limits
except ImportError:
  def threadpool_limits(*args, **kwargs):
    return nullcontext()

def interaction_matrix(frame, positive_only=True):
  n_users = np.max(frame.user_idx.to_numpy(dtype=int)) + 1
  n_items = np.max(frame.item_idx.to_numpy(dtype=int)) + 1
  matrix = sparse.csr_matrix(
      (np.ones(len(frame)), (frame.user_idx.to_numpy(dtype=int), frame.item_idx.to_numpy(dtype=int))),
      shape=(n_users, n_items), dtype=np.float64
  )
  matrix.sum_duplicates()
  matrix.data[:] = 1.0
  return matrix
def rating_matrix(frame, positive_only=True):
  n_users = np.max(frame.user_idx.to_numpy(dtype=int)) + 1
  n_items = np.max(frame.item_idx.to_numpy(dtype=int)) + 1
  matrix = sparse.csr_matrix(
      (frame["rating"], (frame.user_idx.to_numpy(dtype=int), frame.item_idx.to_numpy(dtype=int))),
      shape=(n_users, n_items), dtype=np.float64
  )
  return matrix
def user_metrics(top, relevant, k):
  hit_mask = np.isin(top, list(relevant)).astype(float)
  discounts = 1 / np.log(np.arange(2, k + 2))
  hits = float(hit_mask.sum())
  ideal = discounts[:min(k, len(relevant))].sum()
  return {f"Recall@{k}": hits / len(relevant), f"Precision@{k}": hits/k,
          f"NDCG@{k}": float(hit_mask @ discounts[:len(top)] / ideal),
          f"HR@{k}": float(hits > 0)}
def top_items(scores, seen, candidate_mask, k = K):
  eligible = candidate_mask.copy()
  eligible[seen] = False
  candidates = np.flatnonzero(eligible)
  order = np.lexsort((candidates, -scores[candidates]))
  return candidates[order[:k]]
def evaluate(model, history, heldout, k=K, progress=True):
  positives = interaction_matrix(history)
  n_users = positives.shape[0]
  n_items = positives.shape[1]
  all_seen = interaction_matrix(history, positive_only = False)
  item_counts = np.asarray(positives.sum(axis=0)).ravel()
  candidates = item_counts > 0
  warm_users = np.diff(positives.indptr) > 0
  relevant = heldout.drop_duplicates(["user_id", "item_id"])
  mapped = relevant.dropna(subset=["user_idx", "item_idx"]).copy()
  mapped[["user_idx", "item_idx"]] = mapped[["user_idx", "item_idx"]].astype(np.int64)
  warm = mapped[candidates[mapped.item_idx.to_numpy()]
                & warm_users[mapped.user_idx.to_numpy()]].copy()
  seen_mask = all_seen.toarray().astype(bool)
  repeated = seen_mask[warm.user_idx.to_numpy(), warm.item_idx.to_numpy()]
  eligible = warm[~repeated]
  with threadpool_limits(limits=1):
    scores = np.asarray(model.score_matrix())
  assert scores.shape == (n_users, n_items)
  assert scores.shape == (n_users, n_items) and np.isfinite(scores).all()
  rows, recommended = [], set()
  groups = eligible.groupby("user_idx", sort=True)
  for u, group in tqdm(groups, total=groups.ngroups,
                       desc=f"Evaluate {type(model).__name__}", unit="customer",
                       leave=False, mininterval=0.5, disable=not progress):
      seen = all_seen.indices[all_seen.indptr[u]:all_seen.indptr[u + 1]]
      truth = set(group.item_idx)
      assert truth.isdisjoint(seen)
      top = top_items(scores[u], seen, candidates, k)
      recommended.update(top.tolist())
      rows.append({"user_idx": int(u), "history_length": len(seen),
                   "relevant_items": len(truth), **user_metrics(top, truth, k)})
  details = pd.DataFrame(rows)
  metrics = {name: float(details[name].mean()) for name in
              [f"Recall@{k}", f"Precision@{k}", f"NDCG@{k}", f"HR@{k}"]}
  metrics.update(eval_users=len(details), heldout_positives=len(relevant),
                  unmapped_pairs=len(relevant) - len(mapped), repeated_pairs=int(repeated.sum()),
                  eligible_positives=len(eligible), candidate_items=int(candidates.sum()),
                  coverage=len(recommended) / candidates.sum())
  return metrics, details
def evaluate_test(model, history, heldout, k=K, progress=True):
  positives = interaction_matrix(history)
  n_users = positives.shape[0]
  n_items = positives.shape[1]
  all_seen = interaction_matrix(history, positive_only = False)
  item_counts = np.asarray(positives.sum(axis=0)).ravel()
  candidates = item_counts > 0
  warm_users = np.diff(positives.indptr) > 0
  relevant = heldout.drop_duplicates(["user_id", "item_id"])
  mapped = relevant.dropna(subset=["user_idx", "item_idx"]).copy()
  mapped[["user_idx", "item_idx"]] = mapped[["user_idx", "item_idx"]].astype(np.int64)
  warm = mapped[candidates[mapped.item_idx.to_numpy()]
                & warm_users[mapped.user_idx.to_numpy()]].copy()
  seen_mask = all_seen.toarray().astype(bool)
  repeated = seen_mask[warm.user_idx.to_numpy(), warm.item_idx.to_numpy()]
  eligible = warm[~repeated]
  if callable(getattr(model, "score_matrix", None)):
    with threadpool_limits(limits=1):
      scores = np.asarray(model.score_matrix())
  elif callable(getattr(model, "predict_item", None)):
    scores_v = interaction_matrix(history).copy()
    scores_v.data[:] = 0
    for user_idx in mapped.user_idx.to_numpy(dtype=int):
      for item_idx in mapped.item_idx.to_numpy(dtype=int):
        scores_v[user_idx, item_idx] = model.predict_item(user_idx, item_idx)
    scores = scores_v.toarray()
  else:
    raise ValueError
  assert scores.shape == (n_users, n_items) and np.isfinite(scores).all()
  rows, recommended = [], set()
  groups = eligible.groupby("user_idx", sort=True)
  for u, group in tqdm(groups, total=groups.ngroups,
                       desc=f"Evaluate {type(model).__name__}", unit="customer",
                       leave=False, mininterval=0.5, disable=not progress):
      seen = all_seen.indices[all_seen.indptr[u]:all_seen.indptr[u + 1]]
      truth = set(group.item_idx)
      assert truth.isdisjoint(seen)
      top = top_items(scores[u], seen, candidates, k)
      recommended.update(top.tolist())
      rows.append({"user_idx": int(u), "history_length": len(seen),
                   "relevant_items": len(truth), **user_metrics(top, truth, k)})
  details = pd.DataFrame(rows)
  metrics = {name: float(details[name].mean()) for name in
              [f"Recall@{k}", f"Precision@{k}", f"NDCG@{k}", f"HR@{k}"]}
  metrics.update(eval_users=len(details), heldout_positives=len(relevant),
                  unmapped_pairs=len(relevant) - len(mapped), repeated_pairs=int(repeated.sum()),
                  eligible_positives=len(eligible), candidate_items=int(candidates.sum()),
                  coverage=len(recommended) / candidates.sum())
  return metrics, details
def fit_with_one_blas_thread(model, matrix, validation=None):
  with threadpool_limits(limits=1, user_api="blas"):
    if validation is None:
      return model.fit(matrix)
    return model.fit(matrix, validation=validation)
def sample_negatives(rng, user_indices, seen_mask, candidate_ids):
  if np.any(seen_mask[user_indices][:, candidate_ids].all(axis=1)):
    raise ValueError
  negative = rng.choice(candidate_ids, size=len(user_indices))
  collisions = seen_mask[user_indices, negative]
  while np.any(collisions):
    negative[collisions] = rng.choice(candidate_ids, size=int(collisions.sum()))
    collisions = seen_mask[user_indices, negative]
  return negative
def normalize_rows(matrix):
  matrix = sparse.csr_matrix(matrix, dtype=np.float64)
  norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
  return sparse.diags(np.divide(1, norms, out=np.zeros_like(norms), where=norms > 0)) @ matrix
class Popularity:
  def fit(self, matrix):
    self.counts = np.asarray(matrix.sum(axis=0)).ravel()
    self.n_users = matrix.shape[0]
    return self
  def score_matrix(self):
    return np.broadcast_to(self.counts, (self.n_users, len(self.counts)))

def timed_fit(model, *args, **kwargs):
  started = perf_counter()
  with threadpool_limits(limits=1):
    model.fit(*args, **kwargs)
  model.fit_seconds_ = perf_counter() - started
  return model

class FeatureMap:
  def __init__(self, use_side_features, item_side, n_users, n_items):
    self.label = "IDs + metadata" if use_side_features else "IDs"
    self.user_features = sparse.eye(n_users, format="csr")
    self.item_features = sparse.eye(n_items, format="csr")
    if use_side_features:
      self.user_features = sparse.hstack([self.user_features], format="csr")
      self.item_features = sparse.hstack([self.item_features, item_side], format="csr")
    self.user_dim = self.user_features.shape[1]
    self.n_features = self.user_dim + self.item_features.shape[1]

  def pairs(self, user_indices, item_indices):
    return sparse.hstack([self.user_features[user_indices], self.item_features[item_indices]], format="csr")


