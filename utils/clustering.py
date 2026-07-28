# -*- coding: utf-8 -*-
"""
Clustering Utilities — GPU-accelerated G-means
Uses PyTorch for all compute-intensive operations (KMeans split, Anderson-Darling test).
Falls back to CPU when CUDA is unavailable.
"""

import numpy as np
import torch
from typing import Optional, Tuple


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GMeansClusterer:
    """GPU-accelerated G-means clustering.

    The public API (fit / predict / labels_ / cluster_centers_ / n_clusters)
    is compatible with the original NumPy / sklearn version so that callers
    need only minimal changes.
    """

    def __init__(
        self,
        k_max: int = 30,
        random_state: int = 42,
        similarity_threshold: float = 0.7,
        alpha: float = 0.05,
        device: Optional[torch.device] = None,
    ):
        self.k_max = k_max
        self.random_state = random_state
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha
        self.device = device or _auto_device()

        self.labels_: Optional[np.ndarray] = None
        self.n_clusters: Optional[int] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.feature_type: Optional[str] = None

    # ------------------------------------------------------------------
    # GPU helpers
    # ------------------------------------------------------------------

    def _kmeans_split(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Binary (2-means) clustering on *self.device*."""
        n, d = X.shape
        if n < 2:
            return torch.zeros(n, dtype=torch.long, device=self.device), X

        rng = torch.Generator(device=self.device)
        rng.manual_seed(self.random_state)

        best_labels = None
        best_centers = None
        best_inertia = float("inf")

        for attempt in range(3):
            seed = self.random_state + attempt
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)

            # initialise 2 centres using k-means++ style
            idx0 = torch.randint(0, n, (1,), generator=g, device=self.device)
            centres = X[idx0].clone()

            # pick second centre
            dists = torch.cdist(X, centres, p=2).min(dim=1).values ** 2
            probs = dists / (dists.sum() + 1e-12)
            idx1 = torch.multinomial(probs, 1)
            centres = torch.cat([centres, X[idx1]], dim=0)

            for _ in range(100):
                # assign
                dists = torch.cdist(X, centres, p=2)
                labels = dists.argmin(dim=1)

                # update
                new_centres = torch.zeros_like(centres)
                for c in range(2):
                    mask = labels == c
                    if mask.sum() == 0:
                        new_centres[c] = centres[c]
                    else:
                        new_centres[c] = X[mask].mean(dim=0)

                if torch.allclose(new_centres, centres, atol=1e-6):
                    break
                centres = new_centres

            inertia = sum(
                ((X[labels == c] - centres[c]) ** 2).sum()
                for c in range(2)
                if (labels == c).sum() > 0
            )
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centres

        return best_labels, best_centers

    def _project_to_line(
        self, points: torch.Tensor, center1: torch.Tensor, center2: torch.Tensor
    ) -> torch.Tensor:
        direction = center2 - center1
        direction_norm = direction / (direction.norm() + 1e-8)
        centered = points - center1
        return centered @ direction_norm

    def _anderson_test_torch(self, projections: torch.Tensor) -> bool:
        """Anderson-Darling normality test implemented in pure PyTorch.

        Returns True when the sample is *consistent with* a normal
        distribution (i.e. do NOT split).
        """
        n = projections.shape[0]
        sorted_x = torch.sort(projections).values

        # standardise
        mu = sorted_x.mean()
        sigma = sorted_x.std(unbiased=True)
        z = (sorted_x - mu) / (sigma + 1e-12)

        # CDF of standard normal via error function
        import math
        def norm_cdf(x: torch.Tensor) -> torch.Tensor:
            return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

        Fz = norm_cdf(z)

        # avoid log(0)
        eps = 1e-12
        Fz = Fz.clamp(eps, 1 - eps)

        A2 = -n - torch.sum(
            (2 * torch.arange(1, n + 1, device=projections.device) - 1)
            * (torch.log(Fz) + torch.log(1 - Fz.flip(0)))
        ) / n

        # critical value at 5% significance (for testing normality)
        critical_value_5pct = 0.757
        return A2 < critical_value_5pct

    # ------------------------------------------------------------------
    # Recursive split (same logic as the original, but on GPU)
    # ------------------------------------------------------------------

    def _recursive_split(
        self,
        cluster_points: torch.Tensor,
        cluster_indices: torch.Tensor,
        current_depth: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_points = cluster_points.shape[0]
        n_features = cluster_points.shape[1]

        if n_points < n_features * 2 or self._global_cluster_count >= self.k_max:
            return (
                torch.zeros(n_points, dtype=torch.long, device=self.device),
                cluster_points.mean(dim=0, keepdim=True),
            )

        sub_labels, centers = self._kmeans_split(cluster_points)

        unique_labels = torch.unique(sub_labels)
        if len(unique_labels) < 2:
            return (
                torch.zeros(n_points, dtype=torch.long, device=self.device),
                cluster_points.mean(dim=0, keepdim=True),
            )

        counts = torch.bincount(sub_labels, minlength=2)
        if counts.min() < n_features:
            return (
                torch.zeros(n_points, dtype=torch.long, device=self.device),
                cluster_points.mean(dim=0, keepdim=True),
            )

        center1, center2 = centers[0], centers[1]
        projections = self._project_to_line(cluster_points, center1, center2)
        is_normal = self._anderson_test_torch(projections)

        if is_normal:
            return (
                torch.zeros(n_points, dtype=torch.long, device=self.device),
                cluster_points.mean(dim=0, keepdim=True),
            )

        self._global_cluster_count += 1
        if self._global_cluster_count > self.k_max:
            self._global_cluster_count -= 1
            return (
                torch.zeros(n_points, dtype=torch.long, device=self.device),
                cluster_points.mean(dim=0, keepdim=True),
            )

        mask0 = sub_labels == 0
        mask1 = sub_labels == 1

        labels0, centers0 = self._recursive_split(
            cluster_points[mask0], cluster_indices[mask0], current_depth + 1
        )
        labels1, centers1 = self._recursive_split(
            cluster_points[mask1], cluster_indices[mask1], current_depth + 1
        )

        n_clusters0 = len(torch.unique(labels0))
        n_clusters1 = len(torch.unique(labels1))

        merged_labels = torch.zeros(n_points, dtype=torch.long, device=self.device)
        merged_labels[mask0] = labels0
        merged_labels[mask1] = labels1 + n_clusters0

        merged_centers = torch.cat([centers0, centers1], dim=0)

        return merged_labels, merged_centers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, features, feature_type="original"):
        """Fit G-means on *features* (np.ndarray or torch.Tensor)."""
        self.feature_type = feature_type
        print(f"\n[G-means] Clustering with {feature_type} features...")
        print(f"  - Input shape: {features.shape}")
        print(f"  - Max clusters k_max: {self.k_max}")
        print(f"  - Device: {self.device.type}")

        if isinstance(features, np.ndarray):
            X = torch.from_numpy(features).float().to(self.device)
        else:
            X = features.clone().to(self.device).float()

        self._global_cluster_count = 1
        n_samples = X.shape[0]
        all_indices = torch.arange(n_samples, device=self.device)

        self.labels_, self.cluster_centers_ = self._recursive_split(
            X, all_indices, current_depth=0
        )
        self.n_clusters = len(torch.unique(self.labels_))

        # move back to numpy for compatibility
        self.labels_ = self.labels_.cpu().numpy()
        self.cluster_centers_ = self.cluster_centers_.cpu().numpy()

        print(f"  Done: {self.n_clusters} clusters found")

        unique, counts = np.unique(self.labels_, return_counts=True)
        print(f"  - Cluster size distribution:")
        for cluster_id, count in zip(unique, counts):
            print(f"      Cluster {cluster_id}: {count} nodes ({count / n_samples * 100:.1f}%)")

        return self

    def predict(self, features):
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet")
        if isinstance(features, np.ndarray):
            features_t = torch.from_numpy(features).float().to(self.device)
        else:
            features_t = features.clone().to(self.device)
        centres_t = torch.from_numpy(self.cluster_centers_).to(self.device)

        distances = torch.cdist(features_t, centres_t, p=2)
        labels = distances.argmin(dim=1).cpu().numpy()
        return labels

    def compute_cluster_representations(self, features, labels=None):
        if labels is None:
            labels = self.labels_
        if labels is None:
            raise ValueError("No cluster labels available")
        n_clusters = len(np.unique(labels))
        feature_dim = features.shape[1]
        cluster_reps = np.zeros((n_clusters, feature_dim))
        for i in range(n_clusters):
            mask = labels == i
            if mask.sum() > 0:
                cluster_reps[i] = features[mask].mean(axis=0)
        return cluster_reps
