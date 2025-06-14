import numpy as np
from itertools import combinations
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import itertools

def evaluate_model(X: np.ndarray, feature_indices: tuple, eps: float = 1.086, min_samples: int = 5) -> float:
    """
    Evaluates clustering quality (silhouette score) for a given subset of features.

    Parameters:
        X (np.ndarray): Input data matrix
        feature_indices (tuple): Indices of selected features
        eps (float): DBSCAN neighborhood radius
        min_samples (int): DBSCAN minimum samples to form a core point

    Returns:
        float: Silhouette score (or -1 if only one cluster)
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    subset = X[:, feature_indices]
    labels = dbscan.fit_predict(subset)

    if len(set(labels)) > 1 and -1 not in set(labels):
        return silhouette_score(subset, labels)
    return -1  # invalid clustering (e.g., all points in same cluster or noise)

def wrapper_feature_selection(X: np.ndarray, eps: float = 1.086, min_samples: int = 5) -> tuple:
    """
    Performs wrapper-based feature selection using DBSCAN and silhouette score.

    Parameters:
        X (np.ndarray): Input feature matrix
        eps (float): DBSCAN epsilon parameter
        min_samples (int): DBSCAN min_samples parameter

    Returns:
        tuple: (best_feature_indices, best_silhouette_score)
    """
    num_features = X.shape[1]
    features_to_use = [i for i in range(num_features) if i != 4]  # exclude 'sex'

    best_score = -1
    best_features = None

    for k in range(2, len(features_to_use) + 1):  # Require at least 2 features
        for subset in combinations(features_to_use, k):
            score = evaluate_model(X, subset, eps, min_samples)
            if score > best_score:
                best_score = score
                best_features = subset

    return best_features, best_score

def plot_feature_subplots(X, labels, feature_names, title="DBSCAN Clustering (Wrapper-Based Selection)"):
    """
    Plots 2x3 subplots of 2D scatterplots for each feature pair, colored by cluster labels.

    Parameters:
        X (np.ndarray): Feature matrix [n_samples, n_features]
        labels (np.ndarray): Cluster labels
        feature_names (list[str]): Feature names (length must match X.shape[1])
        title (str): Title of the whole plot
    """
    pairs = list(itertools.combinations(range(len(feature_names)), 2))[:6]  # first 6 pairs
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (idx_x, idx_y) in enumerate(pairs):
        ax = axes[i]
        scatter = ax.scatter(X[:, idx_x], X[:, idx_y], c=labels, cmap='viridis', edgecolor='k', s=40, alpha=0.8)
        ax.set_xlabel(feature_names[idx_x])
        ax.set_ylabel(feature_names[idx_y])
        ax.set_title(f"{feature_names[idx_x]} vs {feature_names[idx_y]}")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Klastry")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

