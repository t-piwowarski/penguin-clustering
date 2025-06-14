import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

def evaluate_clusters(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Evaluates clustering using internal metrics.

    Parameters:
        X (np.ndarray): Feature matrix (can be PCA-reduced or original)
        labels (np.ndarray): Cluster labels

    Returns:
        dict: Dictionary with silhouette, Davies-Bouldin and Calinski-Harabasz scores
    """
    result = {}

    if len(set(labels)) <= 1 or all(label == -1 for label in labels):
        # Not a valid clustering
        result['silhouette'] = -1
        result['davies_bouldin'] = -1
        result['calinski_harabasz'] = -1
        return result

    result['silhouette'] = silhouette_score(X, labels)
    result['davies_bouldin'] = davies_bouldin_score(X, labels)
    result['calinski_harabasz'] = calinski_harabasz_score(X, labels)
    
    print("\n=== Clustering Evaluation Metrics ===")
    print(f"Silhouette Score      : {result['silhouette']:.3f}")
    print(f"Davies-Bouldin Index  : {result['davies_bouldin']:.3f}")
    print(f"Calinski-Harabasz     : {result['calinski_harabasz']:.3f}")

    return result

def count_cluster_members(labels: np.ndarray) -> pd.DataFrame:
    """
    Counts how many points are assigned to each cluster.

    Parameters:
        labels (np.ndarray): Cluster labels

    Returns:
        pd.DataFrame: Cluster ID and number of points in each
    """
    unique, counts = np.unique(labels, return_counts=True)
    data = {
        'Cluster': unique,
        'Count': counts
    }
    return pd.DataFrame(data)
