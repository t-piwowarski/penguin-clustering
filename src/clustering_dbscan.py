import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def silhouette_scorer(estimator, X):
    """
    Custom silhouette score function for use with GridSearchCV.

    Parameters:
        estimator: Clustering estimator (DBSCAN)
        X (np.ndarray): Input data

    Returns:
        float: Silhouette score or -1 if invalid
    """
    labels = estimator.fit_predict(X)
    if len(set(labels)) > 1 and -1 not in set(labels):  # Ensure valid clustering
        return silhouette_score(X, labels)
    else:
        return -1

def find_best_dbscan_params(X: np.ndarray,
                            eps_range=np.linspace(0.1, 2.0, 80),
                            min_samples_range=range(1, 10)) -> dict:
    """
    Manually searches for best DBSCAN parameters using silhouette score.

    Parameters:
        X (np.ndarray): Input data
        eps_range (iterable): Range of epsilon values
        min_samples_range (iterable): Range of min_samples values

    Returns:
        dict: {'eps': best_eps, 'min_samples': best_min_samples}
    """
    best_score = -1
    best_params = {'eps': None, 'min_samples': None}

    for eps in eps_range:
        for min_samples in min_samples_range:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X)
            if len(set(labels)) > 1 and -1 not in set(labels):  # valid clustering
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}

    return best_params




def run_dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Runs DBSCAN with given parameters and returns labels.

    Parameters:
        X (np.ndarray): Input data
        eps (float): Neighborhood radius
        min_samples (int): Minimum samples to form a core point

    Returns:
        np.ndarray: Cluster labels
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels

def plot_dbscan_pca(X: np.ndarray, labels: np.ndarray, title_suffix: str = ""):
    """
    Applies PCA and plots DBSCAN clustering results in 2D if possible.

    Parameters:
        X (np.ndarray): Input data
        labels (np.ndarray): Cluster labels from DBSCAN
        title_suffix (str): Optional title text
    """
    if X.shape[1] < 2:
        print("[Warning] Cannot plot PCA – less than 2 features selected.")
        return

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k')
    plt.title(f'DBSCAN Clustering (PCA-reduced) {title_suffix}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


