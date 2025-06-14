import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def calculate_elbow_values(X: np.ndarray, max_k: int = 10) -> list:
    """
    Computes inertia values for different numbers of clusters (k) using KMeans.

    Parameters:
        X (np.ndarray): Input feature matrix
        max_k (int): Maximum number of clusters to evaluate

    Returns:
        list: Inertia values for k in range(1, max_k+1)
    """
    inertia_values = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)
    return inertia_values

def plot_elbow_curve(inertia_values: list):
    """
    Plots the elbow curve for KMeans clustering.

    Parameters:
        inertia_values (list): Inertia values corresponding to each k
    """
    k_values = range(1, len(inertia_values) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia_values, marker='o', color='blue')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(k_values)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def perform_kmeans(X: np.ndarray, k: int = 4) -> tuple:
    """
    Runs KMeans clustering and returns labels and cluster centers.

    Parameters:
        X (np.ndarray): Input data matrix
        k (int): Number of clusters

    Returns:
        tuple: (labels, cluster_centers)
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    return labels, centers

def plot_kmeans_pca(X: np.ndarray, labels: np.ndarray):
    """
    Reduces data to 2D using PCA and visualizes the clusters.

    Parameters:
        X (np.ndarray): Input data (before or after PCA)
        labels (np.ndarray): Cluster labels from KMeans
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k')
    plt.title(f'KMeans clustering with {len(np.unique(labels))} clusters (PCA-reduced)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def count_cluster_members(labels: np.ndarray) -> dict:
    """
    Counts number of elements in each cluster.

    Parameters:
        labels (np.ndarray): Cluster labels

    Returns:
        dict: {cluster_id: count}
    """
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))
