from src.preprocessing import load_and_clean_data, impute_missing_values, standardize_data
from src.clustering_kmeans import calculate_elbow_values, plot_elbow_curve, perform_kmeans, plot_kmeans_pca
from src.clustering_dbscan import find_best_dbscan_params, run_dbscan, plot_dbscan_pca
from src.feature_selection import wrapper_feature_selection, plot_feature_subplots
from src.evaluation import evaluate_clusters, count_cluster_members

def main():
    print("=== Clustering Penguins Dataset ===\n")

    # Load and preprocess data
    df = load_and_clean_data('data/penguins.csv')
    df = impute_missing_values(df)
    X = standardize_data(df).values

    # === KMeans Clustering ===
    print(">>> KMeans Clustering")
    inertia = calculate_elbow_values(X)
    plot_elbow_curve(inertia)

    labels_kmeans, _ = perform_kmeans(X, k=4)
    plot_kmeans_pca(X, labels_kmeans)

    metrics_kmeans = evaluate_clusters(X, labels_kmeans)
    print(count_cluster_members(labels_kmeans), "\n")

    # === DBSCAN (Full features) ===
    print(">>> DBSCAN (All Features)")
    best_params = find_best_dbscan_params(X)
    print("Best parameters:", best_params)
    
    best_params = find_best_dbscan_params(X)
    labels = run_dbscan(X, **best_params)
    plot_dbscan_pca(X, labels, title_suffix="(All Features)")

    labels_dbscan = run_dbscan(X, **best_params)
    plot_dbscan_pca(X, labels_dbscan, title_suffix="(All Features)")

    metrics_dbscan = evaluate_clusters(X, labels_dbscan)
    print(count_cluster_members(labels_dbscan), "\n")

    # === Wrapper-Based Feature Selection ===
    print(">>> Wrapper-Based Feature Selection")
    best_features, score = wrapper_feature_selection(X, **best_params)
    print("Best feature indices:", best_features)
    print("Best silhouette score:", score)

    X_selected = X[:, best_features]
    labels_wrapped = run_dbscan(X_selected, **best_params)

    selected_names = df.columns[list(best_features)]
    plot_feature_subplots(X_selected, labels_wrapped, selected_names.tolist())

    plot_dbscan_pca(X_selected, labels_wrapped, title_suffix="(Selected Features)")
    metrics_wrapped = evaluate_clusters(X_selected, labels_wrapped)
    print(count_cluster_members(labels_wrapped), "\n")


if __name__ == "__main__":
    main()
