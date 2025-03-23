# 🐧 Clustering Penguins Dataset – DBSCAN & KMeans
Machine learning project for unsupervised clustering and feature selection on the Palmer Penguins dataset.\
Uses standardized KMeans and DBSCAN algorithms to identify natural groupings of samples based on biological features.\
Includes grid search optimization (silhouette-based) for DBSCAN parameters and wrapper-based feature selection.\

✨ Features:
  - Data cleaning, outlier removal, and KNN-based imputation
  - Feature standardization using StandardScaler
  - Clustering with:
    - KMeans (with elbow method)
    - DBSCAN (with GridSearchCV and silhouette scoring)
  - PCA-based 2D visualization of clustering results
  - Wrapper-based feature selection using DBSCAN and silhouette score
  - Evaluation using internal clustering metrics:
    - Silhouette Score
    - Davies-Bouldin Index
    - Calinski-Harabasz Score
