# 🐧 Clustering Penguins Dataset – DBSCAN & KMeans
This repository contains a machine learning project for unsupervised clustering and feature selection on the **Palmer Penguins dataset**. The goal is to discover natural groupings in the data based on physical characteristics of penguins using two clustering methods: **KMeans** and **DBSCAN**.

The project includes extensive data preprocessing, cluster evaluation using internal metrics, grid search-based optimization of DBSCAN parameters, and wrapper-based feature selection.

## ✨ Features:
  - Data cleaning, outlier removal, and KNN-based imputation
  - Feature standardization using `StandardScaler`
  - Clustering with:
    - KMeans (with elbow method)
    - DBSCAN (with GridSearchCV and silhouette scoring)
  - PCA-based 2D visualization of clustering results
  - Wrapper-based feature selection using DBSCAN and silhouette score
  - Evaluation using internal clustering metrics:
    - Silhouette Score
    - Davies-Bouldin Index
    - Calinski-Harabasz Score
   
---

## 🦜 Dataset: Palmer Penguins

The dataset contains physical measurements of penguins
  - culmen_length_mm (bill length)
  - culmen_depth_mm (bill depth)
  - flipper_length_mm
  - body_mass_g
  - sex (binary: FEMALE, MALE, with missing values)

---

## 🔢 Exploratory Data Analysis (EDA)

### Data Quality

  - Two rows with completely missing values were removed.
  | id | culmen_length_mm | culmen_depth_mm | flipper_length_mm | body_mass_g | sex |
  |:--:|:----------------:|:---------------:|:-----------------:|:-----------:|:---:|
  | 3 | NaN | NaN | NaN | NaN | NaN |
  | 8 | 34.1 | 18.1 | 193.0 | 3475.0 | NaN |
  | 10 | 37.8 | 17.1 | 186.0 | 3300.0 | NaN |
  | 11 | 37.8 | 17.3 | 180.0 | 3700.0 | NaN |
  | 47 | 37.5 | 18.9 | 179.0 | 2975.0 | NaN |
  | 246 | 44.5 | 14.3 | 216.0 | 4100.0 | NaN |
  | 286 | 46.2 | 14.4 | 214.0 | 4650.0 | NaN |
  | 324 | 47.3 | 13.8 | 216.0 | 4725.0 | NaN |
  | 339 | NaN | NaN | NaN | NaN | NaN |
  - One flipper length was negative, and one exceeded 500mm → both were removed.

  - The column `sex` contained values ".", which were replaced with `NaN`, and then encoded: `MALE` = 1, `FEMALE` = 0.

### Missing Values

  - Missing values were imputed using **KNNImputer** with `n_neighbors=5`.

  - The dataset was standardized using **StandardScaler** before clustering.

### Outlier Detection

  - Boxplots showed extreme outliers in `flipper_length_mm` which were removed.
