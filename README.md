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

  - Two rows with completely missing values were removed (`id=3` and `id=339`).

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

  | id | culmen_length_mm | culmen_depth_mm | flipper_length_mm | body_mass_g | sex |
  |:--:|:----------------:|:---------------:|:-----------------:|:-----------:|:---:|
  | 14 | 34.6 | 21.1 | -132.0 | 4400.0 | MALE |

  - The column `sex` contained values ".", which were replaced with `NaN`, and then encoded: `MALE` = 1, `FEMALE` = 0.

### Missing Values

  - Missing values (`id=8`, `id=10`, `id=11`, `id=47`, `id=246`, `id=286`, `id=324`) were imputed using **KNNImputer** with `n_neighbors=5`.
  - The dataset was standardized using **StandardScaler** before clustering.

### Outlier Detection

  - Boxplots showed extreme outliers in `flipper_length_mm` which were removed.

## tu ma być obrazek

---

## 🔍 Clustering Algorithms

### KMeans Clustering

  - The elbow method was used to determine the optimal number of clusters (`k = 4`).
## tu ma być obrazek
  - PCA was applied to visualize the clusters in 2D.
## tu ma być obrazek

### DBSCAN Clustering

  - Hyperparameters (`eps`, `min_samples`) were optimized using **own algoritm** with a custom silhouette scoring function.
  - DBSCAN was able to find meaningful groupings including noise points. PCA was used to visualize the clusters in 2D.
## tu ma być obrazek

---

## 🔍 Feature Selection

### Wrapper-Based Selection

  - Subsets of features were evaluated using DBSCAN with fixed parameters (`eps=1.086`, `min_samples=5`).
  - Silhouette score was used to evaluate each subset.
  - The best performing combination was selected and used for final clustering.

---

## ✅ Results summary

  Internal metrics used:
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Index

  Each clustering result was visualized using:
  - PCA 2D scatter plots
  - Subplots of feature pairs

---

## 📂 Repository structure

penguins-clustering\
│── data\
│   │── penguins.csv\
│\
│── docs\
│   │── images\
│   │   │──\
│\
│── src\
│   │── preprocessing.py\
│   │── clustering_kmeans.py\
│   │── clustering_dbscan.py\
│   │── feature_selection.py\
│   │── evaluation.py\
│\
│── main.py\
│── README.md\
│── requirements.txt

---

## 🚀 Installation

1. **Clone repository:**

   ```bash
   git clone https://github.com/t-piwowarski/penguin-clustering.git
   cd text-generator
   ```
2. **Create and activate a virtual environment (optional but recommended):**
   
- On Windows:
     
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   
- On Linux/macOS:
     
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   
3. **Install the required packages:**
   
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the main pipeline:**

   ```bash
   python main.py
   ```

   This will:

    - Load and clean the data
    - Impute and scale features
    - Run KMeans and DBSCAN
    - Optimize DBSCAN parameters via GridSearchCV
    - Perform wrapper-based feature selection
    - Evaluate and visualize the results
