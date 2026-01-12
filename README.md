# Clustering-Reddit-Posts

This project performs **unsupervised clustering of Reddit posts** to discover latent topics and content patterns using classical NLP and machine learning techniques.

The pipeline loads Reddit data from Hugging Face, cleans and preprocesses text, converts posts into TF-IDF vectors, and benchmarks multiple clustering algorithms including **KMeans**, **Hierarchical Clustering**, and **DBSCAN**. Clustering quality is evaluated using unsupervised metrics and visualized using PCA.

---

## 🚀 Features

* Load large-scale Reddit dataset using Hugging Face `datasets`
* Text preprocessing: URL removal, stopword filtering, normalization
* Feature extraction using **TF-IDF**
* Multiple clustering algorithms:

  * KMeans
  * Agglomerative (Hierarchical)
  * DBSCAN
* Unsupervised evaluation:

  * Silhouette Score
  * Davies–Bouldin Index
* 2D cluster visualization using PCA + Matplotlib
* Modular, production-style codebase

---

## 🛠️ Installation

```bash
git clone <repo-url>
cd reddit-clustering
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python main.py
```

You can modify clustering parameters in `config.py`.

---

## 📁 Project Structure

```
reddit-clustering/
├── data/
├── preprocessing/
├── vectorization/
├── clustering/
├── evaluation/
├── visualization/
├── config.py
└── main.py
```

---

## 📈 Sample Output

* Cluster labels for each algorithm
* Evaluation metrics printed in console
* PCA scatter plots showing cluster separation

---

## 📌 Future Improvements

* Automatic selection of optimal cluster count
* Sentence embedding based clustering
* Interactive dashboard visualization

---

