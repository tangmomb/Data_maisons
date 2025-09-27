import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Charger le CSV
df = pd.read_csv("maisons_cluster_multiples.csv")

# Features
features = ["garden", "surface", "bedrooms", "bathrooms"]
X = df[features]

# Normaliser
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_
df["cluster"] = labels

# PCA 2D pour visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Centres des clusters en PCA
centers_pca = pca.transform(kmeans.cluster_centers_)

# Composantes principales
pca_components = pd.DataFrame(pca.components_, columns=features, index=["PC1", "PC2"])
print(pca_components)

# Scatter plot
plt.figure(figsize=(8,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="viridis", s=50, alpha=0.6)
plt.scatter(centers_pca[:,0], centers_pca[:,1], c='red', s=200, marker='X', label='Centres')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Clustering KMeans 2D avec tous les paramètres + centres")
plt.legend()
plt.show()
