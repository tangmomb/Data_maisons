import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv("maisons_cluster_multiples.csv")

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

# PCA 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Ajouter les labels KMeans
df["Cluster"] = kmeans.labels_

# Colonnes à analyser
features = ["garden", "surface", "bedrooms", "bathrooms"]

# Minimum par cluster
print("🔹 Minimum par cluster :")
print(df.groupby("Cluster")[features].min())

# Maximum par cluster
print("\n🔹 Maximum par cluster :")
print(df.groupby("Cluster")[features].max())

# Points très éloignés du centre de leur cluster
import numpy as np

for i in range(kmeans.n_clusters):
    # Points du cluster i
    cluster_points = X_scaled[kmeans.labels_ == i]
    cluster_indices = np.where(kmeans.labels_ == i)[0]  # indices dans le DataFrame
    
    # Centre du cluster
    center = kmeans.cluster_centers_[i]
    
    # Distances euclidiennes au centre
    distances = np.linalg.norm(cluster_points - center, axis=1)
    
    # Indices des 2 points les plus éloignés
    farthest_idx = distances.argsort()[-4:]
    
    print(f"\nCluster {i} - 2 points les plus éloignés :")
    print(df.iloc[cluster_indices[farthest_idx]])

# Prédiction pour une nouvelle maison
def predict_cluster(new_house, kmeans, scaler):
    """
    Prédit le cluster pour une nouvelle maison.
    
    new_house : liste [garden, surface, bedrooms, bathrooms]
    kmeans : modèle KMeans entraîné
    scaler : scaler utilisé pour normaliser
    """
    new_house_scaled = scaler.transform([new_house])
    cluster = kmeans.predict(new_house_scaled)[0]
    return cluster

# Exemple
house_to_predict = [92, 132, 3, 2]  # garden, surface, bedrooms, bathrooms
cluster = predict_cluster(house_to_predict, kmeans, scaler)
print(f"Cluster prédit : {cluster}")


# Transformer la nouvelle maison avec la PCA
new_house = [house_to_predict]
new_house_scaled = scaler.transform(new_house)
new_house_pca = pca.transform(new_house_scaled)

# Visualisation avec la nouvelle maison en rouge
plt.figure(figsize=(8,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="viridis", s=50, alpha=0.6, label="Maisons existantes")
plt.scatter(new_house_pca[:,0], new_house_pca[:,1], c='red', s=200, marker='X', label="Nouvelle maison")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Clustering KMeans 2D (projection PCA) + nouvelle maison")
plt.legend()
plt.show()
