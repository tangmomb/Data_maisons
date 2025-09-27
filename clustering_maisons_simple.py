import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1️⃣ Charger les données
df = pd.read_csv("maisons_cluster_simple.csv")

# 2️⃣ Features à utiliser pour le clustering
X = df[["Surface_m2", "Jardin_m2"]]

# 3️⃣ Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Clustering KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Ajouter les clusters au DataFrame
df["Cluster"] = clusters

# Minimum par cluster
print("🔹 Minimum par cluster :")
print(df.groupby("Cluster")[["Surface_m2","Jardin_m2"]].min())

# Maximum par cluster
print("\n🔹 Maximum par cluster :")
print(df.groupby("Cluster")[["Surface_m2","Jardin_m2"]].max())

# 5️⃣ Prédiction pour une nouvelle maison

# Nouvelle maison à tester : [Surface_m2, Jardin_m2]
nouvelle_maison = [[300, 300]]  # Exemple : m² surface, m² jardin

# Normaliser la nouvelle maison avec le même scaler
nouvelle_maison_scaled = scaler.transform(nouvelle_maison)

# Prédire le cluster
cluster_pred = kmeans.predict(nouvelle_maison_scaled)[0]

# Interprétation : quel cluster correspond aux grandes maisons ?
moyennes_clusters = df.groupby("Cluster")[["Surface_m2","Jardin_m2"]].mean()
grand_cluster = moyennes_clusters["Surface_m2"].idxmax()  # cluster avec plus grande surface

if cluster_pred == grand_cluster:
    categorie = "Grande maison"
else:
    categorie = "Petite maison"

print(f"La nouvelle maison appartient au cluster : {cluster_pred} → {categorie}")


#  6️⃣ Visualisation
plt.figure(figsize=(8,6))
scatter = plt.scatter(df["Surface_m2"], df["Jardin_m2"], c=df["Cluster"], cmap="viridis", alpha=0.7)

# Légende automatique
handles, labels = scatter.legend_elements()
plt.legend(handles, ["Grandes maisons","Petites maisons"], title="Cluster")
plt.xlabel("Surface m²")
plt.ylabel("Jardin m²")
plt.title("Clustering des maisons : petit vs grand")
plt.scatter(nouvelle_maison[0][0], nouvelle_maison[0][1], color='red', s=100, label="Nouvelle maison")
plt.show()


#  7️⃣ Visualisation avec frontière de décision
import numpy as np

# Créer une grille sur tout le graphique
x_min, x_max = df["Surface_m2"].min() - 10, df["Surface_m2"].max() + 10
y_min, y_max = df["Jardin_m2"].min() - 10, df["Jardin_m2"].max() + 10
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Mettre la grille dans le même format que les features et normaliser
grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)

# Prédire le cluster pour chaque point de la grille
Z = kmeans.predict(grid_scaled)
Z = Z.reshape(xx.shape)

# Tracer la frontière avec contourf (couleur de fond)
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.2, cmap="viridis")

# Scatter des maisons réelles
plt.scatter(df["Surface_m2"], df["Jardin_m2"], c=df["Cluster"], cmap="viridis", alpha=0.7)

# Nouvelle maison en rouge
plt.scatter(nouvelle_maison[0][0], nouvelle_maison[0][1], color='red', s=100, label="Nouvelle maison")

plt.xlabel("Surface m²")
plt.ylabel("Jardin m²")
plt.title("Clustering des maisons avec frontière KMeans")
plt.legend()
plt.show()
