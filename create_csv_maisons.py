import pandas as pd
import numpy as np

# Nombre d'exemples
n = 1000
np.random.seed(42)  # pour reproductibilité

# Génération des features
chambres = np.random.randint(1, 7, n)  # entre 1 et 6 chambres
surface = np.random.randint(20, 251, n)  # surface habitable en m²
jardin = np.random.randint(0, 501, n)  # surface jardin en m²

# Génération du prix : combinaison linéaire + un peu de bruit
prix = (
    chambres * 20000 +
    surface * 1500 +
    jardin * 300 +
    np.random.normal(0, 10000, n)  # bruit gaussien
).astype(int)

# Créer le DataFrame
df = pd.DataFrame({
    "Chambres": chambres,
    "Surface_m2": surface,
    "Jardin_m2": jardin,
    "Prix_euros": prix
})

# Sauvegarde en CSV
df.to_csv("maisons.csv", index=False, encoding="utf-8")

print("✅ Fichier maisons.csv généré avec", n, "maisons !")
print(df.head())
