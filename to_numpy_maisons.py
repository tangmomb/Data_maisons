import pandas as pd
import numpy as np

# Charger le CSV avec pandas
df = pd.read_csv("maisons.csv")

# Conversion en numpy array
X = df.to_numpy()

print(type(X))    # <class 'numpy.ndarray'>
print(X.shape)    # dimensions du tableau
print(X[:5])      # 5 premières lignes

X = df[["Chambres", "Surface_m2", "Jardin_m2"]].to_numpy()
y = df["Prix_euros"].to_numpy()

print("X shape:", X.shape)
print("y shape:", y.shape)

# Sauvegarde en CSV
np.savetxt("maisons_numpy.csv", X, delimiter=",", fmt="%d", header="Chambres,Surface_m2,Jardin_m2", comments="")
np.savetxt("prix_numpy.csv", y, delimiter=",", fmt="%d", header="Prix_euros", comments="")
