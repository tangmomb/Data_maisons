# 1️⃣ Charger et préparer les données

import pandas as pd

# Charger le CSV
df = pd.read_csv("maisons.csv")

# Features (X) et cible (y)
X = df[["Chambres", "Surface_m2", "Jardin_m2"]]   # variables explicatives
y = df["Prix_euros"]                              # variable cible

# 2️⃣ Découper en train/test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ Entraîner un modèle de régression linéaire

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)  # entraînement

# 4️⃣ Évaluer le modèle

from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# 5️⃣ Visualiser les résultats

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Prix réel")
plt.ylabel("Prix prédit")
plt.title("Régression linéaire - Prédiction du prix des maisons")
plt.show()

# 6️⃣ Faire une prédiction

maison = [[4, 150, 100]]
prix_estime = model.predict(maison)
print("Prix estimé :", int(prix_estime[0]), "€")

#  7️⃣ Comparer quelques prédictions
for i in range(10):  # affiche les 10 premières
    reel = int(y_test.iloc[i])
    predit = int(y_pred[i])
    erreur = abs(reel - predit)
    print(f"Maison {i+1}: Réel = {reel} €, Prédit = {predit} €, Erreur = {erreur} €")


# 8️⃣ Afficher les coefficients de la régression
coefficients = model.coef_
intercept = model.intercept_

print("✅ Régression linéaire trouvée :")
print(f"Prix = {coefficients[0]:.2f} * Chambres + {coefficients[1]:.2f} * Surface_m2 + {coefficients[2]:.2f} * Jardin_m2 + {intercept:.2f}")

# Optionnel : afficher chaque coefficient séparément
for feature, coef in zip(X.columns, coefficients):
    print(f"{feature} -> coefficient = {coef:.2f}")
print(f"Intercept (constante) = {intercept:.2f}")

# 9️⃣ Calculer le prix d'une maison "à la main" avec les coefficients
prix_calcul = sum([c * v for c, v in zip(coefficients, maison[0])]) + intercept

# Vérification avec le modèle
prix_modele = model.predict(maison)[0]

print(f"Prix calculé manuellement : {prix_calcul:.2f} €")
print(f"Prix prédit par le modèle : {prix_modele:.2f} €")