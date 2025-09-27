import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1️⃣ Charger les données
df = pd.read_csv("maisons.csv")

# 2️⃣ Features et cible
X = df[["Chambres", "Surface_m2", "Jardin_m2"]]
y = df["Prix_euros"]

# 3️⃣ Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Créer et entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

# 5️⃣ Récupérer coefficients et intercept
coefficients = model.coef_
intercept = model.intercept_

# 6️⃣ Exemple : maison fictive
maison = {
    "Chambres": 4,
    "Surface_m2": 150,
    "Jardin_m2": 100
}

# Calcul manuel du prix
prix_calcul = (
    coefficients[0] * maison["Chambres"] +
    coefficients[1] * maison["Surface_m2"] +
    coefficients[2] * maison["Jardin_m2"] +
    intercept
)

# Calcul avec le modèle (pour vérifier)
prix_modele = model.predict([[maison["Chambres"], maison["Surface_m2"], maison["Jardin_m2"]]])[0]

print(f"Maison : {maison}")
print(f"Prix calculé manuellement : {prix_calcul:.2f} €")
print(f"Prix prédit par le modèle : {prix_modele:.2f} €")
