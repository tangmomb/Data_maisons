import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le CSV
df = pd.read_csv("maisons.csv")

sns.pairplot(df, hue="Chambres", diag_kind="kde")
plt.suptitle("Relations entre toutes les variables", y=1.02)
plt.show()

