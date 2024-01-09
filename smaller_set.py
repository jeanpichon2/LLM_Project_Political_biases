import pandas as pd

# Spécifiez le chemin vers votre fichier texte
chemin_fichier = 'datasets/reddit_center.txt'

# Lisez le fichier ligne par ligne
with open(chemin_fichier, 'r', encoding='utf-8') as fichier:
    lignes = fichier.readlines()

# Créez un dataframe à partir des lignes
df = pd.DataFrame({'Texts': lignes})

# Affichez les premières lignes du dataframe
print(df.head())

subdf = df.sample(frac = 0.1)

subdf.to_csv('datasets/small_reddit_center.csv', index=False)

print(len(subdf))