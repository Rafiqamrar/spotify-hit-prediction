# 00_data_understanding.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Charger la dataset
# -----------------------------
df = pd.read_csv("../data/spotify_songs.csv")  # changer le chemin si besoin

print("Dimensions du dataset :", df.shape)
print("Colonnes :", df.columns)

# -----------------------------
# 2. Aperçu des données
# -----------------------------
print("\nAperçu des 5 premières lignes :")
print(df.head())

print("\nInformations sur les colonnes :")
print(df.info())

# -----------------------------
# 3. Valeurs manquantes
# -----------------------------
missing = df.isnull().sum()
print("\nValeurs manquantes par colonne :")
print(missing[missing > 0])

# -----------------------------
# 4. Statistiques descriptives
# -----------------------------
print("\nStatistiques descriptives des colonnes numériques :")
print(df.describe())

# -----------------------------
# 5. Valeurs uniques et distributions
# -----------------------------
print("\nNombre de genres uniques :", df['playlist_genre'].nunique())
print("Nombre d'artistes uniques :", df['track_artist'].nunique())

# -----------------------------
# 6. Analyse rapide de la popularité
# -----------------------------
plt.figure(figsize=(10,6))
sns.histplot(df['track_popularity'], bins=30, kde=True, color='skyblue')
plt.title("Distribution de la popularité des morceaux")
plt.xlabel("Popularity")
plt.ylabel("Count")
plt.show()

# -----------------------------
# 7. Corrélations simples
# -----------------------------
numeric_features = ['danceability', 'energy', 'loudness', 'speechiness',
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

plt.figure(figsize=(12,8))
corr = df[numeric_features + ['track_popularity']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Corrélations entre features audio et popularité")
plt.show()

# -----------------------------
# 8. Détection de doublons
# -----------------------------
duplicates = df.duplicated(subset='track_id').sum()
print("\nNombre de doublons sur track_id :", duplicates)

# -----------------------------
# Fin
# -----------------------------
print("\nData understanding terminé ! Dataset prêt pour le nettoyage et EDA.")
