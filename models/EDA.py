# 01_EDA.py
# --------------------------------------
# Exploratory Data Analysis (EDA)
# Objectif : Identifier tendances, patterns et relations
# entre les features audio et la popularité
# --------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Charger les données
# -----------------------------
df = pd.read_csv("../data/spotify_songs.csv")

# Nettoyage léger pour EDA
df = df.dropna(subset=["track_name", "track_artist", "track_album_name"])

# -----------------------------
# 2. Distribution de la popularité
# -----------------------------
plt.figure(figsize=(10, 6))
sns.histplot(df["track_popularity"], bins=30, kde=True)
plt.title("Distribution de la popularité des morceaux")
plt.xlabel("Track Popularity")
plt.ylabel("Nombre de morceaux")
plt.show()

# -----------------------------
# 3. Popularité par genre
# -----------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x="playlist_genre", y="track_popularity", data=df)
plt.title("Popularité par genre musical")
plt.xlabel("Genre")
plt.ylabel("Track Popularity")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 4. Relation features audio vs popularité
# -----------------------------
audio_features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms"
]

for feature in audio_features:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[feature], y=df["track_popularity"], alpha=0.3)
    plt.title(f"{feature} vs Popularité")
    plt.xlabel(feature)
    plt.ylabel("Track Popularity")
    plt.show()

# -----------------------------
# 5. Heatmap de corrélation
# -----------------------------
plt.figure(figsize=(12, 8))
corr = df[audio_features + ["track_popularity"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Corrélations entre features audio et popularité")
plt.show()

# -----------------------------
# 6. Popularité selon tempo (binning)
# -----------------------------
df["tempo_bin"] = pd.cut(df["tempo"], bins=5)

plt.figure(figsize=(10, 6))
sns.boxplot(x="tempo_bin", y="track_popularity", data=df)
plt.title("Popularité selon le tempo")
plt.xlabel("Intervalle de tempo (BPM)")
plt.ylabel("Track Popularity")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# Fin
# -----------------------------
print("EDA terminé : insights visuels extraits avec succès.")
