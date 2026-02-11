📘 Phase 00 – Data Understanding

Projet : Prédiction du succès musical sur Spotify

🎯 Objectif de la phase

L’objectif de cette première phase est de comprendre la structure et la qualité du dataset Spotify, d’identifier les variables disponibles, de détecter d’éventuels problèmes de données (valeurs manquantes, doublons) et de vérifier la cohérence globale avant toute étape de nettoyage, d’analyse exploratoire ou de modélisation.

📂 Description du dataset

Nombre d’observations : 32 833 morceaux

Nombre de variables : 23

Source : Spotify audio features + métadonnées playlists

Types de variables

Numériques (13) :
audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms) + track_popularity

Catégorielles (10) :
artistes, albums, playlists, genres, sous-genres

🔍 Aperçu général des données

Les données sont globalement propres et cohérentes

Les valeurs numériques sont dans des plages réalistes d’un point de vue musical

La variable cible track_popularity varie de 0 à 100, avec une moyenne autour de 42

⚠️ Valeurs manquantes

Trois colonnes présentent des valeurs manquantes :

Colonne	Nombre de valeurs manquantes
track_name	5
track_artist	5
track_album_name	5

👉 Ces valeurs représentent moins de 0,02 % du dataset.
Décision : suppression des lignes concernées (impact négligeable).

🔁 Doublons

4 477 doublons détectés sur track_id

Un même morceau apparaît dans plusieurs playlists

Impact potentiel

Risque de biais lors de l’entraînement des modèles

Risque de fuite d’information

👉 Cette problématique sera traitée lors de la phase de nettoyage ou de feature engineering.

📊 Statistiques descriptives clés

Danceability moyenne : ~0.65

Energy moyenne : ~0.70

Valence moyenne : ~0.51

Tempo moyen : ~121 BPM

Durée médiane : ~216 000 ms (~3 min 36 s)

Ces valeurs sont cohérentes avec des morceaux populaires (pop, EDM, mainstream).

🎼 Genres et artistes

6 genres principaux

10 692 artistes uniques

👉 Le dataset est suffisamment varié pour limiter les biais liés à un artiste dominant.

🔗 Corrélations initiales

Une heatmap de corrélation a permis d’observer que :

Certaines features audio présentent de faibles corrélations linéaires avec la popularité

Aucune variable seule ne suffit à expliquer le succès d’un morceau

👉 Cela justifie l’utilisation de modèles multivariés et non linéaires (réseaux neuronaux, modèles bayésiens, SVM).

✅ Conclusion de la phase 00

Le dataset est de bonne qualité, propre et exploitable

Les problèmes identifiés (doublons, variables catégorielles) sont maîtrisables

Les données sont adaptées à une approche Machine Learning / IA

👉 La phase Data Understanding est validée.
Le projet peut passer à l’Analyse Exploratoire des Données (EDA).