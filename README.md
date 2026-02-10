✈️ NASA Engine RUL Predictor - Smart Maintenance
Une solution de maintenance prédictive interactive permettant d'estimer la Durée de Vie Utile Restante (RUL - Remaining Useful Life) de turboréacteurs à partir des jeux de données CMAPSS de la NASA.

🚀 Démo en direct : [https://rulpredictionbyldesloges.streamlit.app]

📝 Présentation du projet
Ce projet utilise le Machine Learning pour transformer les flux de télémétrie bruts (température, pression, vitesse) en indicateurs de maintenance actionnables. L'objectif est de prédire le nombre de cycles restants avant une défaillance moteur pour optimiser les révisions et garantir la sécurité des vols.

⚙️ Méthodologie & Engineering
Le pipeline repose sur un traitement de signal et une architecture statistique précise :

Lissage de Signal (Smoothing) : Utilisation d'une moyenne mobile (rolling mean) sur une fenêtre de 15 cycles pour filtrer le bruit thermique et les pics parasites des capteurs.

Feature Engineering Temporel : * Volatilité : Calcul de l'écart-type glissant (std) pour détecter les instabilités de fonctionnement.

Tendance : Calcul du gradient (diff) pour mesurer la vitesse de dégradation.

Target Clipping : La RUL est plafonnée à 125 cycles. On considère mathématiquement que l'usure n'est pas linéairement détectable au-delà de ce seuil, ce qui stabilise l'apprentissage du modèle.

Random Forest Regressor : Un modèle d'ensemble de 100 arbres de décision pour capturer les relations non-linéaires complexes entre les 21 capteurs.

🛠️ Stack Technique
Langage : Python 🐍

Data Science : Pandas, NumPy

Machine Learning : Scikit-learn (Random Forest, MinMaxScaler)

Visualisation : Matplotlib (Graphiques de tendance), Streamlit (Interface)

Persistance : Joblib (Sérialisation des modèles et scalers)

📂 Structure des fichiers
RUL.py : Le moteur du projet. Contient le pipeline de nettoyage, le feature engineering et le script d'entraînement.

app.py : L'interface utilisateur interactive Streamlit.

data/ : Contient les jeux de données bruts train_FD001.txt et test_FD001.txt.

model_RUL.pkl : Le modèle entraîné prêt pour l'inférence.

requirements.txt : Liste des dépendances pour un déploiement rapide.
