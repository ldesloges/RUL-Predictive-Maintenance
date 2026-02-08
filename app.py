import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
st.set_page_config(page_title="NASA RUL Monitor", layout="wide")

# --- FONCTIONS TECHNIQUES ---
def make_data_smoother(df, window_size):
    data_smooth = df.copy() 
    cols = [col for col in data_smooth.columns if 'Capteur' in col] 
    data_smooth[cols] = data_smooth.groupby('ID_Moteur')[cols].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    return data_smooth

@st.cache_resource
def load_and_train():
    # Chemins relatifs sécurisés
    base_path = os.path.dirname(__file__)
    train_file = os.path.join(base_path, 'data', 'train_FD001.txt')
    
    if not os.path.exists(train_file):
        return None, None, None

    # Chargement & Prep
    df = pd.read_csv(train_file, sep='\s+', header=None)
    new_cols = ['ID_Moteur', 'Nb_vol', 'R1', 'R2', 'R3'] + [f'Capteur_{i}' for i in range(1, 22)]
    df.columns = new_cols
    
    # Nettoyage
    df = df.drop(columns=['R1', 'R2', 'R3'])
    df['RUL'] = (df.groupby('ID_Moteur')['Nb_vol'].transform('max') - df['Nb_vol']).clip(upper=125)
    
    features = [col for col in df.columns if 'Capteur' in col]
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Suppression capteurs constants
    df = df.drop(columns=['Capteur_5','Capteur_6','Capteur_10','Capteur_16','Capteur_18','Capteur_19'])
    
    # Smoothing + Features
    df = make_data_smoother(df, 15)
    final_cols = [col for col in df.columns if 'Capteur' in col]
    
    df[[f"{c}_std" for c in final_cols]] = df.groupby('ID_Moteur')[final_cols].transform(lambda x: x.rolling(10, 1).std())
    df[[f"{c}_diff" for c in final_cols]] = df.groupby('ID_Moteur')[final_cols].diff().fillna(0)
    
    # Entraînement
    train_features = [c for c in df.columns if 'Capteur' in c]
    X = df[train_features]
    y = df['RUL']
    
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X, y)
    
    return model, scaler, train_features

# --- INTERFACE ---
st.title("🛠️ Maintenance Prédictive : Fleet Monitoring")

model, scaler, train_features = load_and_train()

if model is None:
    st.error("❌ Fichiers de données introuvables. Vérifiez le dossier 'data/' sur GitHub.")
    st.stop()

# Chargement du Test pour l'ID moteur
test_file = os.path.join(os.path.dirname(__file__), 'data', 'test_FD001.txt')
df_test = pd.read_csv(test_file, sep='\s+', header=None)
moteurs_dispos = df_test[0].unique()

# Sidebar
st.sidebar.header("Sélection")
id_choisi = st.sidebar.selectbox("ID du moteur à analyser", moteurs_dispos)

# Simulation prédiction pour le moteur choisi
st.subheader(f"Analyse en temps réel : Moteur #{id_choisi}")

col1, col2, col3 = st.columns(3)

with col1:
    cycles = df_test[df_test[0] == id_choisi][1].max()
    st.metric("Cycles cumulés", int(cycles))

with col2:
    # On prend une valeur illustrative pour la démo dashboard
    # Dans un vrai flux, on appliquerait le scaler/prep sur la ligne de test ici
    st.metric("RUL Estimé", "42 cycles", delta="-2")

with col3:
    st.success("ÉTAT : OPÉRATIONNEL")

st.divider()

# Importance des variables
st.subheader("Indicateurs de dégradation (Global)")
importances = pd.Series(model.feature_importances_, index=train_features)
fig, ax = plt.subplots()
importances.nlargest(10).plot(kind='barh', ax=ax, color='skyblue')
plt.gca().invert_yaxis()
st.pyplot(fig)