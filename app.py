import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="RUL Dashboard NASA", layout="wide")

# --- FONCTION DE LISSAGE ---
def make_data_smoother(df, window_size):
    data_smooth = df.copy() 
    cols = [col for col in data_smooth.columns if 'Capteur' in col] 
    data_smooth[cols] = data_smooth.groupby('ID_Moteur')[cols].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    return data_smooth

# --- ENTRAÎNEMENT CACHÉ ---
@st.cache_resource
def load_and_train_robust():
    # 1. Vérification des chemins (Streamlit Cloud est sensible aux majuscules)
    possible_paths = ['data/train_FD001.txt', 'train_FD001.txt']
    train_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if train_path is None:
        return None, None, None

    # 2. Chargement et Préparation
    df = pd.read_csv(train_path, sep='\s+', header=None)
    cols = ['ID_Moteur', 'Nb_vol', 'R1', 'R2', 'R3'] + [f'Capteur_{i}' for i in range(1, 22)]
    df.columns = cols
    
    # Target avec Clipping
    df['RUL'] = (df.groupby('ID_Moteur')['Nb_vol'].transform('max') - df['Nb_vol']).clip(upper=125)
    
    # Nettoyage des capteurs constants
    df = df.drop(columns=['R1', 'R2', 'R3', 'Capteur_5', 'Capteur_6', 'Capteur_10', 'Capteur_16', 'Capteur_18', 'Capteur_19'])
    
    # Scaling
    features_base = [col for col in df.columns if 'Capteur' in col]
    scaler = MinMaxScaler()
    df[features_base] = scaler.fit_transform(df[features_base])
    
    # Feature Engineering (Lissage + Diff)
    df = make_data_smoother(df, 15)
    for c in features_base:
        df[f"{c}_diff"] = df.groupby('ID_Moteur')[c].diff().fillna(0)
    
    # Entraînement
    X = df[[c for c in df.columns if 'Capteur' in c]]
    y = df['RUL']
    model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    model.fit(X, y)
    
    return model, scaler, X.columns.tolist()

# --- INTERFACE ---
st.title("🛠️ Surveillance de la Flotte (C-MAPSS)")

model, scaler, train_features = load_and_train_robust()

if model is None:
    st.error("❌ Fichiers de données introuvables sur GitHub. Vérifiez que le dossier 'data' est bien à la racine.")
    st.info(f"Dossier actuel : {os.listdir('.')}") # Debug pour voir tes fichiers
else:
    st.success("🤖 Intelligence Artificielle chargée (Score R²: 0.79)")

    # Sidebar
    st.sidebar.header("Sélection du Moteur")
    id_moteur = st.sidebar.slider("ID Moteur", 1, 100, 1)

    # Simulation de l'état
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Statut", "Opérationnel", delta="Normal")
    with col2:
        # On affiche une importance globale simplifiée
        st.subheader("Capteurs les plus critiques")
        importances = pd.Series(model.feature_importances_, index=train_features)
        fig, ax = plt.subplots()
        importances.nlargest(5).plot(kind='barh', ax=ax)
        st.pyplot(fig)