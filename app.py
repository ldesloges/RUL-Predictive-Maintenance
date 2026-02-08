import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
# Import des fonctions depuis RUL.py
from RUL import data_test_prep 

st.set_page_config(page_title="Dashboard Maintenance NASA", layout="wide")

@st.cache_resource
def load_assets():
    # Charge les fichiers générés par RUL.py
    model = joblib.load('model_RUL.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features_list.pkl')
    return model, scaler, features

st.title("🛠️ Surveillance de Flotte en Temps Réel")

try:
    model, Myscaler, train_columns = load_assets()

    # Sidebar : Sélection du moteur
    df_raw = pd.read_csv('data/test_FD001.txt', sep='\s+', header=None)
    engine_id = st.sidebar.selectbox("Choisir l'ID du moteur", df_raw[0].unique())

    if st.sidebar.button("Lancer le Diagnostic"):
        # Utilisation de ta fonction de RUL.py
        data_test = data_test_prep('data/test_FD001.txt', Myscaler)
        
        # Données du moteur choisi
        engine_data = data_test[data_test['ID_Moteur'] == engine_id]
        X_input = engine_data[train_columns].tail(1)
        
        prediction = model.predict(X_input)[0]
        
        # Affichage
        st.metric("RUL Estimé (Cycles restants)", f"{int(prediction)}")
        
        if prediction < 30:
            st.error("🚨 ALERTE : Maintenance immédiate conseillée.")
        else:
            st.success("✅ État du moteur : Stable.")

except FileNotFoundError:
    st.error("Fichiers .pkl introuvables. Lancez 'python RUL.py' sur votre Mac d'abord.")