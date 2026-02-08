import streamlit as st
import pandas as pd
import joblib
import os
# On importe seulement les fonctions
from RUL import data_test_prep 

st.set_page_config(page_title="NASA Engine Dashboard", layout="wide")

@st.cache_resource
def load_assets():
    # On vérifie si les fichiers existent sur GitHub
    if os.path.exists('model_RUL.pkl'):
        return joblib.load('model_RUL.pkl'), joblib.load('scaler.pkl'), joblib.load('features_list.pkl')
    return None, None, None

model, scaler, features = load_assets()

if model is None:
    st.error("⚠️ Les fichiers .pkl sont manquants sur GitHub.")
    st.info("Lance 'python RUL.py' sur ton Mac, puis fais un 'make push'.")
else:
    st.title("🛠️ Surveillance de Flotte en Temps Réel")
    # ... la suite de ton code dashboard ...