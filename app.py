import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
# On importe tes outils depuis ton fichier RUL.py
from RUL import data_test_prep 

st.set_page_config(page_title="NASA Dashboard", layout="wide")
st.title("🚀 Maintenance Prédictive NASA")

@st.cache_resource
def load_resources():
    model = joblib.load('model_RUL.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features_list.pkl')
    return model, scaler, features

model, Myscaler, train_columns = load_resources()

# Interface utilisateur
engine_id = st.sidebar.number_input("ID Moteur", 1, 100, 1)

if st.button("Lancer le diagnostic"):
    # On utilise ta fonction de RUL.py !
    data_test = data_test_prep('data/test_FD001.txt', Myscaler)
    
    # Extraction et prédiction
    engine_data = data_test[data_test['ID_Moteur'] == engine_id]
    last_vols = engine_data[train_columns].tail(1)
    prediction = model.predict(last_vols)[0]
    
    st.metric("RUL Estimé", f"{int(prediction)} cycles")