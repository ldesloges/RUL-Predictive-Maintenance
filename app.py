import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Configuration
st.set_page_config(page_title="NASA Maintenance Dashboard", layout="wide")

# Chargement sécurisé
@st.cache_resource
def load_model_assets():
    model = joblib.load('model_RUL.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features_list.pkl')
    # On charge les résultats pré-calculés pour la démo
    final_preds = joblib.load('final_preds.pkl')
    return model, scaler, features, final_preds

try:
    model, scaler, train_columns, final_predictions = load_model_assets()
    
    # Titre
    st.title("🛠️ Fleet Monitoring : Maintenance Prédictive")
    st.sidebar.header("Options")
    
    # Sélection du moteur
    id_moteur = st.sidebar.selectbox("Choisir un moteur", range(1, 101))
    
    # Affichage des KPIs
    col1, col2 = st.columns(2)
    with col1:
        # On récupère la prédiction via l'index (ID 1 est à l'index 0)
        pred = final_predictions.iloc[id_moteur - 1]
        st.metric("RUL Estimé (Cycles)", f"{int(pred)}")
    
    with col2:
        if pred < 30:
            st.error("STATUT : MAINTENANCE CRITIQUE")
        else:
            st.success("STATUT : OPÉRATIONNEL")

    # Importance des variables
    st.subheader("Analyse de défaillance")
    importances = pd.Series(model.feature_importances_, index=train_columns)
    fig, ax = plt.subplots()
    importances.nlargest(10).plot(kind='barh', ax=ax)
    st.pyplot(fig)

except Exception as e:
    st.error(f"Erreur de chargement des fichiers : {e}")
    st.info("Vérifiez que les fichiers .pkl sont bien à la racine de votre dépôt GitHub.")