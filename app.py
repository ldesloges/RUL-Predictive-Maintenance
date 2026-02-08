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
    # 1. Préparation des données de test
    # On utilise ta fonction importée de RUL.py pour traiter le fichier brut
    data_test = data_test_prep('data/test_FD001.txt', scaler)
    
    # 2. Barre latérale : Sélection du moteur
    st.sidebar.header("🕹️ Menu de Contrôle")
    engine_ids = data_test['ID_Moteur'].unique()
    selected_id = st.sidebar.selectbox("Sélectionner l'ID du moteur", engine_ids)
    
    # 3. Extraction et Prédiction
    # On récupère les données du moteur choisi et on prédit sur le dernier vol connu
    engine_data = data_test[data_test['ID_Moteur'] == selected_id]
    X_input = engine_data[features].tail(1)
    prediction = model.predict(X_input)[0]
    
    # --- SECTION 1 : CHIFFRES CLÉS (KPIs) ---
    st.markdown(f"### 📊 État de santé du Moteur **#{selected_id}**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cycles effectués", int(engine_data['Nb_vol'].max()))
    
    with col2:
        # RUL (Remaining Useful Life) prédit
        st.metric("RUL Estimé", f"{int(prediction)} cycles", delta="-1 vol")
    
    with col3:
        # Alerte visuelle dynamique
        if prediction < 30:
            st.error("🚨 STATUT : CRITIQUE")
        elif prediction < 60:
            st.warning("⚠️ STATUT : SURVEILLANCE")
        else:
            st.success("✅ STATUT : OPÉRATIONNEL")

    st.divider()

    # --- SECTION 2 : ANALYSE VISUELLE ---
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("📈 Évolution des paramètres capteurs")
        # On affiche le capteur le plus influent ou un choix de l'utilisateur
        sensor = st.selectbox("Choisir un capteur à visualiser", [f for f in features if 'std' not in f and 'diff' not in f])
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(engine_data['Nb_vol'], engine_data[sensor], color='#1f77b4', linewidth=2)
        ax.set_xlabel("Nombre de Vols")
        ax.set_ylabel("Valeur Normalisée")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with right_col:
        st.subheader("🧬 Facteurs de dégradation")
        # Importance des variables calculée par ton Random Forest
        importances = pd.Series(model.feature_importances_, index=features)
        fig_imp, ax_imp = plt.subplots()
        importances.nlargest(10).plot(kind='barh', ax=ax_imp, color='#ff7f0e')
        ax_imp.invert_yaxis()
        st.pyplot(fig_imp)

    # --- SECTION 3 : DONNÉES BRUTES ---
    with st.expander("🔍 Voir les dernières mesures de télémétrie"):
        st.dataframe(engine_data.tail(10), use_container_width=True)