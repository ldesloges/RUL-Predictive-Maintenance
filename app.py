import streamlit as st
import pandas as pd
import joblib
import os

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
    st.error("Les fichiers .pkl sont manquants sur GitHub.")
else:
    data_test = data_test_prep('data/test_FD001.txt', scaler)
    
    st.sidebar.header("🕹️ Menu de Contrôle")
    engine_ids = data_test['ID_Moteur'].unique()
    selected_id = st.sidebar.selectbox("Sélectionner l'ID du moteur", engine_ids)
    
    engine_data = data_test[data_test['ID_Moteur'] == selected_id]
    X_input = engine_data[features].tail(1)
    prediction = model.predict(X_input)[0]
    
    st.markdown(f"### 📊 État de santé du Moteur **#{selected_id}**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cycles effectués", int(engine_data['Nb_vol'].max()))
    
    with col2:
        st.metric("RUL Estimé", f"{int(prediction)} cycles", delta="-1 vol")
    
    with col3:
        if prediction < 30:
            st.error("🚨 STATUT : CRITIQUE")
        elif prediction < 60:
            st.warning("⚠️ STATUT : SURVEILLANCE")
        else:
            st.success("✅ STATUT : OPÉRATIONNEL")

    st.divider()

    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("📈 Évolution des paramètres capteurs")
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
        importances = pd.Series(model.feature_importances_, index=features)
        fig_imp, ax_imp = plt.subplots()
        importances.nlargest(10).plot(kind='barh', ax=ax_imp, color='#ff7f0e')
        ax_imp.invert_yaxis()
        st.pyplot(fig_imp)

    with st.expander("🔍 Voir les dernières mesures de télémétrie"):
        st.dataframe(engine_data.tail(10), use_container_width=True)