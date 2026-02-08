import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="Maintenance Prédictive NASA", layout="wide")

# --- CHARGEMENT DES DONNÉES ET DU MODÈLE ---
@st.cache_resource # Pour ne pas recharger à chaque clic
def load_assets():
    model = joblib.load('model_RUL.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features_list.pkl')
    return model, scaler, features

model, scaler, train_columns = load_assets()

# Titre principal
st.title("🛠️ Dashboard de Maintenance Prédictive (C-MAPSS)")
st.markdown("Ce dashboard prédit la durée de vie restante (**RUL**) des moteurs en temps réel.")

# --- BARRE LATÉRALE ---
st.sidebar.header("Paramètres")
# Simulation : on charge le test pour l'interface
# (Dans un vrai cas, tu chargerais les données capteurs en direct)
data_test = pd.read_csv('data/test_FD001.txt', sep='\s+', header=None) # Simplifié pour l'exemple
# ... Applique tes fonctions de prep ici ...

id_moteur = st.sidebar.selectbox("Sélectionner l'ID du Moteur", range(1, 101))

# --- CALCULS ---
data_moteur = data_test[data_test[0] == id_moteur] # Colonne 0 = ID_Moteur
# Ici on ferait la prédiction sur la dernière ligne
# pred_rul = model.predict(derniere_ligne_prep)

# --- AFFICHAGE ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Cycles effectués", len(data_moteur))

with col2:
    # Exemple de valeur (à remplacer par ta variable final_predictions)
    val_pred = 45 
    st.metric("RUL Estimé", f"{val_pred} cycles", delta="-5 cycles", delta_color="inverse")

with col3:
    status = "Sain" if val_pred > 30 else "Critique"
    st.write(f"État : **{status}**")
    if status == "Critique":
        st.error("🚨 MAINTENANCE REQUISE")
    else:
        st.success("✅ FONCTIONNEMENT NORMAL")

# --- GRAPHIQUES ---
st.divider()
st.subheader("Analyse des Capteurs")

capteur_select = st.selectbox("Visualiser un capteur", train_columns)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data_moteur.index, data_moteur[10]) # Exemple index vs valeur capteur
ax.set_xlabel("Temps (Cycles)")
ax.set_ylabel("Valeur Scalée")
st.pyplot(fig)

st.sidebar.info(f"Modèle : RandomForest\nPrécision (R²) : 0.79")