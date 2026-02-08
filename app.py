import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="RUL Engine Dashboard", layout="wide")

# --- TES FONCTIONS DE PREPARATION ---
def make_data_smoother(df, window_size):
    data_smooth = df.copy() 
    columns = [col for col in data_smooth.columns if 'Capteur' in col] 
    data_smooth[columns] = data_smooth.groupby('ID_Moteur')[columns].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean())
    return data_smooth

@st.cache_resource # On entraîne une seule fois et on garde en mémoire
def train_model():
    # Chargement train
    df = pd.read_csv('data/train_FD001.txt', sep='\s+', header=None)
    new_cols = ['ID_Moteur', 'Nb_vol', 'R1', 'R2', 'R3'] + [f'Capteur_{i}' for i in range(1, 22)]
    df.columns = new_cols
    df = df.drop(columns=['R1', 'R2', 'R3'])
    
    # RUL + Clipping
    df['RUL'] = (df.groupby('ID_Moteur')['Nb_vol'].transform('max') - df['Nb_vol']).clip(upper=125)
    
    # Scaling
    features = [col for col in df.columns if 'Capteur' in col]
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Drop colonnes constantes
    df = df.drop(columns=['Capteur_5','Capteur_6','Capteur_10','Capteur_16','Capteur_18','Capteur_19'])
    
    # Feature engineering
    df = make_data_smoother(df, 15)
    cols_sensors = [col for col in df if 'Capteur' in col]
    df[[f"{c}_std" for c in cols_sensors]] = df.groupby('ID_Moteur')[cols_sensors].transform(lambda x: x.rolling(10,1).std()).fillna(0)
    df[[f"{c}_diff" for c in cols_sensors]] = df.groupby('ID_Moteur')[cols_sensors].diff().fillna(0)
    
    # Training
    X = df[[c for c in df.columns if 'Capteur' in c]]
    y = df['RUL']
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X, y)
    
    return model, scaler, X.columns.tolist()

# --- CHARGEMENT DU TEST ---
@st.cache_data
def load_test_data(_scaler):
    df_test = pd.read_csv('data/test_FD001.txt', sep='\s+', header=None)
    new_cols = ['ID_Moteur', 'Nb_vol', 'R1', 'R2', 'R3'] + [f'Capteur_{i}' for i in range(1, 22)]
    df_test.columns = new_cols
    df_test = df_test.drop(columns=['R1', 'R2', 'R3'])
    
    features = [col for col in df_test.columns if 'Capteur' in col]
    df_test[features] = _scaler.transform(df_test[features])
    df_test = df_test.drop(columns=['Capteur_5','Capteur_6','Capteur_10','Capteur_16','Capteur_18','Capteur_19'])
    
    df_test = make_data_smoother(df_test, 15)
    cols_sensors = [col for col in df_test if 'Capteur' in col]
    df_test[[f"{c}_std" for c in cols_sensors]] = df_test.groupby('ID_Moteur')[cols_sensors].transform(lambda x: x.rolling(10,1).std()).fillna(0)
    df_test[[f"{c}_diff" for col in cols_sensors]] = df_test.groupby('ID_Moteur')[cols_sensors].diff().fillna(0)
    
    return df_test

# --- MAIN DASHBOARD ---
st.title("🛠️ Maintenance Prédictive NASA : Flotte FD001")

with st.spinner('Initialisation de l\'IA...'):
    model, scaler, train_features = train_model()
    data_test = load_test_data(scaler)

# Sidebar
id_list = data_test['ID_Moteur'].unique()
engine_id = st.sidebar.selectbox("Choisir l'ID du moteur", id_list)

# Prédiction
engine_data = data_test[data_test['ID_Moteur'] == engine_id]
X_input = engine_data[train_features].tail(1)
prediction = model.predict(X_input)[0]

# --- AFFICHAGE ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Vols effectués", int(engine_data['Nb_vol'].max()))
with col2:
    st.metric("RUL Prédit (Cycles)", int(prediction))
with col3:
    if prediction < 30:
        st.error("STATUT : CRITIQUE")
    else:
        st.success("STATUT : NORMAL")

st.divider()

# Graphiques
c1, c2 = st.columns(2)
with c1:
    st.subheader("Évolution des Capteurs")
    sensor = st.selectbox("Capteur", [c for c in train_features if 'std' not in c and 'diff' not in c])
    fig, ax = plt.subplots()
    ax.plot(engine_data['Nb_vol'], engine_data[sensor], color='orange')
    ax.set_xlabel("Cycles")
    st.pyplot(fig)

with c2:
    st.subheader("Importance des Variables")
    importances = pd.Series(model.feature_importances_, index=train_features)
    fig, ax = plt.subplots()
    importances.nlargest(10).plot(kind='barh', ax=ax)
    plt.gca().invert_yaxis()
    st.pyplot(fig)