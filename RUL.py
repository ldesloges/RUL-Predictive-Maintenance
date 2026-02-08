import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

def make_data_smoother(df,window_size):
        data_smooth=df.copy() 
        columns=[col for col in data_smooth.columns if 'Capteur' in col] 
        data_smooth[columns]=data_smooth.groupby('ID_Moteur')[columns].transform(lambda x: x.rolling(window=window_size ,min_periods=1).mean())
        return data_smooth

def data_train_prep(fichier):
    df=pd.read_csv(fichier, sep='\s+',header=None)
    new_cols = []
    for i, col in enumerate(df.columns):
        if i == 0:
            new_cols.append('ID_Moteur')
        elif i == 1:
            new_cols.append('Nb_vol')
        elif i <= 5:
            new_cols.append(f'Reglage_{i-1}')
        else:
            new_cols.append(f'Capteur_{i-4}')
    df.columns = new_cols
    cols_to_drop = ['Reglage_1', 'Reglage_2', 'Reglage_3','Reglage_4']
    df = df.drop(columns=cols_to_drop)

    max_cycle=df.groupby('ID_Moteur')['Nb_vol'].transform('max')

    df['RUL'] = max_cycle - df['Nb_vol']
    
    # 2. LE CLIPPING (La solution magique)
    df['RUL'] = df['RUL'].clip(upper=125)

    features = [col for col in df.columns if 'Capteur' in col]

    scaler = MinMaxScaler()

    df[features] = scaler.fit_transform(df[features])

    df=df.drop(columns=['Capteur_5','Capteur_6','Capteur_10','Capteur_16','Capteur_18','Capteur_19'])


    data_smooth=make_data_smoother(df,15)

    columns=[col for col in data_smooth if 'Capteur' in col]
    std_columns = [f"{col}_std" for col in columns]
    diff_columns = [f"{col}_diff" for col in columns]


    data_smooth[std_columns]=data_smooth.groupby('ID_Moteur')[columns].transform(lambda x: x.rolling(window=10,min_periods=1).std()).fillna(0)
    data_smooth[diff_columns]=data_smooth.groupby('ID_Moteur')[columns].diff().fillna(0)

    return data_smooth,scaler

data_train,Myscaler=data_train_prep('data/train_FD001.txt')

def data_test_prep(fichier,scaler):
    df=pd.read_csv(fichier, sep='\s+',header=None)
    new_cols = []
    for i, col in enumerate(df.columns):
        if i == 0:
            new_cols.append('ID_Moteur')
        elif i == 1:
            new_cols.append('Nb_vol')
        elif i <= 5:
            new_cols.append(f'Reglage_{i-1}')
        else:
            new_cols.append(f'Capteur_{i-4}')
    df.columns = new_cols
    cols_to_drop = ['Reglage_1', 'Reglage_2', 'Reglage_3','Reglage_4']
    df = df.drop(columns=cols_to_drop)

    features = [col for col in df.columns if 'Capteur' in col]


    df[features] = scaler.transform(df[features])

    df=df.drop(columns=['Capteur_5','Capteur_6','Capteur_10','Capteur_16','Capteur_18','Capteur_19'])

    data_smooth=make_data_smoother(df,15)

    columns=[col for col in data_smooth if 'Capteur' in col]
    std_columns = [f"{col}_std" for col in columns]
    diff_columns = [f"{col}_diff" for col in columns]


    data_smooth[std_columns]=data_smooth.groupby('ID_Moteur')[columns].transform(lambda x: x.rolling(window=10,min_periods=1).std()).fillna(0)
    data_smooth[diff_columns]=data_smooth.groupby('ID_Moteur')[columns].diff().fillna(0)

    return data_smooth


#Train
data_test=data_test_prep('data/test_FD001.txt',Myscaler)
df_RUL=pd.read_csv('data/RUL_FD001.txt',header=None)
df_RUL.columns=['true_RUL']


columns=[col for col in data_train if 'Capteur' in col]

X=data_train[columns]
y=data_train['RUL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
print("Entraînement de l'IA en cours...")
model.fit(X_train, y_train)

X_test=data_test[columns]
predictions = model.predict(X_test)

# On trouve l'index de la ligne où le Nb_vol est maximum pour chaque moteur
last_indices = data_test.groupby('ID_Moteur')['Nb_vol'].idxmax()
# On transforme les prédictions en Series avec le même index que data_test
preds_series = pd.Series(predictions, index=data_test.index)
# On extrait uniquement les prédictions correspondant aux derniers vols
final_predictions = preds_series.loc[last_indices]

mae = mean_absolute_error(df_RUL['true_RUL'], final_predictions)
r2 = r2_score(df_RUL['true_RUL'], final_predictions)
print(f"Score R² : {r2:.2f}")

plt.scatter(df_RUL['true_RUL'],final_predictions, alpha=0.5)
plt.xlabel('true RUL')
plt.ylabel('prédiction du RUL')





# 1. Récupérer les scores d'importance
importances = model.feature_importances_

# 2. Associer les scores aux noms des colonnes
feature_importance_df = pd.DataFrame({
    'Variable':columns,
    'Importance': importances
})

# 3. Trier par importance décroissante
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 4. Afficher le top 10 dans la console
print("Top 10 des colonnes qui font pencher la balance :")
print(feature_importance_df.head(10))

# 5. Bonus : Le voir en graphique
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Variable'].head(10), feature_importance_df['Importance'].head(10))
plt.gca().invert_yaxis() # Pour avoir la plus importante en haut
plt.title("Qu'est-ce qui cause la panne selon l'IA ?")
plt.xlabel("Niveau d'importance")
plt.show()


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.title("🛠️ Tableau de Bord : Maintenance Prédictive")

# Barre latérale pour charger les données ou choisir le moteur
st.sidebar.header("Configuration")
id_moteur = st.sidebar.selectbox("Choisir un moteur", data_test['ID_Moteur'].unique())

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"État du Moteur {id_moteur}")
    # On récupère la dernière prédiction pour ce moteur
    derniere_pred = final_predictions[id_moteur - 1] # Exemple si trié
    st.metric(label="Vols restants estimés (RUL)", value=f"{int(derniere_pred)} cycles")
    
    if derniere_pred < 30:
        st.error("⚠️ ALERTE : Maintenance urgente requise !")
    else:
        st.success("✅ Moteur en bon état")

with col2:
    st.subheader("Variables Clés")
    # Affichage de ton graphique d'importance des variables
    fig, ax = plt.subplots()
    feat_importances.nlargest(5).plot(kind='barh', ax=ax)
    st.pyplot(fig)

# Graphique temporel des capteurs
st.divider()
st.subheader("Historique des capteurs")
capteur_choisi = st.selectbox("Choisir un capteur à surveiller", train_columns[:5])
data_moteur = data_test[data_test['ID_Moteur'] == id_moteur]
st.line_chart(data_moteur[capteur_choisi])