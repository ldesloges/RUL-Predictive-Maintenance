import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
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


if __name__ == "__main__":
    # Ce code ne s'exécute QUE quand tu lances RUL.py manuellement
    print("🚀 Entraînement sur ton Mac...")
    data_train, Myscaler = data_train_prep('data/train_FD001.txt')
    cols_train = [col for col in data_train if 'Capteur' in col]
    
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(data_train[cols_train], data_train['RUL'])
    
    # On sauvegarde les cerveaux
    joblib.dump(model, 'model_RUL.pkl')
    joblib.dump(Myscaler, 'scaler.pkl')
    joblib.dump(cols_train, 'features_list.pkl')
    print("✅ Fichiers .pkl créés !")