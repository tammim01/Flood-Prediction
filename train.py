import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

df = pd.read_csv('data/processed/flood_data_preprocessed.csv')

numeric_features = ['Max_Temp', 'Min_Temp', 'Rainfall log', 'Relative_Humidity', 
                    'Wind_Speed log', 'Cloud_Coverage', 'Bright_Sunshine', 
                    'X_COR', 'Y_COR', 'LATITUDE', 'LONGITUDE', 'ALT log', 'Period']

df_dummies = pd.get_dummies(df[['Station_Names']], drop_first=True)

X = pd.concat([df[numeric_features], df_dummies], axis=1)
y = df['Flood?']

X_columns = X.columns.tolist()

test_size = params['train']['test_size']
random_state = params['train']['random_state']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[X_train.columns] = scaler.fit_transform(X_train)
X_test_scaled[X_test.columns] = scaler.transform(X_test)

smote = SMOTE(random_state=random_state)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

n_estimators = params['train']['n_estimators']
max_depth = params['train']['max_depth']

rf_model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=random_state,
    class_weight='balanced'
)

rf_model.fit(X_train_res, y_train_res)

test_data = X_test_scaled.copy()
test_data['Flood?'] = y_test.values
test_data.to_csv('data/processed/test.csv', index=False)

joblib.dump(rf_model, 'models/flood_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(X_columns, 'models/model_features.pkl')

print(f"Training complete. Model saved to models/")
print(f"Training samples: {len(X_train_res)}, Test samples: {len(X_test)}")