import pandas as pd
import numpy as np
import yaml

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

clip_quantiles = params['prepare']['clip_quantiles']
iqr_passes = params['prepare']['iqr_passes']

df = pd.read_csv('data/raw/flooddata.csv')

column_name = 'Flood?'
df[column_name] = df[column_name].replace(r'^\s*$', 0, regex=True)
df[column_name] = df[column_name].fillna(0)
df[column_name] = df[column_name].astype(int)

cols_to_clip = ['Max_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed', 'ALT']
for col in cols_to_clip:
    lower = df[col].quantile(clip_quantiles[0])
    upper = df[col].quantile(clip_quantiles[1])
    df[col] = df[col].clip(lower, upper)

for i in range(iqr_passes):
    for col in cols_to_clip:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

cols_to_clip_2 = ['Rainfall', 'Bright_Sunshine', 'Wind_Speed']
for col in cols_to_clip_2:
    lower = df[col].quantile(clip_quantiles[0])
    upper = df[col].quantile(clip_quantiles[1])
    df[col] = df[col].clip(lower, upper)

df['Rainfall log'] = np.log1p(df['Rainfall'])
df['Wind_Speed log'] = np.log1p(df['Wind_Speed'])
df['ALT log'] = np.log1p(df['ALT'])

df = df.drop(columns=['Station_Number'])

df.to_csv('data/processed/flood_data_preprocessed.csv', index=False)
print(f"Preprocessing complete. Final dataset: {len(df)} rows")