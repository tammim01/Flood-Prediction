import pandas as pd
import numpy as np
import joblib
import json
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

model = joblib.load('models/flood_model.pkl')
test_data = pd.read_csv('data/processed/test.csv')

threshold = params['evaluate']['threshold']

y_test = test_data['Flood?']
feature_cols = [c for c in test_data.columns if c != 'Flood?']
X_test = test_data[feature_cols]

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= threshold).astype(int)

roc_auc = roc_auc_score(y_test, y_proba)

precision = (y_pred[y_test == 1] == 1).sum() / (y_pred == 1).sum() if (y_pred == 1).sum() > 0 else 0
recall = (y_pred[y_test == 1] == 1).sum() / (y_test == 1).sum()
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

metrics = {
    'roc_auc': float(roc_auc),
    'precision': float(precision),
    'recall': float(recall),
    'f1': float(f1)
}

with open('metrics/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Flood (0)', 'Flood (1)'], yticklabels=['No Flood (0)', 'Flood (1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Threshold={threshold})')
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png', dpi=150)
plt.close()

print(f"Metrics: ROC-AUC={roc_auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
print("Evaluation complete. Results saved to metrics/ and reports/")