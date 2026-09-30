# Flood Prediction with DVC

## Problem and Dataset

This project predicts floods in Bangladesh using monthly weather station data from 32 stations. The task is a binary classification problem where the target is `Flood?` (0 = no flood, 1 = flood).

### Dataset Description
- **Total samples**: ~20,544 rows (monthly data from 1948-2013)
- **Features**: 18 weather and location features including Max_Temp, Min_Temp, Rainfall, Relative_Humidity, Wind_Speed, Cloud_Coverage, Bright_Sunshine, geographic coordinates, and elevation
- **Target**: Flood? (binary, with significant class imbalance - only ~22% positive cases)

### Preprocessing Steps
1. Replace blank/whitespace/nan in `Flood?` column with 0
2. Clip at 1st/99th percentile for temperature, rainfall, humidity, wind, and altitude
3. Apply IQR-based outlier removal (1.5×IQR) for 5 passes
4. Create log-transformed features for Rainfall, Wind_Speed, and ALT
5. Drop Station_Number column

**Final dataset**: 17,560 rows after preprocessing

## ML Model

### Random Forest with SMOTE
The model uses a Random Forest classifier with:
- **SMOTE** (Synthetic Minority Over-sampling Technique) for handling class imbalance
- **Class weight balancing** as an additional safeguard
- **100 features** after one-hot encoding Station_Names (31 new columns)

### Training Configuration
- 80/20 stratified train/test split
- StandardScaler fitted on training data only
- 200 estimators, no max depth limit
- Random state 42 for reproducibility

### Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9944 |
| Precision | 0.9059 |
| Recall | 0.9472 |
| F1 Score | 0.9261 |

### Model Comparison (from notebooks)

| Model | ROC-AUC | Accuracy |
|-------|---------|----------|
| Random Forest | 0.9946 | 0.98 |
| Logistic Regression | ~0.95 | ~0.95 |
| XGBoost | ~0.97 | ~0.97 |

Random Forest performed best on this dataset.

## Project Structure

```
flood-prediction/
├── data/
│   ├── raw/
│   │   └── flooddata.csv          # Raw data (tracked by DVC)
│   └── processed/
│       ├── flood_data_preprocessed.csv  # Preprocessed data
│       └── test.csv               # Test set with features
├── src/
│   ├── prepare.py                 # Data preprocessing
│   ├── train.py                   # Model training
│   └── evaluate.py                # Model evaluation
├── models/
│   ├── flood_model.pkl            # Trained Random Forest model
│   ├── scaler.pkl                 # Fitted StandardScaler
│   └── model_features.pkl         # List of feature names
├── metrics/
│   └── metrics.json               # Evaluation metrics
├── reports/
│   └── confusion_matrix.png       # Confusion matrix visualization
├── notebooks/
│   ├── final_preprocessing.ipynb  # Preprocessing notebook
│   ├── final_trainRF.ipynb        # Random Forest notebook
│   ├── final_trainLR.ipynb        # Logistic Regression notebook
│   └── XgBoost.ipynb              # XGBoost comparison
├── dvc.yaml                       # DVC pipeline definition
├── params.yaml                    # Pipeline parameters
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

## DVC Pipeline

The pipeline consists of three stages:

1. **prepare**: Preprocesses raw flood data
2. **train**: Trains the Random Forest model with SMOTE
3. **evaluate**: Evaluates the model and generates metrics

```
dvc dag output:
+----------------------------+ 
| data\raw\flooddata.csv.dvc | 
+----------------------------+ 
               *               
               *               
               *               
          +---------+          
          | prepare |          
          +---------+          
               *               
               *               
               *               
          +-------+            
          | train |            
          +-------+            
               *               
               *               
               *               
         +----------+          
         | evaluate |          
         +----------+          
```

### Parameters (params.yaml)
- `prepare.clip_quantiles`: [0.01, 0.99] - quantiles for clipping outliers
- `prepare.iqr_passes`: 5 - number of IQR-based outlier removal passes
- `train.test_size`: 0.2 - test set proportion
- `train.random_state`: 42 - random seed
- `train.n_estimators`: 200 - number of trees
- `train.max_depth`: null - no max depth limit
- `evaluate.threshold`: 0.5 - classification threshold

## How to Run

```bash
# Clone the repository
git clone <repo-url>
cd flood-prediction

# Set up Python (ensure Python 3.14 is installed with py launcher)
pip install -r requirements.txt

# Pull the tracked data from DVC remote
dvc pull

# Or run the full pipeline
dvc repro

# Check pipeline status
dvc status

# Visualize pipeline
dvc dag

# Push to remote storage
dvc push
```

### DVC Commands Cheat Sheet
- `git clone <repo>` - Clone repository
- `pip install -r requirements.txt` - Install dependencies
- `dvc pull` - Download tracked data files
- `dvc repro` - Run the entire pipeline
- `dvc status` - Check for changes
- `dvc dag` - Visualize pipeline graph
- `dvc push` - Upload data to remote storage
- `dvc metrics show` - Display metrics.json

## Results

### Confusion Matrix
![Confusion Matrix](reports/confusion_matrix.png)

### Performance Summary
The model achieves excellent performance with:
- **98% accuracy** on the test set
- **ROC-AUC of 0.9944** indicating strong discriminative ability
- **91% precision** and **95% recall** for flood prediction

The high recall is particularly important for flood prediction as we want to catch most flood events (minimize false negatives), while the high precision ensures we don't generate too many false alarms.

### Class Imbalance Handling
The original dataset had only ~22% flood events. SMOTE was applied after train/test split to synthetically generate more flood samples in the training set, while keeping the test set distribution realistic.