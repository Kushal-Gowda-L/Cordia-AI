# Training data for the multi-source model

Put these files here (as .csv, or the original Kaggle .csv.zip download; subfolders are fine).
They are not committed. `train_multisource.py` skips any that are missing.

| File | Source |
|---|---|
| `cardio_train.csv` | Kaggle "Cardiovascular Disease dataset" (S. Ulianova) |
| `heart_disease_health_indicators_BRFSS2015.csv` | Kaggle "Heart Disease Health Indicators Dataset" (CDC BRFSS 2015) |
| `heart_2020_cleaned.csv` | Kaggle "Personal Key Indicators of Heart Disease" (CDC BRFSS 2020) |
| `heart_attack_prediction_dataset.csv` | Kaggle "Heart Attack Risk Prediction Dataset" |
| `ecg.csv` | ECG5000 heartbeats, 140 samples + label (1 = normal) per row |

UCI Heart Disease and Framingham are read from `datasets/uci/` and `datasets/framingham/`.
