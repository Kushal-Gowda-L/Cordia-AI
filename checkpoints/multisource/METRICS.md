# Multi-source model: held-out test metrics

Each dataset's own 15% test split. Threshold for sensitivity/specificity chosen on validation (Youden). Probabilities Platt-calibrated on validation.

| Dataset | Outcome | Test n | Prevalence | FT-Transformer AUROC (95% CI) | AUPRC | Brier | ECE | Sens. | Spec. | LogReg AUROC | GBM AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uci | Coronary artery disease on angiography (>50% narrowing in any major vessel) | 138 | 0.551 | 0.914 (0.851-0.954) | 0.911 | 0.122 | 0.067 | 0.882 | 0.790 | 0.882 | 0.914 |
| framingham | Coronary heart disease within 10 years | 636 | 0.151 | 0.692 (0.623-0.748) | 0.356 | 0.118 | 0.016 | 0.594 | 0.670 | 0.699 | 0.682 |
| cardio | Cardiovascular disease present at examination | 10500 | 0.500 | 0.798 (0.790-0.807) | 0.784 | 0.182 | 0.009 | 0.711 | 0.761 | 0.787 | 0.799 |
| brfss2015 | Self-reported coronary heart disease or heart attack | 38052 | 0.094 | 0.849 (0.843-0.854) | 0.371 | 0.070 | 0.003 | 0.836 | 0.710 | 0.845 | 0.849 |
| key_indicators_2020 | Self-reported coronary heart disease or heart attack | 47970 | 0.086 | 0.840 (0.835-0.846) | 0.344 | 0.066 | 0.002 | 0.827 | 0.704 | 0.837 | 0.841 |
| heart_attack_risk | 'Heart Attack Risk' label | 1315 | 0.358 | 0.500 (0.470-0.538) | 0.364 | 0.230 | 0.002 | 0.159 | 0.836 | 0.496 | 0.513 |
| ecg5000 | Abnormal heartbeat | 750 | 0.416 | 0.995 (0.991-0.998) | 0.994 | 0.026 | 0.025 | 0.958 | 0.984 | - | - |
