# CordiaAI: folder review and how to raise accuracy

Reviewed 2026-10-23, read-only. Nothing in your folder was changed. The accuracy tests below ran on a copy of `heart_disease_raw.csv` in a separate sandbox.

## What the folder is

A final-year major project: **CordiaAI**, a multimodal heart-disease risk prediction system. It has two halves.

**Academic material (folder root).** Synopsis and Phase 1 reports (`synopsis.docx`, `phase1(final)report.pdf`, `synfinal report.pdf`), the methodology (`CordiaAI_Methodology.docx`), slide decks (`CordiaAI_Synopsis.pptx`, `CordiaAI_Phase2.pptx`, `phase1_final.pptx`), a viva prep guide (`CordiaAI_Defense_Guide.html/.pdf`), a conference paper template, 8 reference papers plus `related_work_papers/`, and figure scripts.

**Code (`cordia_ai/`).**

| Part | What it does | State |
|---|---|---|
| Tabular branch | FT-Transformer on UCI Heart Disease (920 rows, 5 severity classes) | Done. 53.3% acc. XGBoost reaches 58.7% |
| Augmentation | SMOTE vs CTGAN vs diffusion. Diffusion picked | Done, but it *lowers* 5-class accuracy (58.7 to 55.4) |
| ECG branch | ViT-B/16 on PTB-XL spectrograms (trained on Colab) | Done. 34.7% acc after class rebalancing |
| Fusion | Cross-attention of tabular and ECG embeddings | Architecture only. UCI and PTB-XL rows are paired at random, so it learns nothing real |
| Federated + DP | FedAvg + Opacus over 3 simulated hospitals | Done. 56.5% acc, only predicts 2 classes |
| Fairness, Grad-CAM | AIF360 audit, ECG heatmaps | Done |
| Digital twin | Framingham / Synthea risk projection, plus a React/Three.js 3D heart | Done |
| Web app | FastAPI + SQLite, doctor/patient roles, SHAP/LIME/DiCE reports, offline Llama summary, ECG upload | Done |
| Docker | Dockerfile + compose | Written, never run |
| Echo branch | EchoNet-Dynamic | Not started (`datasets/echo/` is empty) |

`cordia_ai/README.md` is detailed and honest about limitations; it is the best single description of the project.

## Problems and notable issues

1. **Cholesterol bug is still unfixed.** 172 of 920 records store cholesterol as 0 (meaning "missing"). The model learned "low cholesterol means disease", so in the app raising cholesterol lowers risk. The README documents it but the fix was never applied.
2. **The most predictive features are thrown away.** Feature selection keeps 10 columns and drops `slope`, `ca` and `thal`. They are mostly missing outside Cleveland, but they still help (see below).
3. **The Hungarian data has no severity labels.** All 294 Hungarian patients are labelled only 0 or 1, never 2 to 4. For 5-class severity that is label noise: a severe Hungarian patient is labelled "Mild".
4. **Results come from one 80/20 split (184 test patients).** Re-running the same XGBoost on 20 different random splits gave anywhere from **52.2% to 60.9%**. Differences of a few points between models are mostly luck of the split.
5. **Fusion result is not a real multimodal result** (random pairing). Present it as an architecture test only.
6. **App issues for a demo:** logins are kept in memory, so every server restart logs everyone out. CORS allows any origin. Docker was never tested.
7. **Housekeeping:** the two ECG checkpoints are 346 MB each inside OneDrive. The root `README.md` is just a title. `.claude/launch.json` points at temp test scripts that no longer exist. Leftovers such as `unpacked/`, `conference-template-a4.backup.docx`, `iks.js`, `syn.js` and duplicate report PDFs clutter the root.

## Raising the ~58% accuracy

Measured with XGBoost, 5-fold cross-validation repeated 5 times on all 920 patients:

| Change | 5-class acc | Binary acc |
|---|---|---|
| Current 10 features | 56.6% | 80.6% |
| + treat chol = 0 as missing | 56.2% | 80.6% |
| + add slope, ca, thal | **58.7%** | **82.3%** |
| + missing-value flags + data-source column | **59.0%** | 82.1% |
| 3 classes (none / mild / moderate+) | **66.8%** | n/a |
| Cleveland only (303 rows) | 55.9% | 80.7% |

### What the numbers say

On this dataset, 5-class severity tops out at about **58 to 60%**. Published multi-class results on UCI land in the same range, so your 58% is not a sign that something is broken. Tuning will not take it to 70%; the labels themselves are the limit (point 3 above).

### Recommended, in order of impact

1. **Change what you report.** Make the headline **disease vs no disease** (binary). Your own `benchmark_xgboost.py` already gets 86.4% / AUC 0.91 on your split, and it is the number every comparable paper reports. Then present severity as a second step, which `risk_reading.py` already does.
2. **If you keep severity, use 3 classes.** Merge Moderate/Severe/Critical (classes 2 to 4 have only 33 to 107 patients each). That moves accuracy from about 59% to about **67%** with no other change.
3. **Put `slope`, `ca`, `thal` back in**, using XGBoost's native missing-value handling (or median + an "is missing" flag). That's about +2 points on both tasks.
4. **Fix the cholesterol zeros** (and `trestbps = 0`). Accuracy barely moves, but it fixes the wrong direction of the cholesterol effect, which an examiner using the twin would notice.
5. **Switch the main model to XGBoost/LightGBM/CatBoost.** Tree models beat the FT-Transformer here by about 5 points; keep the transformer as a comparison. Tune with Optuna under cross-validation, expecting about +1 to 2 points.
6. **Handle the Hungarian labels.** Either train severity only on the three sources that have real severity labels, or use Hungarian for the binary step only.
7. **Drop diffusion augmentation for training** (it hurt 5-class accuracy) or keep it only as a compared experiment. Class weights or an ordinal model (severity is ordered 0 < 1 < 2 < 3 < 4) are better ways to handle imbalance.
8. **Report cross-validated mean ± std** instead of one split, for example "58.7 ± 2.8%". That is more defensible in a viva than a single number.
9. **A soft-voting ensemble** of XGBoost + Random Forest + logistic regression usually adds another 1 to 2 points.

**A warning about 95%+ numbers:** many papers on the Kaggle "heart.csv" (1,025 rows) get 95 to 100% because that file is the 303 Cleveland rows duplicated about 3 times, so test rows leak into training. Don't chase those numbers by switching to that file.

**Realistic targets:** binary 84 to 87%, 3-class 66 to 68%, 5-class 59 to 61%.
