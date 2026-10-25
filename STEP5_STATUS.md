# Step 5 (ECG branch) — resolved

**Goal:** fix `HYP` class F1=0.00 in the PTB-XL/ViT ECG branch, caused by severe
class imbalance (only 535/16244 training rows). First (unweighted) run scored test
accuracy 62.1% / macro F1 0.382, but `HYP` scored 0.00 precision/recall/F1 — the
model never predicted it.

**What fixed it:** two changes together, not class-weighted loss alone —

1. **Capped every majority class at 500 training examples** (HYP kept all ~535,
   so it's no longer a minority) instead of the original 2000-per-class cap.
   This was also a pragmatic response to Colab's free-tier runtime disconnecting
   mid-run repeatedly (5-6 times) whenever a real-world time gap fell between
   working sessions — a smaller job fits in one uninterrupted stretch.
2. **Class-weighted `CrossEntropyLoss`** on top, for the residual imbalance.

Reduced scope also meant EPOCHS 5 → 3.

**Result (test set, 271 samples):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| NORM | 0.35 | 0.55 | 0.42 | 51 |
| MI | 0.30 | 0.30 | 0.30 | 56 |
| STTC | 0.36 | 0.39 | 0.38 | 54 |
| CD | 0.36 | 0.22 | 0.27 | 55 |
| **HYP** | **0.38** | **0.29** | **0.33** | 55 |

Accuracy 0.35, macro F1 0.34, weighted F1 0.34 (val macro F1 peaked at 0.411,
epoch 3). This is *lower* overall accuracy than the unweighted run's 62.1%, but
that number was inflated by the model collapsing onto majority classes — HYP
going from 0.00 to 0.33 F1, with all five classes now landing in a similar
0.27-0.42 F1 band, is the actual fix: a model that predicts something for every
class instead of ignoring the rare one. More training data and epochs (the
original 2000-per-class, 5-epoch plan) would likely lift the overall numbers
further; this reduced-scope run was chosen to guarantee completion within one
Colab session rather than risk another lost run.

**Artifacts replaced:** `checkpoints/ecg_vit.pt` (previous unweighted version
kept as `checkpoints/ecg_vit_unweighted_backup.pt`),
`datasets/ecg-images/ecg_embeddings_{train,val,test}.csv`.

**For posterity — why this took several attempts:** Colab's free-tier runtime
disconnected/recycled mid-run repeatedly whenever a real-world time gap fell
between work sessions (driven by the user's own usage-limit resets), wiping
`/content` each time since nothing there persists. One earlier attempt reached
val F1 0.408 but was lost before `files.download()` could fire. The eventual
fix: run the whole pipeline (download → train → zip → download) as background
processes within a single continuous session, and reduce scope so that session
comfortably finishes before any gap could occur.
