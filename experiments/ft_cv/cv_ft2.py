"""5-fold stratified CV (same folds as cv_ft.py) of the FT-Transformer with stacked improvements.
Usage: cv_ft2.py CFG   where CFG in
  C: 13 features + missing flags, d_token=128           (smaller model)
  D: C + class-weighted CrossEntropyLoss
  E: D + categorical tokens for sex/cp/fbs/restecg/exang/slope/ca/thal (missing = own category)
  F: E + 5-seed ensemble per fold (averaged probabilities)
  G: C + 5-seed ensemble per fold (no class weights, no categorical tokens)
Everything else matches the original: depth 3, 8 heads, dropout 0.2, AdamW lr 1e-4 wd 1e-4,
batch 32, early stopping on validation macro F1 with patience 40, max 300 epochs."""
import sys, json, time
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from ft_transformer import FTTransformer, FeatureTokenizer
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
cfg = sys.argv[1]
NUM = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
EXTRA = ["slope", "ca", "thal"]
BASE = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"]
D_TOKEN = 128
WEIGHTED = cfg in "DEF"
CATEGORICAL = cfg in "EF"
SEEDS = 5 if cfg in "FG" else 1


class CatFTTransformer(FTTransformer):
    """Same FT-Transformer, but categorical features get a learned embedding per value
    (index 0 = missing/unseen) instead of the x * W + b numerical tokenizer."""
    def __init__(self, n_num, cat_cards, **kw):
        super().__init__(n_features=n_num, **kw)
        d = kw["d_token"]
        self.cat_emb = nn.ModuleList(nn.Embedding(c, d) for c in cat_cards)
        self.n_num = n_num

    def embed(self, x):
        num, cat = x[:, :self.n_num], x[:, self.n_num:].long()
        tokens = [self.tokenizer(num)] + [e(cat[:, i]).unsqueeze(1) for i, e in enumerate(self.cat_emb)]
        tokens = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        return self.norm(self.encoder(torch.cat([cls, tokens], dim=1))[:, 0])


df = pd.read_csv("heart_disease_raw.csv")
y = df["num"].values


def prep(tr, te):
    """Fit imputation/scaling/category maps on the training fold only."""
    if not CATEGORICAL:
        cols = BASE + EXTRA
        Xtr, Xte = tr[cols].copy(), te[cols].copy()
        for c in EXTRA:
            Xtr[c + "_missing"] = Xtr[c].isna().astype(float)
            Xte[c + "_missing"] = Xte[c].isna().astype(float)
        med = Xtr.median()
        sc = StandardScaler().fit(Xtr.fillna(med))
        return sc.transform(Xtr.fillna(med)), sc.transform(Xte.fillna(med)), None
    med = tr[NUM].median()
    sc = StandardScaler().fit(tr[NUM].fillna(med))
    ntr, nte = sc.transform(tr[NUM].fillna(med)), sc.transform(te[NUM].fillna(med))
    ctr, cte, cards = [], [], []
    for c in CAT:
        vals = sorted(tr[c].dropna().unique())
        m = {v: i + 1 for i, v in enumerate(vals)}
        ctr.append(tr[c].map(m).fillna(0).values)
        cte.append(te[c].map(m).fillna(0).values)
        cards.append(len(vals) + 1)
    return np.hstack([ntr, np.array(ctr).T]), np.hstack([nte, np.array(cte).T]), cards


def make_model(n_features, cards):
    kw = dict(n_classes=5, d_token=D_TOKEN, depth=3, heads=8, dropout=0.2)
    if cards is None:
        return FTTransformer(n_features=n_features, **kw)
    return CatFTTransformer(len(NUM), cards, **kw)


def train_one(Xtr, ytr, cards, seed):
    fi, vi = train_test_split(np.arange(len(ytr)), test_size=0.15, random_state=42, stratify=ytr)
    torch.manual_seed(seed)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    dl = DataLoader(TensorDataset(t(Xtr[fi]), torch.tensor(ytr[fi])), batch_size=32, shuffle=True)
    m = make_model(Xtr.shape[1], cards)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
    if WEIGHTED:
        counts = np.bincount(ytr[fi], minlength=5)
        w = torch.tensor(len(fi) / (5 * counts), dtype=torch.float32)
        crit = nn.CrossEntropyLoss(weight=w)
    else:
        crit = nn.CrossEntropyLoss()
    best, state, bad = -1, None, 0
    for ep in range(300):
        m.train()
        for xb, yb in dl:
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = m(t(Xtr[vi])).argmax(1).numpy()
        f = f1_score(ytr[vi], p, average="macro")
        if f > best:
            best, state, bad = f, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 40:
                break
    m.load_state_dict(state); m.eval()
    return m, ep + 1


res = []
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for k, (tr, te) in enumerate(skf.split(df, y)):
    t0 = time.time()
    Xtr, Xte, cards = prep(df.iloc[tr], df.iloc[te])
    probs, eps = 0, []
    for s in range(SEEDS):
        m, ep = train_one(Xtr, y[tr], cards, seed=k * 100 + s)
        eps.append(ep)
        with torch.no_grad():
            probs = probs + torch.softmax(m(torch.tensor(Xte, dtype=torch.float32)), 1).numpy()
    p = probs.argmax(1)
    a, f = accuracy_score(y[te], p), f1_score(y[te], p, average="macro")
    res.append((a, f))
    print(f"cfg {cfg} fold {k+1}: acc={a:.4f} macroF1={f:.4f} epochs={eps} ({time.time()-t0:.0f}s)", flush=True)
a = np.array(res)
print(f"cfg {cfg} MEAN acc={a[:,0].mean():.4f} ± {a[:,0].std():.4f}  macroF1={a[:,1].mean():.4f} ± {a[:,1].std():.4f}", flush=True)
json.dump(res, open(f"res_{cfg}.json", "w"))
