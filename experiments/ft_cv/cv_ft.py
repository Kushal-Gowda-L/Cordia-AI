"""5-fold stratified CV of the unchanged FT-Transformer (depth3, heads8, d256, dropout0.2, lr1e-4, wd1e-4,
early stopping on macro F1, patience 40), on real UCI rows only.
Config A: current 10 SF-2 features. Config B: + slope, ca, thal + missing-value flags for them.
Per fold: medians and scaler fit on the training folds only; 15% of training folds held out for early stopping."""
import sys, json, time
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from ft_transformer import FTTransformer
torch.set_num_threads(2)
BASE=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak"]
EXTRA=["slope","ca","thal"]
cfg=sys.argv[1]
df=pd.read_csv("heart_disease_raw.csv"); y=df["num"].values
def build(d):
    X=d[BASE+(EXTRA if cfg=="B" else [])].copy()
    if cfg=="B":
        for c in EXTRA: X[c+"_missing"]=X[c].isna().astype(float)
    return X
X=build(df)
def run_fold(Xtr,ytr,Xte,yte,seed):
    med=Xtr.median(); Xtr=Xtr.fillna(med); Xte=Xte.fillna(med)
    sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
    fi,vi=train_test_split(np.arange(len(ytr)),test_size=0.15,random_state=42,stratify=ytr)
    torch.manual_seed(seed)
    t=lambda a:torch.tensor(a,dtype=torch.float32)
    dl=DataLoader(TensorDataset(t(Xtr[fi]),torch.tensor(ytr[fi])),batch_size=32,shuffle=True)
    m=FTTransformer(n_features=Xtr.shape[1],n_classes=5)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4); crit=nn.CrossEntropyLoss()
    best=-1;state=None;bad=0
    for ep in range(300):
        m.train()
        for xb,yb in dl:
            opt.zero_grad(); crit(m(xb),yb).backward(); opt.step()
        m.eval()
        with torch.no_grad(): p=m(t(Xtr[vi])).argmax(1).numpy()
        f=f1_score(ytr[vi],p,average="macro")
        if f>best: best=f;state={k:v.clone() for k,v in m.state_dict().items()};bad=0
        else:
            bad+=1
            if bad>=40: break
    m.load_state_dict(state); m.eval()
    with torch.no_grad(): p=m(t(Xte)).argmax(1).numpy()
    return accuracy_score(yte,p),f1_score(yte,p,average="macro"),ep+1
res=[]
skf=StratifiedKFold(5,shuffle=True,random_state=42)
for k,(tr,te) in enumerate(skf.split(X,y)):
    t0=time.time()
    a,f,ep=run_fold(X.iloc[tr],y[tr],X.iloc[te],y[te],seed=k)
    res.append((a,f)); print(f"cfg {cfg} fold {k+1}: acc={a:.4f} macroF1={f:.4f} epochs={ep} ({time.time()-t0:.0f}s)",flush=True)
a=np.array(res)
print(f"cfg {cfg} MEAN acc={a[:,0].mean():.4f} ± {a[:,0].std():.4f}  macroF1={a[:,1].mean():.4f} ± {a[:,1].std():.4f}",flush=True)
json.dump(res,open(f"res_{cfg}.json","w"))
