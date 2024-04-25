import pickle
import matplotlib.pyplot as plt  
import numpy as np
import seaborn as sns
from sklearn import metrics

sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("deep"))

fig, ax = plt.subplots(figsize=(5,4))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

for i,layer in enumerate([1,2,4,8,16]):

    results = pickle.load(open(f"final_MLP/emb_8/layer_{layer}.pickle",'rb'))
    events = list(results[0].keys())
    
    y_true = []
    y_score = []
    roc_auc = []

    for seed in range(30):
        _y_true = []
        _y_score = []
        
        for testEv in events:    
            _y_true += results[seed][testEv]["y_true"]
            _y_score += results[seed][testEv]["y_score"]

        fpr, tpr, thresholds = metrics.roc_curve(_y_true, _y_score)
        roc_auc += [metrics.auc(fpr, tpr)]
        y_true += _y_true
        y_score += _y_score
        
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=np.mean(roc_auc), estimator_name='GNN layers : '+str(layer))
    display.plot(ax, dashes=[i+1,i+1]) 

pred = [0.5]*len(y_true)

fpr, tpr, thresholds = metrics.roc_curve(y_true, pred)
roc_auc = metrics.auc(fpr, tpr)

display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name="Reference line")
display.plot(ax) 
    
plt.savefig("roc_curve.pdf")
print("Figure saved : roc_curve.pdf")