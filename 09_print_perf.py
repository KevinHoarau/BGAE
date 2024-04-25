import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

results = pickle.load(open("final_MLP/emb_8/layer_4.pickle",'rb'))

events = list(results[0].keys())

for testEv in events:

    accs = []
    f1s = []
    aucs = []

    for seed, data in results.items():

        accs.append(accuracy_score(data[testEv]["y_true"], data[testEv]["y_pred"]))
        f1s.append(f1_score(data[testEv]["y_true"], data[testEv]["y_pred"]))
        aucs.append(roc_auc_score(data[testEv]["y_true"], data[testEv]["y_score"]))
        
    metrics = {
        "Accuracy (Mean)": np.array(accs).mean(),
        "Accuracy (Std dev)": np.array(accs).std(),
        "F1 score (Mean)": np.array(f1s).mean(),
        "F1 score (Std dev)": np.array(f1s).std(),
        "Auc (Mean)": np.array(aucs).mean(),
        "Auc (Std dev)": np.array(aucs).std(),
    }
    
    print( "%s & %.2f & %.2f & %.2f & %.2f & %.2f  & %.2f \\\\ \hline" % (testEv, metrics["Accuracy (Mean)"], metrics["Accuracy (Std dev)"],metrics["F1 score (Mean)"],metrics["F1 score (Std dev)"], metrics["Auc (Mean)"],metrics["Auc (Std dev)"]))


accs = []
f1s = []
aucs = []

for seed, data in results.items():  
    
    y_true = []
    y_pred = []
    y_score = []
    
    for testEv in events:
        
        y_true += data[testEv]["y_true"]
        y_pred += data[testEv]["y_pred"]
        y_score += data[testEv]["y_score"]
        
    
    accs.append(accuracy_score(y_true, y_pred))
    f1s.append(f1_score(y_true, y_pred))
    aucs.append(roc_auc_score(y_true, y_score))     
    
metrics = {
    "Accuracy (Mean)": np.array(accs).mean(),
    "Accuracy (Std dev)": np.array(accs).std(),
    "F1 score (Mean)": np.array(f1s).mean(),
    "F1 score (Std dev)": np.array(f1s).std(),
    "Auc (Mean)": np.array(aucs).mean(),
    "Auc (Std dev)": np.array(aucs).std(),
}

print( "\\textbf{Overall} & \\textbf{%.2f} & \\textbf{%.2f} & \\textbf{%.2f} & \\textbf{%.2f} & \\textbf{%.2f} & \\textbf{%.2f}  \\\\ \hline" % (metrics["Accuracy (Mean)"], metrics["Accuracy (Std dev)"],metrics["F1 score (Mean)"],metrics["F1 score (Std dev)"], metrics["Auc (Mean)"],metrics["Auc (Std dev)"]))

