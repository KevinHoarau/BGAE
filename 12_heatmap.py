import pickle
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

embs = [32,16,8,4,2,1]
layers = [1,2,4,8,16,32]

r = []

for emb in embs:
    
    rr = []
    
    for layer in layers:
        
        results = pickle.load(open("final_MLP/emb_"+str(emb)+"/layer_"+str(layer)+".pickle", 'rb'))

        events = list(results[0].keys())
        
        accs = []
        f1s = []
        aucs = []
        
        for seed, data in results.items():  
            
            y_true = []
            y_pred = []
            
            for testEv in events:
                
                y_true += data[testEv]["y_true"]
                y_pred += data[testEv]["y_pred"]
                
            accs.append(accuracy_score(y_true, y_pred))

        rr.append(np.array(accs).mean())
    
    r.append(rr)
    
data = np.array(r)

fts=13

fig, ax = plt.subplots()

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

cmap = sns.color_palette("mako", as_cmap=True)

sns.heatmap(data, annot=True, xticklabels=layers, yticklabels=embs, linewidth=1, annot_kws={"size": fts}, cmap=cmap)

ax.set_xlabel("GNN Layers", fontsize=fts)
ax.set_ylabel("Embedding size", fontsize=fts)

fig.tight_layout()

plt.savefig("heatmap.pdf")
print("Figure saved : heatmap.pdf")