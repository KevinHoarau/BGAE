import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

device = "cuda"

layers = [1,2,4,8,16,32]
seeds = list(range(30))

nb_epoch = 100

data = []

for layer in layers:
    for seed in seeds:

        GAE_results = pickle.load(open("final_results/results_emb_8/layer_"+str(layer)+"_seed_"+str(seed)+".pickle","rb"))

        for epoch in range(100): 

            train_loss = 0
            test_loss = 0
            
            stop = False
            for event in GAE_results.keys():
                if(len(GAE_results[event]["train_losses"])>epoch):
                    train_loss += GAE_results[event]["train_losses"][epoch]/len(GAE_results)
                    test_loss += GAE_results[event]["test_losses"][epoch]/len(GAE_results)
                else:
                    stop = True
            
            if(not stop):
                data.append({
                                "epoch": epoch+1,
                                "layer": layer,
                                "type": "train",
                                "loss": train_loss
                })

                data.append({
                                "epoch": epoch+1,
                                "layer": layer,
                                "type": "test",
                                "loss": test_loss
                })

data = pd.DataFrame(data)

sns.set_style("whitegrid")
fts = 12

plt.figure(figsize=(8,5))

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


i=0
for layer in layers:
    
    i+=1
    ax = plt.subplot(2,3,i)
    
    cmap=sns.diverging_palette(250, 30, l=65, as_cmap=True)
    cmap = sns.color_palette()[2:4]

    dataNew = data[data["layer"]==layer]

    sns.lineplot(x="epoch", y="loss", hue="type", data=dataNew, ax=ax, palette=cmap)
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    _max = dataNew["epoch"].max()
    ax.set_xticks(np.arange(_max//5,_max,_max//5))
    
    if(i in [4,5,6]):
        ax.set_xlabel("Epochs", fontsize=fts)
    
    if(i%2==1):
        ax.set_ylabel("RE", fontsize=fts)
    
    ax.set_title("GCN layers: %d" % layer, fontsize=fts*1)
    
    ax.legend().remove()
    if(i==5):
        plt.tight_layout(h_pad=1) 
        plt.rcParams['legend.title_fontsize'] = fts
        ax.legend(loc="lower left", fontsize=fts, title="Type", bbox_to_anchor=[-0.15,2.6], ncol=2)


plt.savefig('GAE_loss.pdf', format='pdf', bbox_inches='tight')
print("Figure saved as: GAE_loss.pdf")