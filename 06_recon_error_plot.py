import pickle
from torch_geometric.nn.models import GAE, VGAE
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import os
import seaborn as sns

from models import *

device = "cuda"
GAE_results = pickle.load(open("final_results/results_emb_8/layer_4_seed_0.pickle","rb"))
data = pickle.load(open("data_pyg_prepared.pickle","rb"))

layer = 4
seed = 0

def node_edges(g, node):
    node_index = g.ASN.index(node)

    pos_edges_src = g.edge_index[:,(g.edge_index[0]==node_index)]
    pos_edges_dest = g.edge_index[:,(g.edge_index[1]==node_index)]

    all_nodes = torch.arange(start=0, end=g.num_nodes)

    neg_edges_src = torch.tensor(np.setdiff1d(all_nodes, pos_edges_src[1])).reshape(1,-1)
    neg_edges_src = torch.concat([torch.full(neg_edges_src.shape, node_index),neg_edges_src])

    neg_edges_dest = torch.tensor(np.setdiff1d(all_nodes, pos_edges_dest[0])).reshape(1,-1)
    neg_edges_dest = torch.concat([neg_edges_dest, torch.full(neg_edges_dest.shape, node_index)])

    pos_edges = torch.concat([pos_edges_src,pos_edges_dest], dim=1)
    neg_edges = torch.concat([neg_edges_src,neg_edges_dest], dim=1)

    return((pos_edges, neg_edges))


sns.set_style("whitegrid")
fts = 12

plt.figure(figsize=(5,7))


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

i = 0
for testEv in data.keys():
    
    recon_na = []
    recon_a = []

    encoder = GCNEncoder(1, 8, 4)
    model = GAE(encoder).to(device)
    model.load_state_dict(GAE_results[testEv]["state_dict"])
        
    model.eval()
    with torch.no_grad():
        
        node = data[testEv]["node"]
        
        graphs = data[testEv]["NA"][:30]

        for g in graphs:
            g.x = g.nbIp.reshape(-1,1).float()

            z = model.encode(g.x.to(device), g.edge_index.to(device))
            
            pos_edges, neg_edges = node_edges(g, node)
            
            recon_na.append(model.recon_loss(z, pos_edges, neg_edges).item())
            
        
        graphs = data[testEv]["A"][:30]

        for g in graphs:
            g.x = g.nbIp.reshape(-1,1).float()

            z = model.encode(g.x.to(device), g.edge_index.to(device))
            
            pos_edges, neg_edges = node_edges(g, node)
            
            recon_a.append(model.recon_loss(z, pos_edges, neg_edges).item())
    
    i+=1
    ax = plt.subplot(4,2,i)
    
    cmap=sns.diverging_palette(250, 30, l=65, as_cmap=True)
    
    plt.plot(recon_na, label="0")
    plt.plot(recon_a, label="1")
    plt.title(testEv)
    
    
    ax.set_xticks(range(0,31,5))
    ax.set_xticklabels(range(0,61,10))
    ax.set_yticklabels([])

    if(i in [6,7]):
        ax.set_xlabel("Time [Minutes]", fontsize=fts)
    
    if(i%2==1):
        ax.set_ylabel("RE", fontsize=fts)
    
    ax.set_title(testEv, fontsize=fts*1.1)
    plt.tight_layout(h_pad=1.5) 
    
    if(i==7):
        plt.rcParams['legend.title_fontsize'] = fts
        ax.legend(loc="lower left", fontsize=fts, title="Label", bbox_to_anchor=[1.2,0.2], labels=["No anomaly","Anomaly"])

plt.savefig('recon_error_tight.pdf', format='pdf', bbox_inches='tight')

print("Figure saved as: recon_error.pdf")