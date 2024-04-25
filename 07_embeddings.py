import pickle
from torch_geometric.nn.models import GAE, VGAE
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import os

from models import *

device = "cuda"

data = pickle.load(open("data_pyg_prepared.pickle","rb"))

layers = [1,2,4,8,16,32]
seeds = range(30)

embs = [1,2,4,8,16,32]

i = 0

emb = {}

skipIfExist = True

bar = tqdm(total=len(layers)*len(embs))

for layer in layers:
    
    for emb_size in embs:

        folder = "final_emb/emb_"+str(emb_size)

        if not os.path.exists(folder):
           os.makedirs(folder)

        output = folder+"/layer_"+str(layer)+".pickle"
        
        if(not os.path.exists(output) or not skipIfExist):


            for seed in seeds:

                GAE_results = pickle.load(open("final_results/results_emb_"+str(emb_size)+"/layer_"+str(layer)+"_seed_"+str(seed)+".pickle","rb"))

                emb[seed] = {}

                for testEv in data.keys():

                    recon_na = []
                    recon_a = []

                    encoder = GCNEncoder(1, emb_size, layer)
                    model = GAE(encoder).to(device)
                    model.load_state_dict(GAE_results[testEv]["state_dict"])

                    model.eval()
                    with torch.no_grad():

                        node = data[testEv]["node"]

                        graphs = data[testEv]["NA"][:30]

                        z_na = []
                        z_a = []

                        for g in graphs:

                            z = model.encode(g.x.to(device), g.edge_index.to(device))

                            node_index = g.ASN.index(node)

                            z_na.append(z[node_index].tolist())

                        graphs = data[testEv]["A"][:30]

                        for g in graphs:

                            z = model.encode(g.x.to(device), g.edge_index.to(device))

                            node_index = g.ASN.index(node)

                            z_a.append(z[node_index].tolist())

                        emb[seed][testEv] = {
                            "NA": z_na,
                            "A": z_a
                        }

            folder = "final_emb/emb_"+str(emb_size)

            if not os.path.exists(folder):
               os.makedirs(folder)

            pickle.dump(emb, open(output,"wb")) 

        bar.update(1)
            
