import torch
from torch import optim
from torch_geometric.nn.models import GAE
import pickle
from tqdm.auto import tqdm
import random
from torch_geometric.data import Batch
import numpy as np
import sys, os

from models import *

from torch_geometric.utils import negative_sampling

def test(data, model, name):
    
    model.eval()
    with torch.no_grad():
    
        graphs = data[name]["NA"][:30]
        
        loss_sum = 0
        
        for g in graphs:

            z = model.encode(g.x.to(device), g.edge_index.to(device))

            loss = model.recon_loss(z, g.edge_index, g.neg_edges)

            loss_sum += loss.item()
 
    return(loss_sum)

def train(data, trainEv, testEv, seed, epoch, lr, layer, emb):
    
    torch.manual_seed(seed)
    random.seed(seed)

    encoder = GCNEncoder(1, emb, layer)
    model = GAE(encoder).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    
    # initial loss
    initLoss = 0
    for name in trainEv:
        initLoss += test(data, model, name)
            
    prev_loss_sum = initLoss #1e99
    patience = 0
    
    for e in range(epoch):
        
        model.train()
        
        loss_sum = 0
        
        graphs = []
        for name in trainEv:
            
            graphs += data[name]["NA"][:30]
            
        random.shuffle(graphs)
        
        batch_size = 16
            
        for i in range(0,len(graphs),batch_size):
            start, end = (i,min(i+batch_size,len(graphs)))
            
            loss = 0

            for g in graphs[start:end]:

                z = model.encode(g.x.to(device), g.edge_index.to(device))
                
                loss += model.recon_loss(z, g.edge_index, g.neg_edges)

            loss_sum += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        #Early stopping

        delta_loss = prev_loss_sum-loss_sum

        
        if(delta_loss< initLoss * convergence_threshold):
            patience += 1
        else:
            patience = 0
        if(patience>4):
            bar.update(epoch-e)
            break

        prev_loss_sum = loss_sum
            
        train_losses.append(loss_sum/len(trainEv))
        test_losses.append(test(data, model, testEv))
        
        bar.update(1)
        
    return(model,train_losses,test_losses)


def crossVal(seed, layer, emb, nb_epoch):
    
    folder = "final_results/results_emb_"+str(emb)
    
    if not os.path.exists(folder):
       os.makedirs(folder)

    output = folder + "/layer_"+str(layer)+"_seed_"+str(seed)+".pickle"
    
    if(not os.path.exists(output) or not skipIfExist):
        
        events = list(data.keys())

        results = {}

        for testEv in events:
            trainEv = events.copy()
            trainEv.remove(testEv)

            model,train_losses,test_losses = train(data, trainEv, testEv, seed, nb_epoch, 0.001, layer, emb)

            results[testEv] = { 
                                "state_dict" : model.cpu().state_dict(),
                                "train_losses" : train_losses,
                                "test_losses" : test_losses
                                }

        pickle.dump(results, open(output,"wb"))
    else:
        bar.update(7*nb_epoch)


print("BGAE model evaluation")

print("Load data..")
data = pickle.load(open("data_pyg_prepared.pickle","rb"))

device = "cuda"

skipIfExist = True

seeds = list(range(30))
layers = [1,2,4,8,16,32]
embs = [1,2,4,8,16,32]

nb_epoch = 100
convergence_threshold = 5e-3

bar = tqdm(total=len(embs)*len(seeds)*len(layers)*7*nb_epoch)

for emb in embs:
    for seed in seeds:
        for layer in layers:
            crossVal(int(seed), int(layer),emb, nb_epoch)
        

print("##########")
print("Done")