import pickle
import torch
from torch import nn, flatten
from torch import nn, Tensor, optim
import random
from tqdm.auto import tqdm

import numpy as np
import os

class MLP(nn.Module):
    
    def __init__(self, device="cpu", w=30, emb_size=8):
        super(MLP, self).__init__()
        
        self.linear1 = nn.Linear(emb_size*w, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)
        
        self.w = w

    def forward(self, x):
        
        x = flatten(x)
        
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.linear3(x).sigmoid()
        
        return x


def train(trainEv, seed, epoch, lr, emb, emb_size):
    
    torch.manual_seed(seed)
    random.seed(seed)

    model = MLP(emb_size=emb_size).to(device)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for e in range(epoch):
        
        model.train()

        loss_sum = 0

        random.shuffle(trainEv)

        for name in trainEv:

            z = torch.Tensor([emb[seed][name]["A"]])
            pred = model(z.to(device))
            loss = loss_fn(pred.cpu(), Tensor([1]))
            
            loss_sum += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            z = torch.Tensor([emb[seed][name]["NA"]])
            pred = model(z.to(device))
            loss = loss_fn(pred.cpu(), Tensor([0]))
            
            loss_sum += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    return(model)

def test(testEv, model,seed, emb):

    model.eval()
    with torch.no_grad():

        name = testEv
        y_true = [1,0]
        y_score = []
        y_pred = []
        
        z = torch.Tensor([emb[seed][name]["A"]])
        score = model(z.to(device)).item()
        pred = int(score>0.5)
        y_score.append(score)
        y_pred.append(pred)

        z = torch.Tensor([emb[seed][name]["NA"]])
        score = model(z.to(device)).item()
        pred = int(score>0.5)
        y_score.append(score)
        y_pred.append(pred)

    return(y_true, y_score, y_pred)


print("Model evaluation")

device = "cuda"
nb_seeds = 30
nb_epoch = 50

embs=[1,2,4,8,16,32]
layers=[1,2,4,8,16,32]

skipIfExist = True

bar = tqdm(total=len(layers)*len(embs)*nb_seeds*7)

for emb_size in embs:

    for layer in layers :
        
        folder = "final_MLP/emb_"+str(emb_size)

        if not os.path.exists(folder):
           os.makedirs(folder)

        output = folder+"/layer_"+str(layer)+".pickle"
        
        if(not os.path.exists(output) or not skipIfExist):


            emb = pickle.load(open("final_emb/emb_"+str(emb_size)+"/layer_"+str(layer)+".pickle","rb"))

            results = {}

            for seed in range(nb_seeds):

                events = list(emb[0].keys())

                results[seed] = {}

                for testEv in events:

                    trainEv = events.copy()
                    trainEv.remove(testEv)

                    model = train(trainEv, seed, nb_epoch, 0.001, emb, emb_size)

                    y_true, y_score, y_pred = test(testEv, model, seed, emb)

                    results[seed][testEv] = {
                        "y_true": y_true,
                        "y_score": y_score,
                        "y_pred": y_pred,
                        "model_state_dict": model.state_dict()
                    }
                    
                    bar.update(1)

            pickle.dump(results, open(output, "wb")) 
                    
        else:
            bar.update(nb_seeds*7)
