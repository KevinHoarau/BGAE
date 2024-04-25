import pickle
from torch_geometric.utils import negative_sampling

def prepareData(data):
    
    for name in data.keys():
        
        for label in ["A","NA"]:
            for i in range(len(data[name][label])):
                g = data[name][label][i]

                g.x = g.nbIp.reshape(-1,1).float()
                g.neg_edges = negative_sampling(g.edge_index)
                
                data[name][label][i] = g
    
    return(data)

print("Load data..")
data = pickle.load(open("data_pyg.pickle","rb"))
print("Prepare data..")
data = prepareData(data)

filename = "data_pyg_prepared.pickle"
pickle.dump(data, open(filename,"wb"))
print("Data saved to: "+ filename)