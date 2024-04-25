import seaborn as sns
    
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt


models = ["Temporal GNN","Graph Auto Encoder [This paper]","Graph Neural Network [13]","Recurrent Neural Network [26]",
          "Multilayer Perceptron [11]","Support-Vector Machine [11]","Decision Tree [11]","Naive Bayes [11]","K-Nearest Neighbors [11]"]
events = ["Route\n leaks", "Origin\n hijacking", "Path\n hijacking"] 

data = np.array([
                [0,0,0],
                [0.99,0,0],
                [0.96,0,0],
                [0,0,0.67],
                [0.840,0.550,0.598],#MLP
                [0.819, 0.606,0.609],#SVM
                [0.819,0.520,0.577],#DT
                [0.852,0.562,0.605],#NB
                [0.881,0.573,0.522],#KNN
                ])

fts=12

fig, ax = plt.subplots(figsize=(7,5.5))

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

cmap = sns.color_palette("Blues", as_cmap=True)

mask = (data==0)
sns.heatmap(data, annot=True, xticklabels=events, yticklabels=models, cbar=False, 
            linewidth=0, annot_kws={"size": fts, "color":cmap(0)}, cmap=cmap, mask=mask, vmin=0)

ax.tick_params(axis='both', which='major', labelsize=fts)

ax.set_ylabel("Machine learning models", fontsize=fts*1.1, labelpad=15)
ax.set_xlabel("BGP anomalies", fontsize=fts*1.1, labelpad=15)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=cmap(0.15),
                         label='Not explored yet')]

ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=fts)
ax.set_facecolor(cmap(0.2))
fig.tight_layout()

plt.savefig("perspectives_camera_ready.pdf")
print("Figure saved : perspectives_camera_ready.pdf")