"""
 Preprocessing toolbox for the super resolution toolbox
 Author: Edouard Villain (evillain@lrtechnologies.fr) - LR Technologies
 Created: November 2023
 Last updated: Edouard Villain - November 2023
"""
import numpy as np
import matplotlib.pylab as plt 
import torch
from torch.utils.data import Dataset
import matplotlib.patches as mpatches


class Dataset(Dataset):
    '''
    Class used to store the dataset handled by pytorch. 
    '''
    def __init__(self, feat_1, feat_2, target):
        if not torch.is_tensor(feat_1):
            self.feat_1 = torch.from_numpy(feat_1)
        else:
            self.feat_1 = feat_1
            
        if not torch.is_tensor(feat_2):
            self.feat_2 = torch.from_numpy(feat_2)
        else:
            self.feat_2 = feat_2
        
        if not torch.is_tensor(target):
            self.target = torch.from_numpy(target)
        else:
            self.target = target

    def __len__(self):
        return len(self.feat_1)

    def __getitem__(self, idx):
        return [self.feat_1[idx], self.feat_2[idx]], self.target[idx].type(torch.int64)


def create_set(n, list_center):
    """Create a set

    Args:
        n (int): Number of samples in the set
        list_center (list): Center of the set

    Returns:
        list[list]: Return the set with the coordinates of each samples
    """
  
    assert isinstance(list_center, list), 'Error when list_std is not a list'
    np.random.seed(948401971)
    ret = np.random.rand(n, len(list_center))
    for i in range(len(list_center)):
        ret[:,i] += list_center[i] - 0.5

    return ret

def create_dataset(list_N, list_center): 
    """Create a blob data set

    Args:
        list_N (list): list containing the number of samples in each blob
        list_center (list): Center of each blob

    Returns:
        list: List containing all the samples
    """
    dataset = []
    for c in range(len(list_N)):
        s = create_set(list_N[c], list_center[c])
        for data in s:
            dataset.append(data)
    return np.array(dataset)

def plot_dataset(dataset_1, dataset_2,classif, name):
    colors= ['b','r','g','k'] #classe 1 en blue, 2 en red...
    markers= ['o','^'] # pour différencier les point du centroide: rond et chapeur pour les centroides
    fig, ax = plt.subplots(nrows=1, ncols=2)
    red_patch = mpatches.Patch(color='red', label='High risks')
    blue_patch = mpatches.Patch(color='blue', label='Low risks')

    for i in range(len(dataset_1)):#parcours données du dataset
        ax[0].scatter(dataset_1[i][0],dataset_1[i][1],color=colors[classif[i]], marker=markers[0],alpha=0.5)
        ax[0].set_xlabel('Sport')
        ax[0].set_ylabel('Smoke')
        ax[0].grid(True)
        ax[0].legend(handles=[blue_patch,red_patch])
    
        ax[1].scatter(dataset_2[i][0],dataset_2[i][1],color=colors[classif[i]], marker=markers[0],alpha=0.5)
        ax[1].set_xlabel('Medical background')
        ax[1].set_ylabel('Age')
        ax[1].legend(handles=[blue_patch,red_patch])
        ax[1].grid(True)
    
    fig.suptitle('low versus high cancer risks')
    fig.savefig('{}.png'.format(name))
    #plt.show()


list_nb = [200,200] # Number of points per class
list_centroid_1 = [[0.95, 0.2], [0.2, 0.95]] # [smoke, sport]
list_centroid_2 = [[0.2, 0.2], [1, 1]] # [previous condition, age]
blob_1 = create_dataset(list_nb,list_centroid_1)
blob_2 = create_dataset(list_nb,list_centroid_2)
blob_1 = (blob_1 - np.min(blob_1)) / (np.max(blob_1) - np.min(blob_1))
blob_2 = (blob_2 - np.min(blob_2)) / (np.max(blob_2) - np.min(blob_2))
classif = np.array([0.0 for i in range(list_nb[0])] + [1.0 for i in range(list_nb[1])])

#plot_dataset(blob_1, blob_2, classif)

list_nb = [25,25] # Number of points per class
list_centroid_1 = [[0.9, 0.225], [0.25, 0.9]] # [smoke, sport]
list_centroid_2 = [[0.225, 0.175], [0.95, 0.95]] # [previous condition, age]
blob_1_t = create_dataset(list_nb,list_centroid_1)
blob_2_t = create_dataset(list_nb,list_centroid_2)
blob_1_t = (blob_1_t - np.min(blob_1_t)) / (np.max(blob_1_t) - np.min(blob_1_t))
blob_2_t = (blob_2_t - np.min(blob_2_t)) / (np.max(blob_2_t) - np.min(blob_2_t))
classif_t = np.array([0.0 for i in range(list_nb[0])] + [1.0 for i in range(list_nb[1])])

nx, ny = (50, 50)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)

test                 = [[] for _ in range(nx * ny)]
somker_sport         = [[] for _ in range(nx * ny)]
non_smoker_sport     = [[] for _ in range(nx * ny)]
non_smoker_non_sport = [[] for _ in range(nx * ny)]
smoker_non_sport     = [[] for _ in range(nx * ny)]


for i in range(len(x)):
    for j in range(len(y)):
        test[i*len(x) + j] = x[i] , y[j] 
        somker_sport[i*len(x) + j] = [0.85, 0.85]
        non_smoker_sport[i*len(x) + j] = [0.15, 0.85]
        non_smoker_non_sport[i*len(x) + j] = [0.15, 0.15]
        smoker_non_sport[i*len(x) + j] = [0.85, 0.15]
        
test                 = np.array(test)
somker_sport         = np.array(somker_sport)
non_smoker_sport     = np.array(non_smoker_sport)
non_smoker_non_sport = np.array(non_smoker_non_sport)
smoker_non_sport     = np.array(smoker_non_sport)

class_test = np.array([0 for i in range(len(test) ** 2)])

final_visu = [somker_sport, non_smoker_sport, non_smoker_non_sport, smoker_non_sport]
final_text = ['smoke and sport', 'no smoke and sport', 'no sport and no smoke', 'smoke and no sport']
