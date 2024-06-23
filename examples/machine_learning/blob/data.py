"""
 Data set creation for the machine learning "blob" example.
 In this script, the blobs are created in reproductible manne by setting the seed.
 The data loader creation is also taken care in this script.
 Author: Adrien Dorise (adrien.dorise@hotmail.com) - LR Technologies
 Created: May 2024 
 Last updated: Adrien Dorise - May 2024
"""


import numpy as np
import matplotlib.pylab as plt 
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from enum import Enum



class Dataset(Dataset):
    '''
    Class used to store the dataset handled by pytorch. 
    '''
    def __init__(self, feat, target):
        """Initialisation of the Dataset

        Args:
            feat (List): Features of the dataset
            target (list): Targets of the dataset
        """
        if not torch.is_tensor(feat):
            self.feat = torch.from_numpy(feat)
        else:
            self.feat = feat
        
        if not torch.is_tensor(target):
            self.target = torch.from_numpy(target)
        else:
            self.target = target

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, idx):
        return self.feat[idx], self.target[idx]


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

class Paradigm(Enum):
    CLASSIFICATION = 1
    CLUSTERING = 2
    REGRESSION = 3

def create_loaders(list_N, list_center, test_size = 0.2, paradigm = Paradigm.CLASSIFICATION):
    if paradigm in [Paradigm.CLASSIFICATION, Paradigm.CLUSTERING] :
        features = create_dataset(list_N,list_center)
        targets = []
        for indice in range(len(list_N)):
            targets.extend([indice]*list_N[indice])
        targets = np.array(targets)
    if paradigm is Paradigm.REGRESSION:
        data = create_dataset(list_N,list_center)
        features = np.array([[feat] for feat in data[:,0]])
        targets = np.array(data[:,1])
        
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=test_size, shuffle=True, random_state=948401971)    
    train_dataset = Dataset(features_train,targets_train)
    test_dataset = Dataset(features_test, targets_test)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                        shuffle=False,
                                        batch_size=len(train_dataset))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        shuffle=False,
                                        batch_size=len(test_dataset))
    return train_loader, test_loader


def plot_dataset(dataset,list_centroides,cluster):
    colors= ['b','r','g','k'] #classe 1 en blue, 2 en red...
    markers= ['o','^'] # pour différencier les point du centroide: rond et chapeur pour les centroides
    fig= plt.figure()
    ax=fig.gca()
    for i in range(len(dataset)):#parcours données du dataset
        ax.scatter(dataset[i][0],dataset[i][1],color=colors[cluster[i]], marker=markers[0],alpha=0.5)
    for i in range(len(list_centroides)):
        ax.scatter(list_centroides[i][0],list_centroides[i][1],color=colors[i],marker=markers[1],s=80)#size 80
    #fig.savefig('{}.png'.format())
    plt.show()
    


def extract_set(dataset):
        """Extract features and targets from a dataLoader object
        Tool function used by other function of the class (fit, predict, gridSearch)
    
        Args:
            dataset (DataLoader): DataLoader obejct containing a data set

        Raises:
            Warning: Most scikit-learn models do not implement mini-batch training. Therefore, mini-batch is disable for this class. 
                If the DataLoader contain multiple batch, an error is raised.

        Returns:
            feature set (array)
            target set (array)
        """
        for i, data in enumerate(dataset, 0):
            if(dataset.batch_size != len(dataset.dataset)):
                raise Warning(f"The number of batch exceed 1. This program does not support multi-batch processing. Make sure that batch_size is equal to the number of inputs.\nBatch_size / input: [{dataset.batch_size} / {len(dataset.dataset)}]")
            else:
                # get the inputs; data is a list of [inputs, target]
                return data[0],data[1]

if __name__ == "__main__":
    list_nb = [25,25,25] # Number of points per class
    list_centroid = [[-0.4,0.4],[0.4,0.8],[0.8,-0.2]]
    dataset = create_dataset(list_nb,list_centroid)
    print(dataset)
    plot_dataset(dataset,list_centroid,[0 for i in range(list_nb[0])] + [1 for i in range(list_nb[1])] + [2 for i in range(list_nb[2])] )

    train_loader, test_loader = create_loaders(list_nb, list_centroid)
    train = extract_set(train_loader)
    test = extract_set(test_loader)
    
    train_loader, test_loader = create_loaders(list_nb, list_centroid, paradigm=Paradigm.REGRESSION)
    train_regression = extract_set(train_loader)
    test_regression = extract_set(test_loader)
    print(test_regression)
    