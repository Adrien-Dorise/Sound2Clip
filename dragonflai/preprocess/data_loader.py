"""
 data loader creation for a neural network classification model
 Author: Adrien Dorise (adrien.dorise@hotmail.com), Edouard Villain - LR Technologies
 Created: June 2024
 Last updated: Adrien Dorise - June 2024
"""


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



class Dataset(Dataset):
    '''
    Class used to store the dataset handled by pytorch. 
    Note that the targets are get as one hot encoded vector.
    '''
    def __init__(self, feat, target):
        if not torch.is_tensor(feat):
            self.feat = torch.from_numpy(feat)
        else:
            self.feat = feat
        
        if not torch.is_tensor(target):
            self.target = torch.from_numpy(target)
        else:
            self.target = target

        self.n_classes = len(torch.unique(self.target))
        
    def __len__(self):
        return len(self.feat)

    def __getitem__(self, idx):
        one_hot_target = F.one_hot(self.target[idx].to(torch.int64), num_classes=self.n_classes).type(torch.float)
        return self.feat[idx], one_hot_target
    

def create_loaders(features, targets, batch_size=None, test_size=0.2, shuffle=True):
    """Create a PyTorch DataLoader object that can handles the data processing when performing training or prediction of a ML/NN model.
    It is possible to create a test set automatically with the test_size parameter.
    
    Args:
        features (list or np.array): List of features
        targets (list or np.array): List of targets
        batch_size (None or int, optional): Size of the batch used during training. When at None, only one batch is created. 
                                            Note, that ML models can't handle multiple batches. Defaults to None.
        test_size (float, optional): Proportion of samples used for the test set between ]0,1[. Defaults to 0.2.
        shuffle (bool, optional): Set to true to shuffle the data before creating the loader. Defaults to True.

    Returns:
        DataLoader, DataLoader: Train and test DataLoader objects
    """
    
    #Divide between train set and test set
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=test_size, shuffle=shuffle, random_state=948401971)
    
    # Create Dataset objects
    train_dataset = Dataset(features_train,targets_train)
    test_dataset = Dataset(features_test, targets_test)
    
    # Define batch size if not specified in the parameters
    if batch_size is None:
        batch_size = len(train_dataset)
    
    # Create loader objects
    train_loader = DataLoader(dataset=train_dataset,
                            shuffle=False,
                            batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset,
                            shuffle=False,
                            batch_size=int(batch_size*test_size))
    return train_loader, test_loader