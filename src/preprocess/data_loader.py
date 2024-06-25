"""
 data loader creation for a neural network classification model
 Author: Adrien Dorise (adrien.dorise@hotmail.com), Edouard Villain - LR Technologies
 Created: June 2024
 Last updated: Adrien Dorise - June 2024
"""


import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



class Dataset(Dataset):
    '''
    Class used to store the dataset handled by pytorch. 
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
        return self.feat[idx], self.target[idx]
    
def create_dataset(frames_path, wav_path):
    frames = []
    frames_name = os.listdir(frames_path)
    extension = f".{frames_name[0].split('.')[1]}"
    frames_name = [name[0:-len(extension)] for name in frames_name]
    frames_name = sorted(frames_name, key=int)
    for file_name in frames_name:
        frames.append(Image.open(f"{frames_path}/{file_name}{extension}"))
    return frames

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

if __name__ == "__main__":
    frame_folder = "./data/images/attack_on_titan_s2/"
    wav_folder = "./data/audio/attack_on_titan_s2"
    frames = create_dataset(frame_folder, wav_folder)
    print("hello world")