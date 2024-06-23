import numpy as np
import matplotlib.pylab as plt 
import torch
from torch.utils.data import Dataset,Subset
from torchvision.datasets import MNIST
from torchvision import transforms


def create_loaders(data_path):
    """Create a dataLoader object containing the MNIST dataset.

    Args:
        data_path (string): Path in which the dataset is dowloaded

    Returns:
        torch.DataLoader: Train DataLoader
        torch.DataLoader: Test DataLoader
    """
    transform = transforms.Compose([transforms.Resize([28,28]),
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.5], [0.5]),
                                        transforms.Lambda(lambda x: torch.flatten(x))])
    
    train_set = MNIST(data_path, train=True, download=True, transform=transform)
    train_set = Subset(train_set, range(500))
    test_set = MNIST(data_path, train=False, download=True, transform=transform)
    test_set = Subset(test_set, range(50))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                        shuffle=False,
                                        batch_size=len(train_set))
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                        shuffle=False,
                                        batch_size=len(test_set))
    return train_loader, test_loader



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
    from config import data_path
    train, test = create_loaders(data_path=data_path)
    feature, target = extract_set(test)
    plt.imshow(feature[0].reshape(28,28), cmap="gray", interpolation="none")
    plt.show()
    print(target[0].item())