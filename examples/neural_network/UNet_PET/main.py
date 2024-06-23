'''
 Main file to launch a debug test of the LR Technologies artificial intelligence template
 Author: Adrien Dorise (adorise@lrtechnologies.fr) - LR Technologies
 Created: Avril 2024
 Last updated: Adrien Dorise - Avril 2024
'''

from experiment import *

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as T

import dragonflai.model.neural_network_architectures.UNet as UNet
from dragonflai.utils.utils_model import * 

class OxfordIIITPetsAugmented(datasets.OxfordIIITPet):
    def __init__(self, root: str, split: str, target_types="segmentation", download=False, image_size=64):
        super().__init__(root=root, split=split, target_types=target_types, download=download)
        self.image_size = image_size

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        (input, target) = super().__getitem__(idx)

        trans = T.Compose([
            T.ToTensor(),
            T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.NEAREST),
        ])

        input = trans(input)
        target = trans(target)

        target = target * 255
        target = target.to(torch.long)
        target -= 1
        target = target.squeeze()

        return (input, target)

if __name__ == "__main__":
    # parameters 
    n_channels              = 3
    image_size              = 64
    n_classes               = 3
    batch_size              = 8
    nb_workers              = 0
    num_epoch               = 3
    lr                      = 1e-3
    wd                      = 1e-4
    optimizer               = torch.optim.Adam
    crit                    = nn.CrossEntropyLoss()
    scheduler               = torch.optim.lr_scheduler.ReduceLROnPlateau
    numberOfImagesToDisplay = 5

    kwargs_optimizer = {'weight_decay': wd}
    kwargs_scheduler = {'mode': 'min', 'factor': 0.33, 'patience': 1}
    kwargs = {'kwargs_scheduler': kwargs_scheduler}

    base_path = './examples/neural_network/UNet_PET/'

    input_shape = (batch_size, n_channels, image_size, image_size)
    
    train = OxfordIIITPetsAugmented(base_path + 'data', split="trainval", target_types="segmentation", download=True, image_size=image_size)
    test = OxfordIIITPetsAugmented(base_path + 'data', split="test", target_types="segmentation", download=True, image_size=image_size)

    train = torch.utils.data.Subset(train, range(300))
    test = torch.utils.data.Subset(test, range(10))

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=nb_workers,
                                                drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test,
                                                batch_size=batch_size ,
                                                shuffle=False,
                                                num_workers=nb_workers,
                                                drop_last=False)
 
    NN_model = UNet.UNet_PET(n_channels, n_classes)

    NN_model.save_path = base_path + 'results/'
    NN_model.progressBar.set_custom_cursor('▄︻デ══━一💨', '-', '⁍', ' ', '🎯')

    experiment = Experiment(NN_model, train_loader, test_loader, 
                num_epoch=num_epoch,
                batch_size=batch_size,
                learning_rate=lr,
                weight_decay=wd,    
                optimizer=optimizer, 
                criterion=crit,
                scheduler=scheduler, 
                kwargs=kwargs, 
                nb_workers=nb_workers,
                numberOfImagesToDisplay=numberOfImagesToDisplay)

    experiment.fit()
    # disable visualise() for validation script 
    if 0:
        experiment.visualise()