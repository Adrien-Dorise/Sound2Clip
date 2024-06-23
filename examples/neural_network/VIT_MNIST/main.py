'''
 Main file to launch a debug test of the LR Technologies artificial intelligence template
 Author: Adrien Dorise (adorise@lrtechnologies.fr) - LR Technologies
 Created: June 2023
 Last updated: Adrien Dorise - August 2023
'''

from experiment import *

# import dragonflAI required module 
import torch 
# VITransformer because we are creating a VIT 
import dragonflai.model.neural_network_architectures.VITransformer as VITransformer

# set data path or use online dataset 
# here we use MNIST dataset online 
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn

from dragonflai.utils.utils_model import * 

if __name__ == "__main__":
    # parameters 
    n_channels        = 1
    embed_dim         = 64
    n_layers          = 8
    n_attention_heads = 8
    forward_mul       = 2
    image_size        = 32
    patch_size        = 8
    n_classes         = 10
    batch_size        = 256
    nb_workers        = 0
    num_epoch         = 15
    lr                = 1e-3
    wd                = 1e-4
    optimizer         = torch.optim.Adam
    crit              = nn.CrossEntropyLoss()
    scheduler         = torch.optim.lr_scheduler.ReduceLROnPlateau

    kwargs_optimizer = {'weight_decay': wd}
    kwargs_scheduler = {'mode': 'min', 'factor': 0.33, 'patience': 1}
    kwargs = {'kwargs_scheduler': kwargs_scheduler}


    input_shape = (batch_size, n_channels, image_size, image_size)

    # create transform method 
    transform = transforms.Compose([transforms.Resize([image_size, image_size]),
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.5], [0.5])])
    
    # set your base path 
    base_path = './examples/neural_network/VIT_MNIST/'
    # get data 
    train = datasets.MNIST(base_path + 'data', train=True, download=True, transform=transform)
    test = datasets.MNIST(base_path + 'data', train=False, download=True, transform=transform)
    # create loader 
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

    # create your model 
    NN_model = VITransformer.VIT_MNIST(n_channels, embed_dim, 
                                    n_layers, n_attention_heads, 
                                    forward_mul, image_size, 
                                    patch_size, n_classes)

    NN_model.save_path = base_path + 'results/'
    NN_model.progressBar.set_custom_cursor('▄︻デ══━一💨', '-', '⁍', ' ', '🎯')

    # create your experiment 
    experiment = Experiment(NN_model, train_loader, test_loader, 
                num_epoch=num_epoch,
                batch_size=batch_size,
                learning_rate=lr,
                weight_decay=wd,    
                optimizer=optimizer, 
                criterion=crit,
                scheduler=scheduler, 
                kwargs=kwargs, 
                nb_workers=nb_workers)

    experiment.fit()
    # disable visualise() for validation script 
    if 0:
        experiment.visualise()
