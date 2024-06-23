'''
 Main file to launch a debug test of the LR Technologies artificial intelligence template
 Author: Adrien Dorise (adorise@lrtechnologies.fr) - LR Technologies
 Created: June 2023
 Last updated: Adrien Dorise - August 2023
'''

from experiment import *
import torch.nn as nn

from dragonflai.utils.utils_model import * 
import NN as NN 
import dataloader  

if __name__ == "__main__":
    # parameters 
    input_size_1     = 2
    input_size_2 = 2
    batch_size          = 32
    num_epoch           = 15
    lr                  = 1e-3
    wd                  = 1e-4
    nb_workers          = 0
    optimizer           = torch.optim.RMSprop
    crit                = nn.CrossEntropyLoss()
    scheduler           = torch.optim.lr_scheduler.ReduceLROnPlateau

    kwargs_scheduler = {'mode': 'min', 'factor': 0.33, 'patience': 1}
    kwargs           = {'kwargs_scheduler': kwargs_scheduler}

    main_path  = r"examples/neural_network/multiples_inputs/data/"

    train = dataloader.Dataset(dataloader.blob_1, dataloader.blob_2, dataloader.classif)
    test  = dataloader.Dataset(dataloader.blob_1_t, dataloader.blob_2_t, dataloader.classif_t)
    
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

    NN_model = NN.multi_input_MLP(input_size_1, input_size_2)
    # set your base path 
    base_path = './examples/neural_network/multiples_inputs'
    NN_model.save_path = base_path + '/results'

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
        experiment.predict()
        experiment.visualise()