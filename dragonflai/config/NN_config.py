"""
 Parameters example for neural network applications
 Author: Adrien Dorise (adrien.dorise@hotmail.com) - LR Technologies
 Created: June 2023
 Last updated: Adrien Dorise - June 2024
"""

import dragonflai.config.data_config as data_config
import dragonflai.model.neural_network_architectures.FCNN as FCNN
import dragonflai.model.neural_network_architectures.UNet as UNet
import dragonflai.model.neural_network_architectures.VITransformer as VIT  


import torch

input_size = 2
output_size = 3
NN_model = FCNN.fullyConnectedNN(input_size,output_size,save_path=data_config.save_path)

batch_size = 64
num_epoch = 100
lr = 1e-3
wd = 1e-4
optimizer = torch.optim.AdamW
crit = torch.nn.CrossEntropyLoss()