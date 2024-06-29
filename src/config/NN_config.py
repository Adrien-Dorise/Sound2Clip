"""
 Parameters example for neural network applications
 Author: Adrien Dorise (adrien.dorise@hotmail.com) - Law Tech Productions
 Created: June 2024
 Last updated: Adrien Dorise - June 2024
"""

import dragonflai.config.data_config as data_config
import dragonflai.model.neural_network_architectures.FCNN as FCNN
import dragonflai.model.neural_network_architectures.UNet as UNet
import dragonflai.model.neural_network_architectures.VITransformer as VIT  
import src.model.neural_network_architectures.s2c as s2c

import torch

input_size = 920
output_channels = 3
output_shape = 128
NN_model = s2c.S2CModel(input_size,output_shape, output_channels,save_path=data_config.save_path)

batch_size = 32
num_epoch = 1000
lr = 1e-4
wd = 1e-4
optimizer = torch.optim.AdamW
crit = torch.nn.MSELoss()

scheduler           = torch.optim.lr_scheduler.ReduceLROnPlateau
kwargs_scheduler = {'mode': 'min', 'factor': 0.99, 'patience': 30}
kwargs           = {'kwargs_scheduler': kwargs_scheduler}
