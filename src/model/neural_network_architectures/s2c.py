from dragonflai.model.neuralNetwork import NeuralNetwork
from dragonflai.utils.utils_model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.0)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.layer.apply(init_weights)

    def forward(self, x):
        return self.layer(x)
    
class LinearDown(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.layer.apply(init_weights)

    def forward(self, x):
        return self.layer(x)
    
class ExtractFourier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):    
        return x[1]

    
class PeekLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):    
        return x


class S2CModel(NeuralNetwork):
    """Class for a Sound2Clip model creation"""

    def __init__(self, input_size, output_shape, nb_channels=3, save_path="./results/tmp/"):
        """Initialise the S2C Model class

        Args:
            input_size (int): Input size for the model
            output_size (int): number of classes to predict
            save_path (string): Folder path to save the results and checkpoints of the model
        """
        super().__init__(modelType=modelType.NEURAL_NETWORK, taskType=taskType.REGRESSION, save_path=save_path)
        
        # Model construction
        # To USER: Adjust your model here

        linear_neurons = [2048,1024,1024,1024]
        conv_kernels = [64,256,256,128,128,64,64]

        latent_dim = int(linear_neurons[-1] // conv_kernels[0])
        latent_img_shape = int(math.sqrt(latent_dim))
        self.output_shape = output_shape
        
        if(latent_dim % latent_img_shape != 0):
            raise Warning("Error in S2C architectures: Please ensure that the neurons and kernels chosen fits together to recreate an image.")


        # Encoder part -> Linear layers taking Fourier as input
        self.architecture.add_module("extract_fourrier", ExtractFourier())
        self.architecture.add_module('lin0', LinearDown(input_size, linear_neurons[0]))
        for lin_idx in range(len(linear_neurons)-1): 
            self.architecture.add_module(f'lin{lin_idx+1}', LinearDown(linear_neurons[lin_idx], linear_neurons[lin_idx+1] ))

        # Unflatten to switch from linear to conv
        img_shape = latent_img_shape
        self.architecture.add_module("unflatten", nn.Unflatten(1,(conv_kernels[0],latent_img_shape,latent_img_shape)))

        # Decoder part -> Convolution layers that increase image size to the desired output shape.
        conv_idx = 0
        while img_shape < self.output_shape:
            self.architecture.add_module(f"up_con{conv_idx}", Up(conv_kernels[conv_idx], conv_kernels[conv_idx+1]))
            conv_idx += 1
            img_shape *= 2
        self.architecture.add_module("out_conv", nn.Conv2d(conv_kernels[conv_idx], nb_channels, kernel_size=5, stride=1, padding=2, bias=False))
        self.architecture.add_module("out_sig", nn.Sigmoid())

        self.architecture.to(self.device)