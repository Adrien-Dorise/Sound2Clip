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



class S2CModel(NeuralNetwork):
    """Class for a Sound2Clip model creation"""

    def __init__(self, input_size, nb_channels=3, save_path="./results/tmp/"):
        """Initialise the S2C Model class

        Args:
            input_size (int): Input size for the model
            output_size (int): number of classes to predict
            save_path (string): Folder path to save the results and checkpoints of the model
        """
        super().__init__(modelType=modelType.NEURAL_NETWORK, taskType=taskType.REGRESSION, save_path=save_path)
        
        # Model construction
        # To USER: Adjust your model here

        linear_neurons = [512,512,256,128]
        conv_kernels = [32,32,32,32,32,32,32]

        latent_dim = int(linear_neurons[-1] // conv_kernels[0])
        latent_img_shape = int(math.sqrt(latent_dim))
        self.output_shape = math.pow(latent_img_shape,len(conv_kernels))
        
        if(latent_dim % latent_img_shape != 0):
            raise Warning("Error in S2C architectures: Please ensure that the neurons and kernels chosen fits together to recreate an image.")

        self.architecture.add_module("extract_fourrier", ExtractFourier())
        self.architecture.add_module('lin1', LinearDown(input_size, linear_neurons[0]))
        self.architecture.add_module('lin2', LinearDown(linear_neurons[0], linear_neurons[1]))
        self.architecture.add_module('lin3', LinearDown(linear_neurons[1], linear_neurons[2]))
        self.architecture.add_module('lin4', LinearDown(linear_neurons[2], linear_neurons[3]))

        self.architecture.add_module("unflatten", nn.Unflatten(1,(conv_kernels[0],latent_img_shape,latent_img_shape)))

        self.architecture.add_module("up_conv1", Up(conv_kernels[0], conv_kernels[1]))
        self.architecture.add_module("up_conv2", Up(conv_kernels[1], conv_kernels[2]))
        self.architecture.add_module("up_conv3", Up(conv_kernels[2], conv_kernels[3]))
        self.architecture.add_module("up_conv4", Up(conv_kernels[3], conv_kernels[4]))
        self.architecture.add_module("up_conv5", Up(conv_kernels[4], conv_kernels[5]))
        self.architecture.add_module("up_conv6", Up(conv_kernels[5], conv_kernels[6]))
        self.architecture.add_module("out_conv", nn.Conv2d(conv_kernels[6], nb_channels, kernel_size=5, stride=1, padding=2, bias=False))
        self.architecture.add_module("out_sig", nn.Sigmoid())

        self.architecture.to(self.device)