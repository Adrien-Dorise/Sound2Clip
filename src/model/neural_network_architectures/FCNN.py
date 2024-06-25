"""
This is DragonflAI example on a fully connected neural network architecture
Last Update by Adrien Dorise - June 2024

Author: Adrien Dorise (adrien.dorise@hotmail.com) - LR Technologies
Created: May 2024
"""

from dragonflai.model.neuralNetwork import NeuralNetwork
from dragonflai.utils.utils_model import *

import torch.nn as nn



class fullyConnectedNN(NeuralNetwork):
    """Example of a fully connected neural network classification model inheriting from the NeuralNetwork class.

    """
    def __init__(self, input_size, output_size, save_path="./results/tmp/"):
        """Initialise the fullyConnectedNN class

        Args:
            input_size (int): Input size for the model
            output_size (int): number of classes to predict
        """
        super().__init__(modelType=modelType.NEURAL_NETWORK, taskType=taskType.CLASSIFICATION, save_path=save_path)
        
        # Model construction
        # To USER: Adjust your model here

        self.architecture.add_module('lin1', nn.Linear(input_size, 64))
        nn.init.xavier_normal_(self.architecture[-1].weight)
        self.architecture.add_module('relu1', nn.ReLU())
        self.architecture.add_module('dropout1', nn.Dropout(p=0.2))
        
        self.architecture.add_module('lin2', nn.Linear(64 , 64))
        nn.init.xavier_normal_(self.architecture[-1].weight)
        self.architecture.add_module('relu2', nn.ReLU())
        self.architecture.add_module('dropout2', nn.Dropout(p=0.2))
     
        self.architecture.add_module('lin3', nn.Linear(64, output_size))
        nn.init.xavier_normal_(self.architecture[-1].weight)
        
        self.architecture.to(self.device)