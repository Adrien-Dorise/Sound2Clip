from dragonflai.model.neuralNetwork import NeuralNetwork
from dragonflai.utils.utils_model import *

import torch.nn as nn
import torch.optim
    
def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
        

    
import torch
import torch.nn as nn  
               
class MLP(nn.Module):
    def __init__(self, l_neurons, l_batchNorm, l_activation, l_drop):
        super(MLP, self).__init__()
        self.layers = nn.Sequential() 
        
        assert isinstance(l_neurons, list), \
            'l_neurons should be a list'
        assert isinstance(l_batchNorm, list), \
            'l_batchNorm should be a list'
        assert isinstance(l_activation, list), \
            'l_activation should be a list'
        assert isinstance(l_drop, list), \
            'l_drop should be a list'
        assert len(l_neurons) == len(l_batchNorm) and \
            len(l_neurons) == len(l_activation) and \
            len(l_neurons) == len(l_drop), \
            'all lists should have the same length'
            
        # iterate throught l_neurons 
        for i in range(1, len(l_neurons)):
            self.layers.add_module('lin_{}'.format(i), nn.Linear(l_neurons[i-1], l_neurons[i]))
            if l_batchNorm[i]:
                self.layers.add_module('BN_{}'.format(i), nn.BatchNorm1d(l_neurons[i]))
            if l_activation[i]:
                self.layers.add_module('act_{}'.format(i), nn.ReLU())
            if l_drop[i] > 0:
                self.layers.add_module('drop_{}'.format(i), nn.Dropout1d(l_drop[i]))
   
        for l in self.layers:
            init_weights(l)
        
        
    def forward(self, x):
        return self.layers(x)  

class multiple_input_MLP(nn.Module):
    def __init__(self, input_size_1, input_size_2):
        super(multiple_input_MLP, self).__init__()
        
        self.intermediate_size = 64
        
        l_neurons_1    = [input_size_1, 128, 128, self.intermediate_size]
        l_batchNorm_1  = [True, True, True, True]
        l_activation_1 = [True, True, True, False]
        l_drop_1       = [0, 0, 0, 0]
        
        l_neurons_2   = [input_size_2, 128, 128, self.intermediate_size]
        l_batchNorm_2  = [True, True, True, True]
        l_activation_2 = [True, True, True, False]
        l_drop_2      = [0, 0, 0, 0]

        
        self.model_process_input_1 = MLP(l_neurons_1, l_batchNorm_1, l_activation_1, l_drop_1)
        self.model_process_input_2 = MLP(l_neurons_2, l_batchNorm_2, l_activation_2, l_drop_2)
        
        l_neurons_out    = [2 * self.intermediate_size, 128, 128, 2]
        l_batchNorm_out  = [True] * (len(l_neurons_out) - 1) + [False]
        l_activation_out = [True] * len(l_neurons_out)
        l_drop_out       = [0] + [0.1] * (len(l_neurons_out) - 2) + [0]

        self.model_out = MLP(l_neurons_out, l_batchNorm_out, l_activation_out, l_drop_out)
        

        
    def forward(self, x):
            
        x0_process = self.model_process_input_1(x[0])
        x1_process = self.model_process_input_2(x[1])
        
        in_mix = torch.concat((x0_process, x1_process), dim=1)
        out    = self.model_out(in_mix)

        return out

class multi_input_MLP(NeuralNetwork):
    def __init__(self, input_size_1, input_size_2):
        super(multi_input_MLP, self).__init__(modelType=modelType.NEURAL_NETWORK, taskType=taskType.CLASSIFICATION)

        self.architecture = multiple_input_MLP(input_size_1, input_size_2).to(self.device)
