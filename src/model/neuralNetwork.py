"""
This package references all neural network classes used in the application.
Author: Julia Cohen - Adrien Dorise (adrien.dorise@hotmail.com) - Edouard Villain (evillain@lrtechnologies.fr) - LR Technologies
Created: March 2023
Last updated: Edouard Villain - April 2024 

Pytorch is the main API used.
It is organised as follow:
    - NeuralNetwork class: Core class that contains all tools to use a neural network (training, testing, print...)
    - Subsidiary net classes: This class has to be setup by the user. They contains the information about the architecture used for each custom networks.
    
The package works as follow:
    - Use or create a neural network class.
    - Use Sequential.add_modules() to add each layer of the network
    - Available layer type: Conv2d, MaxPool2d, Linear, CrossEntropyLoss, MSELoss, ReLU, Sigmoid, Softmax, Flatten...
    - Available classes: 1) ConcolutionalNN = Convolutional + fully connected network -> image input = (n° channels, width, heidth)
                         2) fullyConnectedNN = Fully connected network -> input = (int)
"""

from os.path import exists
import torchviz 

import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import time 
import json 

from dragonflai.utils.utils_model import *
from dragonflai.utils.utils_path import create_file_path

class NeuralNetwork(nn.Module):
    """Main Pytorch neural network class. 
    
    It acts as a superclass for which neural network subclasses that will describe specific architectures.
    Contains all tools to train, predict, save... a neural network model.
    
    Parameters:
        device (torch.device): sets the workload on CPU or GPU if available
        architecture (torch.nn.Sequential): Contains neural network model
           
    Use example:
        model = nnSubclass(input_size)
        model.print_architecture((1,input_size))
        model.fit(trainset, epoch)
        score = model.predict(testset)
        print(f"Test loss: {score}")
    """
    
    def __init__(self, modelType, taskType, name='NeuralNetwork', save_path="./results/tmp/"):
        super().__init__()
        self.use_gpu = torch.cuda.is_available()
        self.save_path   = save_path
        self.history     = dragonflAI_History(modelType, taskType)
        self.progressBar = dragonflAI_ProgressBar(self.history)
        #If available -> work on GPU
        self.device = torch.device('cuda:0' if self.use_gpu else 'cpu')
        print(f"Pytorch is setup on: {self.device}")
        
        self.architecture = nn.Sequential().to(self.device)
        self.model_name =  name
        
        
    def _compile(self, train_loader, test_loader, crit, lr, opts, scheduler, batch_size, epochs, **kwargs):
        # Check if save folder exists
        if not exists(self.save_path):
            os.makedirs(self.save_path)

        for _, sample in enumerate(train_loader):
            # get batch 
            inputs, targets = self.get_batch(sample=sample)
            _, output = self.loss_calculation(crit, inputs, targets, with_grad=False)
            if isinstance(inputs, tuple):
                self.input_shape = [inp.shape for inp in inputs]
            else:
                self.input_shape = inputs.shape 
            output_shape = output.shape  
            break
        self.opt       = []
        self.scheduler = []
        self.scaler    = []
        self.scaler.append(torch.cuda.amp.GradScaler(enabled=self.use_gpu))
        try:
            kwargs_optimizer = kwargs['kwargs_optimizer']
            self.opt.append(opts(self.architecture.parameters(), lr=lr, **kwargs_optimizer))
        except:
            self.opt.append(opts(self.architecture.parameters(), lr=lr))
        try:
            kwargs_scheduler = kwargs['kwargs_scheduler']
            self.scheduler.append(scheduler(self.opt[0], **kwargs_scheduler))
        except:
            pass 
            
        self.print_architecture(self.input_shape)
        
        self.init_results(train_loader, test_loader, batch_size, epochs)
        print('Training model {} during {} epochs with batch size set to {} on {} training data and validating on {} data'
              .format(self.model_name, self.history.parameters['nb_max_epochs'], 
                      self.history.parameters['batch_size'], 
                      self.history.parameters['batch_size'] * self.history.parameters['steps_per_epoch_train'], 
                      self.history.parameters['batch_size'] * self.history.parameters['steps_per_epoch_val'],
                      ))
        print('\ninput_shape {}  ====> {} {} ====> output_shape {}\n'.format(
            self.input_shape, self.history.modelType, self.history.taskType, output_shape
            ))
    
    

    def init_results(self, train_loader, test_loader, batch_size, epochs, *args, **kwargs):
        #Use GPU if available
        if self.use_gpu:
            print("CUDA compatible GPU found")
        else:
            print("No CUDA compatible GPU found")
        
        parameters = {
            'nb_max_epochs'        : epochs,
            'batch_size'           : batch_size,
            'dataset_size'         : len(train_loader) * batch_size,
            'steps_per_epoch_train': len(train_loader),
            'steps_per_epoch_val'  : len(test_loader),
            }
        self.history.set_new_parameters(parameters)
        
    def update_scheduler(self, *args, **kwargs):
        '''update scheduler'''
        loss = kwargs['loss']
        for scheduler in self.scheduler:
            scheduler.step(loss)

    def get_batch(self, *args, **kwargs):
        '''get batch : return a batch from loader containing input and target 
        input can be multimodal if your loader return [input_0, ... input_n], Y'''
        sample = kwargs['sample']

        available_floats = [torch.float, torch.float16, torch.float32, torch.float64, torch.double, torch.half]
        
        if isinstance(sample[0], list):
            if(sample[0][0].dtype in available_floats):
                input_dtype = torch.float32
            else:
                input_dtype = torch.int64
            inputs = ()
            for i in range(len(sample[0])): 
                inputs += (sample[0][i].to(self.device, dtype=input_dtype),)
        else:
            if(sample[0].dtype in available_floats):
                input_dtype = torch.float32
            else:
                input_dtype = torch.int64
            inputs = sample[0].to(self.device, dtype=input_dtype)
            
        if isinstance(sample[1], list):
            if(sample[1][0].dtype in available_floats):
                target_dtype = torch.float32
            else:
                target_dtype = torch.int64
            targets = ()
            for i in range(len(sample[1])): 
                targets += (sample[1][i].type(target_dtype).to(self.device),)
        else:
            if(sample[1].dtype in available_floats):
                target_dtype = torch.float32
            else:
                target_dtype = torch.int64
            targets = sample[1].type(target_dtype).to(self.device)
            
            
        return inputs, targets
            
    def loss_calculation(self, crit, inputs, target, *args, **kwargs):
        '''compute loss'''
        # get with_grad parameter 
        with_grad=kwargs['with_grad']
        if with_grad:
            # forward pass with gradient computing 
            outputs = self.forward(inputs)
            loss    = crit(outputs, target)
        else: 
            # forward pass without gradient 
            with torch.no_grad():
                outputs = self.forward(inputs)
                loss    = crit(outputs, target)
                torch.cuda.empty_cache()
        
        return loss, outputs

    def train_batch(self, *args, **kwargs):
        '''train a batch '''
        loss = kwargs['loss']
        
        #See here for detail about multiple scaler & optimizer
        # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-models-losses-and-optimizers
        
        # scaler scaling 
        # retain graph set to true in order to get a complete graph 
        # connecting input to output, even in multi modal cases 
        for idx, scaler in enumerate(self.scaler):
            retain_graph = (idx < len(self.scaler)-1)    
            scaler.scale(loss).backward(retain_graph=retain_graph)    
        
        # step optimizer with its scaler 
        for i in range(len(self.scaler)):
            self.scaler[i].step(self.opt[i])

        # update scaler 
        for scaler in self.scaler:
            scaler.update()
        
        # reset optimizers's gradients  
        for opt in self.opt:
            opt.zero_grad()

    def save_epoch_end(self, *args, **kwargs):
        if self.history.current_status['current_epoch'] % 100 == 0: #Save model every X epochs
            self.save_model(f"{self.save_path}/epoch{self.history.current_status['current_epoch']}")
            
        try:  
            if self.history.loss_train[-1] == np.min(self.history.loss_train):
                self.save_model("{}/{}_best_train".format(self.save_path, self.model_name))
            if self.history.loss_val[-1] == np.min(self.history.loss_val):
                self.save_model("{}/{}_best_val".format(self.save_path, self.model_name))
        except:
            pass 
        
    
    def fit(self, 
            train_set, 
            valid_set       = None,
            epochs          = 20, 
            criterion       = nn.L1Loss(),
            ):
        """Train a model on a training set
        print(f"Pytorch is setup on: {self.device}")

        
        Args:
            train_set (torch.utils.data.DataLoader): Training set used to fit the model. This variable contains batch size information + features + target 
            valid_set (torch.utils.data.DataLoader): Validation set used toverify overfitting when training the model. This variable contains batch size information + features + target 
            epochs (int): Amount of epochs to perform during training. Default is 20
            criterion (torch.nn): Criterion used during training for loss calculation (default = L1Loss() - see: https://pytorch.org/docs/stable/nn.html#loss-functions) 
        """
        # training starting callback 
        self._on_training_start()
        
        # iterate throw epochs 
        for _ in range(epochs):
            # starting epoch callback 
            self._on_epoch_start()
            # iterate throw batch 
            for _, sample in enumerate(train_set):
                # get batch 
                inputs, targets = self.get_batch(sample=sample)
                # forward batch with training 
                _, _ = self.forward_batch(inputs, targets, criterion, train=True)
            # update history 
            self.history._end_train_epoch(loss=np.mean(self.batch_loss), 
                                                lr=self.opt[0].param_groups[0]['lr'], 
                                                acc=self.acc)
            # predict on validation set 
            val_loss, _, _ = self.predict(valid_set, criterion=criterion)
            # update sheduler on validation set 
            self.update_scheduler(loss=val_loss)
            # ending epoch callback 
            self._on_epoch_end()
        # ending training callback 
        self._on_training_end()
        self.istrain = False
        return self.history


    def predict(self, test_set, criterion=nn.L1Loss()):
        """Use the trained model to predict a target values on a test set
        
        For now, we assume that the target value is known, so it is possible to calculate an error value.
        
        Args:
            test_set (torch.utils.data.DataLoader): Data set for which the model predicts a target value. This variable contains batch size information + features + target 
            criterion (torch.nn): Criterion used during training for loss calculation (default = L1Loss() - see: https://pytorch.org/docs/stable/nn.html#loss-functions) 

        Returns:
            mean_loss (float): the average error for all batch of data.
            output (list): Model prediction on the test set
            [inputs, targets] ([list,list]): Group of data containing the input + target of test set
        """
        # predict starting callback 
        self._on_predict_start()
        # init list 
        if isinstance(self.input_shape, list):
            self.inputs = [[] for i in range(len(self.input_shape))]
        else:
            self.inputs = []
        self.outputs, self.targets, self.test_loss = [],[],[]
        # iterate validation set 
        for _, sample in enumerate(test_set):
            # get batch 
            input, target = self.get_batch(sample=sample)
            # forward batch without training 
            _, _ = self.forward_batch(input, target, criterion, train=False)
        # predict ending callback 
        self._on_predict_end()
        
        return np.mean(self.test_loss), np.asarray(self.outputs), [np.asarray(self.inputs), np.asarray(self.targets)]


    def forward_batch(self, input, target, crit, train):
        # start batch callback 
        self._on_batch_start()
        # forward pass and get loss 
        loss, output = self.loss_calculation(crit, input, target, with_grad=train)
        # training mode 
        if train:
            # at first batch of first epoch : draw model in png 
            if self.history.current_status['current_epoch'] == 1 and \
                self.history.current_status['current_batch_train'] == 0: #Print network architecture
                #draw_graph(self, input_data=input, save_graph=True, directory=self.save_path, expand_nested=True, depth=5)
                torchviz.make_dot(output.mean(), 
                                  params=dict(self.architecture.named_parameters()), 
                                  show_attrs=True, show_saved=False).render('{}/architecture'.format(self.save_path), format='png')
            # add current batch loss 
            self.batch_loss.append(loss.cpu().item())
            # backward pass 
            self.train_batch(loss=loss)
            # update accuracy if needed 
            if self.history.taskType == taskType.CLASSIFICATION:
                self._update_acc(output, target)
            # update history 
            self.history._end_train_batch(lr=self.opt[0].param_groups[0]['lr'], 
                                            current_loss_train=np.mean(self.batch_loss),
                                            current_acc_train=self.acc)
        else: # validating mode 
            # add current test loss 
            self.test_loss.append(loss.item()) 
            # get input, target, ouput as array 
            if isinstance(input, tuple):
                self.inputs[0].extend(np.array(input[0].cpu().detach(), dtype=np.float32)) 
                self.inputs[1].extend(np.array(input[1].cpu().detach(), dtype=np.float32)) 
            else:
                self.inputs.extend(np.array(input.cpu().detach(), dtype=np.float32))
            self.targets.extend(np.array(target.cpu().detach(), dtype=np.float32))
            self.outputs.extend(np.array(output.cpu().detach(), dtype=np.float32))
            # update accuracy if needed 
            if self.history.taskType == taskType.CLASSIFICATION:
                self._update_acc(output, target, val=True)
            # update history 
            self.history._end_val_batch(current_loss_val=np.mean(self.test_loss), current_acc_val=self.acc_val)
        # ending batch callback 
        self._on_batch_end()
                    
        return loss, output 
        
    def forward(self, data):
        """Forward propagation.
        
        Note that this function can be overided by subclasses to add specific instructions.
        
        Args:
            data (array of shape (data_number, features_number)): data used for inference.
        
        Returns:
            target (array of shape (data_number, target_number))
        """

        return self.architecture(data)
        
        
    def save_model(self, path):
        """Save the model state in a json file
        
        If the folder specified does not exist, an error is sent
        If a file already exist, the saved file name is incremented 

        Args:
            path (string): file path without the extension
        """

        #Check if folder exists
        create_file_path(path)

        #Check if file exists
        iterator = 1
        while(exists(path + str(iterator) + ".json")):
            iterator+=1

        torch.save(self.architecture.state_dict(), path + "_" + str(iterator) + ".json")
        
        
    def load_model(self, path):    
        """Load a model from a file

        Args:
            path (string): file path to load without extension
        """
        try:
            self.architecture.load_state_dict(torch.load(path + ".json", map_location=self.device))
            self.architecture.to(self.device)
            print("Loaded model from disk")
        except Exception:
            raise Exception(f"Error when loading Neural Network model: {path} not found")
        

    def plot_learning_curve(self, loss_train, loss_val, path):
        """Plot the loss after training and save it in folder.

        Args:
            loss_train (list): loss values collected on train set
            loss_val (list): loss values collected on validation set
            path (str): File name including all path without the extension (example: my/folder/my_file)
        """
        fig = plt.figure()
        plt.plot(loss_train, color='blue')
        plt.plot(loss_val, color='red')

        plt.legend(["Training", "Validation"])

        plt.xlabel('epoch')
        plt.ylabel("loss")

        plt.grid(True)
        
        # Check if folder exists
        create_file_path(path)

        # Displaying the title
        plt.title("Loss evolution during neural network training")

        # Display / saving
        #plt.show()
        fig.savefig("{}.png".format(path))
        plt.close()
        
        
    def plot_learning_rate(self, lr, path):
        """Plot the loss after training and save it in folder.

        Args:
            loss_train (list): learning rate values during trainnig
            loss_val (list): loss values collected on validation set
            path (str): File name including all path without the extension (example: my/folder/my_file)
        """
        fig = plt.figure()
        plt.plot(lr, color='blue')

        plt.legend(["Learning rate"])

        plt.xlabel('epoch')
        plt.ylabel("learning rate")

        plt.grid(True)
        

        # Check if folder exists
        create_file_path(path)

        # Displaying the title
        plt.title("Learning rate evolution during neural network training")

        # Display / saving
        #plt.show()
        fig.savefig("{}.png".format(path))
        plt.close()
        
    def print_architecture(self, input_shape):
        """Display neural network architecture
        
        Note that the output size of each layer depends on the input shape given to the model (helps to get a good understansing in case of convolution layers)

        Args:
            input_shape (tensor of shape (batch_size, input_size)): Shape of the input normally given to the model.
        """
        
        print("\nNeural network architecture: \n")
        print(f"Input shape: {input_shape}")
        #summary(self, input_shape[0])
        print(self.architecture)
        print("\n")
        
        
        
        
    ########### Callback methods      
    def _save_end_results(self):
        if self.history.taskType == taskType.CLASSIFICATION:
            row = {'acc_train': self.history.acc_train[-1], 
                        'acc_val': self.history.acc_val[-1]}
        else:            
            row = {'acc_train': 0.0, 'acc_val': 0.0}
            
        row['loss_train'] = self.history.loss_train[-1]
        row['loss_val'] = self.history.loss_val[-1]

        json_object = json.dumps(row)
        
        # Writing to sample.json
        with open(self.save_path + '/end_train_results.json', 'w') as outfile:
            outfile.write(json_object)
        
    def _update_acc(self, output, target, val=False):
        classifications = torch.argmax(output, dim=1)
        if(len(target.shape) > 1):
            if(target.shape[1] > 1):
                # One hot encoded classification case
                target = torch.argmax(target, dim=1)
        correct_predictions = sum(classifications==target).item()
        if val:
            self.total_correct_val   += correct_predictions
            self.total_instances_val += self.history.parameters['batch_size']
            self.acc_val              = (self.total_correct_val/self.total_instances_val) * 100
        else:
            self.total_correct   += correct_predictions
            self.total_instances += self.history.parameters['batch_size']
            self.acc              = (self.total_correct/self.total_instances) * 100
                
    def _on_batch_end_time(self, *args, **kwargs):
        '''callback function, called at each batch's end'''
        curent_duration_t = time.time() - self.history.current_status['start_epoch_t']
        if self.history.current_status['current_batch_val'] == \
            self.history.parameters['steps_per_epoch_val']:
            self.history.set_current_status('duration_t', np.around(curent_duration_t, decimals=2))
        else:
            nb_batch_done   = self.history.current_status['current_batch_train'] + self.history.current_status['current_batch_val']
            total           = self.history.parameters['steps_per_epoch_train'] + self.history.parameters['steps_per_epoch_val']
            ratio           = nb_batch_done / total
            est             = curent_duration_t / ratio
            self.history.set_current_status('duration_t', np.around(est - curent_duration_t, decimals=2))
        
    def _on_epoch_start(self, *args, **kwargs):
        '''callback function, called at each epoch's start'''
        self.architecture.train() 
        self.batch_loss          = []
        self.total_correct       = 0
        self.total_instances     = 0
        self.acc                 = 0
        self.total_correct_val   = 0
        self.total_instances_val = 0
        self.acc_val             = 0
        self.history._start_epoch()
        self.progressBar.plot_log()
        
    def _on_predict_start(self, *args, **kwargs):
        '''callback function, called at each predict start'''
        self.architecture.eval()
        self.history.set_current_status('current_batch_test', 0)
        
    def _on_predict_end(self, *args, **kwargs):
        '''callback function, called at each predict end'''
        self.history._end_val_epoch(loss=np.mean(self.test_loss), acc=self.acc_val) 
    
    def _on_epoch_end(self, *args, **kwargs):
        '''callback function, called at each epoch's end'''
        self.save_epoch_end() 
    
    def _on_batch_start(self, *args, **kwargs):
        '''callback function, called at each batch's start'''
        pass 
    
    def _on_batch_end(self, *args, **kwargs):
        '''callback function, called at each batch's end'''
        self._on_batch_end_time()
        self.progressBar.plot_log()
    
    def _on_training_start(self, *args, **kwargs):
        '''callback function, called at training start'''
        print('\tStart training...') 
    
    def _on_training_end(self, *args, **kwargs):
        '''callback function, called at training end'''
        print('\tEnd training...')
        self._save_end_results()
