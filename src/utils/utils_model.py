"""
This package references all utils neural network classes used in the application.
Author: Adrien Dorise (adrien.dorise@hotmail.com), Edouard Villain (evillain@lrtechnologies.fr) - LR Technologies
Created: April 2024
Last updated: Edouard Villain - April 2024
"""

import os 
import time 
#### ENUM Classes  

from enum import Enum
class modelType(Enum):
    '''Enumeration for model type'''
    NEURAL_NETWORK   = 1
    MACHINE_LEARNING = 2
    
class taskType(Enum):
    '''Enumeration for task type'''
    REGRESSION     = 1
    CLASSIFICATION = 2
    SEGMENTATION   = 3
    CLUSTERING     = 4
    
class dragonflAI_callback():
    '''callback implementation class'''
    def __init__(self):
        pass 
    
    def _on_epoch_start(self):
        pass 
    
    def _on_epoch_end(self):
        pass 
    
    def _on_batch_start(self):
        pass 
    
    def _on_batch_end(self):
        pass 
    


#### Memory and History classes  
class dragonflAI_History():
    '''Classe used for storing training history'''
    def __init__(self, modelType=modelType.NEURAL_NETWORK, 
                 taskType=taskType.REGRESSION):
        '''init default memory '''
        
        self.modelType = modelType
        '''modelType -> NEURAL_NETWORK or MACHINE_LEARNING'''
        self.taskType  = taskType
        '''taskType -> REGRESSION or CLASSIFICATION or SEGMENTATION'''
        
        self.parameters = {
            'nb_max_epochs'        : 0,
            'batch_size'           : 0,
            'dataset_size'         : 0,
            'steps_per_epoch_train': 0,
            'steps_per_epoch_val'  : 0,
            
        }
        '''parameters -> dict of unchanged parameters during training'''

        self.parameters_ML = {
            
        }
        
        self.current_status = {
            'current_epoch'      : 0,
            'current_batch_train': 0,
            'current_batch_val'  : 0,
            'duration_t'         : 0.0,
            'start_epoch_t'      : 0.0,
            'istrain'            : True,
            'current_loss_train' : 0.0,
            'current_loss_val'   : 0.0,
            'current_acc_train'  : 0.0,
            'current_acc_val'    : 0.0,
            'current_lr'         : 0.0
        }
        '''curent_status -> dict of current status'''
        
        # history 
        self.loss_train = []
        '''losses_train -> list of epoch's mean(loss) on training data'''
        self.loss_val   = []
        '''losses_val -> list of epoch's mean(loss) on validation data'''
        self.acc_train    = []
        '''acc_train -> list of epoch's accuracies on training data'''
        self.acc_val      = []
        '''acc_val -> list of epoch's accuracies on validation data'''
        self.lr           = []
        '''self.lr -> list of lr at the end of epoch'''
        self.verbosity    = 1
        '''verbosity -> printing level'''
        
    def set_current_status(self, key, values): 
        '''set a new information into current_status'''
        self.current_status[key] = values
        
    def set_parameter(self, key, value):
        '''set_parameter : set a parameter in self.parameters'''
        self.parameters[key] = value
        
    def set_new_parameters(self, new_parameters):
        '''set_new_parameters : set new parameters in self.parameters'''
        self.parameters = new_parameters
        
    def _start_epoch(self):
        '''_start_epoch : callback at the epoch start in order to update history'''
        self.current_status['current_epoch']       += 1
        self.current_status['start_epoch_t']        = time.time()
        self.current_status['current_batch_train']  = 0
        self.current_status['current_batch_val']    = 0
        self.current_status['current_acc_train']    = 0
        self.current_status['current_acc_val']      = 0
        self.current_status['current_loss_train']   = 0
        self.current_status['current_batch_train']  = 0
        self.current_status['current_loss_val']     = 0
    
    def _end_train_batch(self, lr, current_loss_train, current_acc_train):
        '''_end_train_batch : callbac at the end of training batch in order to update history'''
        self.current_status['current_batch_train'] += 1
        self.current_status['current_lr']           = lr
        self.current_status['current_loss_train']   = current_loss_train
        self.current_status['current_acc_train']    = current_acc_train
    
    def _end_val_batch(self, current_loss_val, current_acc_val):
        '''_end_train_batch : callbac at the end of validation batch in order to update history'''
        self.current_status['current_batch_val'] += 1
        self.current_status['current_loss_val']   = current_loss_val
        self.current_status['current_acc_val']    = current_acc_val
        
    
    def _end_train_epoch(self, loss, lr, acc=0.0):
        '''_end_train_epoch : update history at the end of epoch before validation'''
        self.loss_train.append(loss)
        self.lr.append(lr)
        if self.taskType == taskType.CLASSIFICATION:
            self.acc_train.append(acc)
            
    def _end_val_epoch(self, loss, acc=0.0):
        '''_end_val_epoch : update history at the end of epoch before validation'''
        self.loss_val.append(loss)
        if self.taskType == taskType.CLASSIFICATION:
            self.acc_val.append(acc)
        

class dragonflAI_ProgressBar():
    '''Custom progress bar'''
    def __init__(self, history):
        self.history  = history
        self._before  = '='
        self._current = '>'
        self._after   = ' '
        self._start   = '['
        self._end     = ']'
        
    def set_custom_cursor(self, start, before, current, after, end):
        self._start   = start
        self._before  = before
        self._current = current
        self._after   = after
        self._end     = end
        
    def print_acc(self, lr, size_bar, current, est_t, end):
        print('[{:4d}/{:4d}, {:4d}/{:4d}, {:4d}/{:4d}] : {}{}{}{}{} : lr = {:.3e} - acc = {:05.2f} % - val_acc = {:05.2f} % - {}  '\
                    .format(self.history.current_status['current_epoch'],self.history.parameters['nb_max_epochs'],
                            self.history.current_status['current_batch_train'], 
                            self.history.parameters['steps_per_epoch_train'],
                            self.history.current_status['current_batch_val'], self.history.parameters['steps_per_epoch_val'],
                            self._start, self._before * current, self._current, self._after * (size_bar - current), self._end, 
                            lr, 
                            self.history.current_status['current_acc_train'], 
                            self.history.current_status['current_acc_val'], est_t), end=end)
    
    def print_loss(self, lr, size_bar, current, est_t, end):
        print('[{:4d}/{:4d}, {:4d}/{:4d}, {:4d}/{:4d}] : {}{}{}{}{} : lr = {:.3e} - loss = {:.3e} - val = {:.3e} - {}  '\
                    .format(self.history.current_status['current_epoch'],self.history.parameters['nb_max_epochs'],
                            self.history.current_status['current_batch_train'], 
                            self.history.parameters['steps_per_epoch_train'],
                            self.history.current_status['current_batch_val'], self.history.parameters['steps_per_epoch_val'],
                            self._start, self._before * current, self._current, self._after * (size_bar - current), self._end, 
                            lr, 
                            self.history.current_status['current_loss_train'], 
                            self.history.current_status['current_loss_val'], est_t), end=end)
        
    def plot_log(self, *args, **kwargs):
        '''plot log during training'''
        if self.history.verbosity > 0:
            try:
                column, _ = os.get_terminal_size()
            except:
                column = 175 
            verbose = self.history.verbosity
            
            if self.history.current_status['current_batch_val'] == \
                self.history.parameters['steps_per_epoch_val']:
                verbose = 2
                
            if not self.history.current_status['istrain']:
                lr = 0.0
            else:
                lr = self.history.current_status['current_lr']
                    
            size_bar = column - 130
            current  = (size_bar * 
                        (self.history.current_status['current_batch_train'] + self.history.current_status['current_batch_val']) 
                        // (self.history.parameters['steps_per_epoch_train'] + self.history.parameters['steps_per_epoch_val'] ))
            end      = '\r'
            
            if self.history.current_status['current_batch_val'] == \
                self.history.parameters['steps_per_epoch_val']:
                est_t = 'time used = {} s.'.format(self.history.current_status['duration_t'])
            else:
                est_t = 'time left ~ {} s.'.format(self.history.current_status['duration_t'])
            
            if verbose == 2:
                end = '\n'
            if (verbose == 2 and self.history.verbosity == 3) or self.history.verbosity in [1, 2]:
                if self.history.taskType in [taskType.REGRESSION, taskType.SEGMENTATION]:
                    self.print_loss(lr, size_bar, current, est_t, end)
                if self.history.taskType == taskType.CLASSIFICATION:
                    self.print_acc(lr, size_bar, current, est_t, end)