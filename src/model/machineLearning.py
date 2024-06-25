"""
This is DragonflAI Machine learning main script
Last Update by Adrien Dorise - May 2024

This package references the default behaviour for all machine learning models.
It is done so it can called similarly as the neural network package of DragonflAI
Scikit-learn is the main API used.
Author: Adrien Dorise (adrien.dorise@hotmail.com) - LR Technologies 
Created: Feb 2023
"""

from dragonflai.utils.utils_path import create_file_path
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
import joblib
import numpy as np
import pandas as pd
import os


    


class MachineLearning():
    def __init__(self, model, loss_metric=metrics.mean_absolute_error, output_size=1, save_path="./results/tmp/"):
        self.metric = loss_metric
        self.choice = model
        self.output_size = output_size
        self.model = []
        self.model_name = ""
        self.save_path = save_path

    def _compile(self, **kwargs):
        # Check if save folder exists
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        print('\nCompiling machine learning model {}'.format(self.choice))
        print('Output size is {}'.format(self.output_size))


    def fit(self, train_set, **kwargs):
        """Train a model on a training set
        
        Note that most of scikit learn models do not work with mini-batch training.
        For now, the whole data set is taking for the training: No epochs, no batches
        
        Args:
            train_set (torch.utils.data.DataLoader): Training set used to fit the model. This variable contains batch size information + features + target 
        """

        print(f"\nTraining <{self.choice}> START")
        batch_loss = []

        inputs,target = self.extract_set(train_set)
        
        #Training
        if(len(self.model)==1): #Only one output or multi output model
            self.model[0].fit(inputs,target)
        else:
            modelIter = 0
            for mod in self.model:
                mod.fit(inputs, target[:,modelIter])
                modelIter+=1

            
        #Get loss
        outputs = self.forward(inputs)
        loss = self.metric(target,outputs)
        print(f"Training loss is: {loss}")

        print('Finished Training')
        
    
    def predict(self, test_set, **kwargs):
        """Use the trained model to predict a target values on a test set. The target must be available to calculate score.
        If no target available, use forward method.
        
        For now, we assume that the target value is known, so it is possible to calculate an error value.
        
        Args:
            test_set (torch.utils.data.DataLoader): Data set for which the model predicts a target value. This variable contains batch size information + features + target 

        Returns:
            mean_loss (float): the average error for all batch of data.
            output (list): Model prediction on the test set
            [inputs, targets] ([list,list]): Group of data containing the input + target of test set
        """

        inputs, target = self.extract_set(test_set)
        # forward
        outputs = self.forward(inputs)
        return outputs, [inputs.detach().numpy(), target.detach().numpy()]
    
    
    def forward(self, data):
        """Inference phase

        Args:
            data (array): features used to get a prediction

        Returns:
            array: prediction
        """
        results = []
        for mod in self.model:
            results.append(mod.predict(data))
        return np.array(results).reshape(len(results[0]),-1)
    
    
    
    def grid_search(self, parameters, train_set, verbose=1, parallel_jobs=-1, save_path="models/paramSearch/"):
        """Perform a parameter search of the model using Kfold cross validation.
        The search object is then save into a joblib object as well as a csv file.
    

        Args:
            parameters (dict): Set of parameters to search
            train_set (DataLoader): data set used to search the parameters
            verbose (int, optional): Controls the verbosity between [0,3]. Defaults to 1.
            parallel_jobs (int, optional): Number of jobs in parallel. -1 means all processors. Defaults to -1.
            save_path (str, optional): Path to save the search. Defaults to "models/paramSearch/".
        Return:
            Return the search object.
        """
        
        searchResults = []
        modelIter = 0
        inputs,target = self.extract_set(train_set)
        
        for mod in self.model:
            print(f"Grid search for model{modelIter} of {self.choice}")
            search = GridSearchCV(mod, parameters, verbose=verbose, n_jobs=parallel_jobs)
            if(len(self.model)==1): #Only one output or multi output model
                search.fit(inputs, target)
            else:
                search.fit(inputs, target[:,modelIter])
                
            joblib.dump(search,f"{save_path}{self.choice}{modelIter}.sav")
            pd.DataFrame(search.cv_results_).to_csv(f"{save_path}{self.choice}{modelIter}.csv")
            searchResults.append(search)
            modelIter+=1
        
        print("\nGrid Search best params found:")
        print(f"model: {self.model_name}")
        for res in searchResults:
            print(f"Score: {res.best_score_} / Params: {res.best_params_}\n")
            
        return searchResults
        
    def extract_set(self,dataset):
        """Extract features and targets from a dataLoader object
        Tool function used by other function of the class (fit, predict, gridSearch)
    
        Args:
            dataset (DataLoader): DataLoader obejct containing a data set

        Raises:
            Warning: Most scikit-learn models do not implement mini-batch training. Therefore, mini-batch is disable for this class. 
                If the DataLoader contain multiple batch, an error is raised.

        Returns:
            feature set (array)
            target set (array)
        """
        for i, sample in enumerate(dataset, 0):
            if(dataset.batch_size != len(dataset.dataset)):
                raise Warning(f"The number of batch exceed 1. This program does not support multi-batch processing. Make sure that batch_size is equal to the number of inputs.\nBatch_size / input: [{dataset.batch_size} / {len(dataset.dataset)}]")
            else:
                # get the inputs; data is a list of [inputs, target]
                return sample[0],sample[1]

    def print_architecture(self, **kwargs):
        """
        Display model informations
        """
        print("\nMachine learning model information: ")
        print(f"output size: {self.output_size}")
        print(self.model)

    def save_model(self, path):
        """Save the model state in a sav file
        The model type is added to the file name
        
        If the folder specified does not exist, an error is sent
        If a file already exist, the existing file is erased

        Args:
            path (string): file path without the model type and extension
        """

        #Check if folder exists
        create_file_path(path)

        iter = 0
        for mod in self.model:
            joblib.dump(mod,f"{path}_{self.choice}{iter}.sav")
            iter+=1
    
    def load_model(self, path):
        """Load a model from a .sav file

        Args:
            path (string): file path without the model type and extension (ex: modelFoder/name instead of modelFolder/name_KNN1.sav)
        """
        iter = 0
        try:
            for i in range(len(self.model)): #Can't use enumarator call as it creates a copy of self.model
                self.model[i] = joblib.load(f"{path}_{self.choice}{iter}.sav")
                iter+=1
        except Exception:
            raise Exception(f"Error when loading Machine Learning model: {path}_{self.choice}{iter}.sav not found")    
