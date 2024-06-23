'''
Experiment class to train machine learning algorithm on a blob dataset.
Author: Adrien Dorise (adrien.dorise@hotmail.com) - LR Technologies
Created: May 2023
Last updated: Adrien Dorise - May 2023

'''
import torch 
import pickle
import numpy as np 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd


class Experiment_Grid_Search():
    def __init__(self, model_list,
                param_list,
                train_set,
                save_path,
                ):
        
        #Paths
        self.save_path = save_path

        #Dataset
        self.train_set = train_set

        #Model parameters  
        self.model_list = model_list
        self.param_list = param_list
        self.model = None
    
    def grid_search(self):
        """Start a grid search on the given set of parameters and models
        The results are saved in multiple csv.
        Each model has a detail csv for all the parameters tested
        A more general csv gives a summary of all the best parameters for each model.
        """
        results = []
        for i in range(len(self.model_list)):
            search = self.model_list[i].grid_search(self.param_list[i], self.train_set, save_path=self.save_path, verbose=0)
            results.append(search)
        
        print("\nBest score for each models after grid search:")
        best_results = []
        for i in range(len(results)):
            for j in range(len(results[i])): # The results of a grid search are given in a list form (in case multiple models are used).
                print(f"model: {self.model_list[i].model_name} / best score: {results[i][j].best_score_} / parameters: {results[i][j].best_params_}")
                best_results.append([self.model_list[i].model_name, results[i][j].best_score_, results[i][j].best_params_])
        
        best_results = pd.DataFrame(best_results, columns=["Model", "Score", "Parameters"])
        best_results.to_csv(f"{self.save_path}/grid_search_summary.csv", index=False)
        print(f"Grid search result saved in {self.save_path}/grid_search_summary.csv")
        
       
    def save(self, filename):
        """Save the whole experiment class as a pickle object.

        Args:
            filename (string): Path to save the experiment status
        """
        with open(filename, 'wb') as file:
            try:
                pickle.dump(self, file)
            except EOFError:
                raise Exception("Error in save experiment: Pickle was not able to save the file.")

    @classmethod
    def load(self, filename):
        """Load a pickle object to an Experiment class Python variable
        This is a class method. It means that a reference to the class is NOT necessary to call this method. Simply type <your_experiment = Experiment.load(filename)>

        Args:
            filename (string): Path to the pickle saved object.
        """
        with open(filename, 'rb') as file:
            try:
               return pickle.load(file)
            except EOFError:
                raise Exception("Error in load experiment: Pickle was not able to retrieve the file.")


class Experiment():
    def __init__(self, ML_model,
                train_set,
                test_set,
                save_path,
                ):
        
        #Paths
        self.save_path = save_path

        #Dataset
        self.train_set = train_set
        self.test_set = test_set

        #Model parameters  
        self.model = ML_model
            
       
    def fit(self):
        """Train the model using the data available in the train set.
        """
        self.model._compile()
        self.model.fit(self.train_set)
        

    def predict(self):          
        """Model prediction on the samples available in the test set

        """
        # !!! Data loading !!!
        
        output, (_,target) = self.model.predict(self.test_set)

        print("\nClassification results\nMachine learning example on mnist dataset:")
        for i in range(len(output)):
                print(f"sample {i}: target={target[i]} / prediction={output[i]}")
        
        return output

    def visualise(self):
        """Visualisation of the first picture of the test set.
        The input + predicted images are both shown.
        """
        
        pred, (_, target) = self.model.predict(self.test_set)
        cm = confusion_matrix(target, torch.tensor(pred))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        save_file = f"{self.save_path}/confusion_matrix_{self.model.model_name}.png" 
        disp.plot().figure_.savefig(save_file)
        print("Confusion matrix saved in {}".format(self.save_path))
        plt.close()

    
    def save(self, filename):
        """Save the whole experiment class as a pickle object.

        Args:
            filename (string): Path to save the experiment status
        """
        with open(filename, 'wb') as file:
            try:
                pickle.dump(self, file)
            except EOFError:
                raise Exception("Error in save experiment: Pickle was not able to save the file.")

    @classmethod
    def load(self, filename):
        """Load a pickle object to an Experiment class Python variable
        This is a class method. It means that a reference to the class is NOT necessary to call this method. Simply type <your_experiment = Experiment.load(filename)>

        Args:
            filename (string): Path to the pickle saved object.
        """
        with open(filename, 'rb') as file:
            try:
               return pickle.load(file)
            except EOFError:
                raise Exception("Error in load experiment: Pickle was not able to retrieve the file.")