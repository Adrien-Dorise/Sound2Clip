'''
Experiment class to train machine learning algorithm on a blob dataset.
Author: Adrien Dorise (adrien.dorise@hotmail.com) - LR Technologies
Created: May 2023
Last updated: Adrien Dorise - May 2023

'''
from enum import Enum

import torch 
import pickle
import numpy as np 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class Model_Type(Enum):
    REGRESSION  = 1
    CLASSIFICATION = 2
    CLUSTERING = 3


class Experiment():
    def __init__(self, model,
                train_set,
                test_set,
                model_type,
                save_path,
                ):
        
        #Paths
        self.save_path = save_path

        #Dataset
        self.train_set = train_set
        self.test_set = test_set

        #Model parameters  
        self.model = model
        self.model_type = model_type
        self.criterion = self.model.metric
         
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

        print("\nPrediction results\nMachine learning example on blob dataset:")
        for i in range(len(output)):
                print(f"sample {i}: target={target[i]} / prediction={output[i]}")
        
        return output




    def visualise(self):
        """Visualisation of the first picture of the test set.
        The input + predicted images are both shown.
        """
        if(self.model_type is Model_Type.CLUSTERING):
            feat, _ = self.model.extract_set(self.train_set)
            labels = self.model.model[0].labels_
            plt.scatter(feat[:,0] , feat[:,1], c=labels.astype(int))
            plt.grid(True)
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title(f"Clustering on blob data points with {self.model.model_name}")
            save_file = f"{self.save_path}/blob_{self.model.model_name}.png" 
            plt.savefig(save_file)
            plt.close()
        
        if (self.model_type is Model_Type.CLASSIFICATION):
            pred, (_, target) = self.model.predict(self.test_set)
            print('\nCONFUSION MATRIX:')
            cm = confusion_matrix(target, torch.tensor(pred))
            print(cm)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            save_file = f"{self.save_path}/confusion_matrix_{self.model.model_name}.png" 
            disp.plot().figure_.savefig(save_file)
            print("Confusion matrix saved in {}".format(self.save_path))
            plt.close()
            
        if (self.model_type is Model_Type.REGRESSION):
            pred, (input, target) = self.model.predict(self.test_set)
            plt.scatter(input, pred, color="blue", marker='x', label="Predictions")
            plt.scatter(input, target, color="green", marker='o', label="Targets")
            plt.legend()
            plt.grid(True)
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title(f"Regresion on blob data points with {self.model.model_name}")
            save_file = f"{self.save_path}/prediction_blob_{self.model.model_name}.png" 
            plt.savefig(save_file)
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
