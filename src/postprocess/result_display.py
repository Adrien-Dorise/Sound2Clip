"""
 Display the result of a classification model with a confusion matrix
 Author: Adrien Dorise (adrien.dorise@hotmail.com), Edouard Villain - LR Technologies
 Created: June 2024
 Last updated: Adrien Dorise - June 2024
"""

import torch 
import numpy as np 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def confusion_matrix(model, dataset, save_path):
        """Computes and visualizes a confusion matrix to evaluate the performance of a classification model.

        Args:
            model (NeuralNetwork or MachineLearning object from DragonflAI): Classification model to evaluate with a confusion matrix 
            dataset (DataLoader): dataset used to create the confusion matrix. 
            
        """

        pred, (_, target) = model.predict(dataset)
        cm = confusion_matrix(target, torch.tensor(pred))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        save_file = f"{save_path}/confusion_matrix_{model.model_name}.png" 
        disp.plot().figure_.savefig(save_file)
        print("Confusion matrix saved in {}".format(save_path))
        plt.close()