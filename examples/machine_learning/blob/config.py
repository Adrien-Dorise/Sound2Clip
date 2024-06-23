"""
 Parameters for machine learning applications in the blob example
 Author: Adrien Dorise (adrien.dorise@hotmail.com) - LR Technologies
 Created: May 2024 
 Last updated: Adrien Dorise - May 2024
"""

from experiment import Model_Type
from dragonflai.model.machine_learning_architectures.classification import Classification, Classification_Models
from dragonflai.model.machine_learning_architectures.clustering import Clustering, Clustering_Models
from dragonflai.model.machine_learning_architectures.regressor import Regressor, Regression_Models



# Path parameters
save_path = r"examples/machine_learning/blob/outputs/"

# Data parameters
list_N = [100,100,100] # Number of points per class
list_centroid = [[-1,-1],[1,1],[0.5,-0.5]]


# Machine learning models parameters
classifications_to_test = []
for model in Classification_Models:
    classifications_to_test.append(Classification(model))

clusterings_to_test = []
for model in Clustering_Models:
    clusterings_to_test.append(Clustering(model))

regression_to_test = []  
for model in Regression_Models:
    regression_to_test.append(Regressor(model))



