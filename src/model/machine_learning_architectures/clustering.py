"""
This is DragonflAI Clustering script
Last Update by Adrien Dorise - May 2024

This package contains all clustering methods available in DragonflAI
The clustering class comes from the MachineLearning class that takes care of the API. 
Scikit-learn is the main API used.
Author: Adrien Dorise (adrien.dorise@hotmail.com) - LR Technologies
Created: May 2024
"""

from dragonflai.model.machineLearning import MachineLearning
from sklearn import cluster
import sklearn.metrics as metrics
from enum import Enum
import numpy as np

class Clustering_Models(Enum):
    KMEANS = 1
    MEAN_SHIFT = 2
    HIERARCHICAL_CLUSTERING = 3
    DBSCAN = 4
    
    
class Clustering(MachineLearning):   
    def __init__(self, model, loss_metric=metrics.mean_absolute_error, verbose=False, save_path="./results/tmp/",
                 Kmeans_param = [5, "k-means++", 300], # n_clusters, initialisation, max_iter
                 hierarchical_param = [2, "euclidean"], # n_clusters, metric  
                 DBSCAN_param = [0.5, 5, "euclidean"]): # epsilon, min_samples, metric
        """
        Initialise the model with desired algorithm
        
        Args:
            model (Clustering_Models enum): algorithm selection
            loss_metric (sklearn.metrics): metrics used for loss calculation when evaluating fit or predict. It is also used for gridSearch. List on https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            verbose (bool): Put to true to print details during training. Default to False.
            *args: Parameters of each specific algorithm
        """
        MachineLearning.__init__(self, model, loss_metric, output_size=1, save_path=save_path)

        if model == Clustering_Models.KMEANS: #K-Means
                self.model_name = "KMeans_clustering"
                self.model.append(cluster.KMeans(n_clusters=Kmeans_param[0], init=Kmeans_param[1], max_iter=Kmeans_param[2])) 
        elif model == Clustering_Models.MEAN_SHIFT: #Mean shift
                self.model_name = "mean_shift_clustering"
                self.model.append(cluster.MeanShift())
        elif model == Clustering_Models.HIERARCHICAL_CLUSTERING: #Hierachical clustering
            self.model_name = "hierarchical_clustering"
            self.model.append(cluster.AgglomerativeClustering(n_clusters=hierarchical_param[0], metric=hierarchical_param[1]))
        elif model == Clustering_Models.DBSCAN: #DBSCAN
            self.model_name = "DBSCAN_clustering"
            self.model.append(cluster.DBSCAN(eps=DBSCAN_param[0], min_samples=DBSCAN_param[1], metric=DBSCAN_param[2])) 

    
    def forward(self, data):
        """Inference phase

        Args:
            data (array): features used to get a prediction

        Returns:
            array: prediction
        """
        results = []
        for mod in self.model:
            results.append(mod.labels_)
        return np.array(results).reshape(len(results[0]),-1)