"""
This is DragonflAI Classification script
Last Update by Adrien Dorise - May 2024

This package contains all classification methods available in DragonflAI
The Classification class comes from the MachineLearning class that takes care of the API. 
Scikit-learn is the main API used.
Author: Adrien Dorise (adrien.dorise@hotmail.com) - LR Technologies
Created: May 2024
"""

from dragonflai.model.machineLearning import MachineLearning
from sklearn import linear_model, svm, neighbors, tree, ensemble
import sklearn.metrics as metrics
from enum import Enum

class Classification_Models(Enum):
    SGD = 1
    SVM = 2
    KNN = 3
    DECISION_TREE = 4
    RANDOM_FOREST = 5
    ADA_BOOST = 6
    GBOOST = 7

class Classification(MachineLearning):   
    def __init__(self, model, loss_metric=metrics.mean_absolute_error, output_size=1, verbose=False, save_path="./results/tmp/",
                 SGD_param = ['squared_error', 'l1',0.0001,'optimal'], # loss, penalty, alpha, learning_rate
                 SVM_param = ["rbf",1,0.0,'auto'], # kernel, C, coef0, gamma
                 KNN_param = [3,1], #n_neighbors, p
                 tree_param = ["gini",500], # criterion, max_depth
                 forest_param = ["gini",50,100], # criterion, max_depth, n_estimators
                 AdaBoost_param = [None, 50, 1],  # estimator, n_estimator, learning_rate
                 GBoost_param = [50, 0.5] # n_estimator, learning_rate
                 ):
        """
        Initialise the model with desired algorithm
        
        Args:
            model (Classification_Models enum): algorithm selection
            loss_metric (sklearn.metrics): metrics used for loss calculation when evaluating fit or predict. It is also used for gridSearch. List on https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            output_size (int): Number of outputs to predict. Default to 1
            verbose (bool): Put to true to print details during training. Default to False.
            *args: Parameters of each specific algorithm
        """
        MachineLearning.__init__(self, model, loss_metric, output_size, save_path)

        if model == Classification_Models.SGD: #Stochastic Gradient Descent
            for i in range(self.output_size):
                self.model_name = "SGD_classification"
                self.model.append(linear_model.SGDClassifier(loss=SGD_param[0], penalty=SGD_param[1], alpha=SGD_param[2],  learning_rate=SGD_param[3], verbose=verbose)) 
        elif model == Classification_Models.SVM: #Kernel support vector machines
            self.model_name = "SVM_classification"
            for i in range(self.output_size):
                self.model.append(svm.SVC(kernel=SVM_param[0], C=SVM_param[1], coef0=SVM_param[2], gamma=SVM_param[3],verbose=verbose)) 
        elif model == Classification_Models.KNN: #K-Nearest Neighbors
            self.model_name = "KNN_classification"
            self.model.append(neighbors.KNeighborsClassifier(n_neighbors=KNN_param[0], p=KNN_param[1]))
        elif model == Classification_Models.DECISION_TREE: #Decision Tree
            self.model_name = "decision_tree_classification"
            self.model.append(tree.DecisionTreeClassifier(criterion=tree_param[0], max_depth=tree_param[1]))
        elif model == Classification_Models.RANDOM_FOREST: #Random forest
            self.model_name = "random_forest_classification"
            self.model.append(ensemble.RandomForestClassifier(criterion=forest_param[0], max_depth=forest_param[1], n_estimators=forest_param[2],verbose=verbose))
        elif model == Classification_Models.ADA_BOOST: #AdaBoost
            self.model_name = "ADA_boost_classification"
            for i in range(self.output_size):
                self.model.append(ensemble.AdaBoostClassifier(estimator=AdaBoost_param[0],n_estimators=AdaBoost_param[1], learning_rate=AdaBoost_param[2]))
        elif model == Classification_Models.GBOOST: #Gradient tree boosting GBoost
            self.model_name = "GBoost_classification"
            for i in range(self.output_size):
                self.model.append(ensemble.GradientBoostingClassifier(n_estimators=GBoost_param[0], learning_rate=GBoost_param[1],verbose=verbose))

