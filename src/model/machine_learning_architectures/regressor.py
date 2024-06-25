"""
This is DragonflAI Regressor script
Last Update by Adrien Dorise - May 2024

This package contains all regression methods available in DragonflAI
The regressor class comes from the MachineLearning class that takes care of the API. 
Scikit-learn is the main API used.
Author: Adrien Dorise (adrien.dorise@hotmail.com) - LR Technologies
Created: May 2024
"""

from dragonflai.model.machineLearning import MachineLearning
from sklearn.linear_model import BayesianRidge, SGDRegressor
from sklearn import svm, neighbors, tree, ensemble
import sklearn.metrics as metrics
from enum import Enum

class Regression_Models(Enum):
    BAYES_LINEAR = 1
    SGD = 2
    SVM = 3
    KNN = 4
    DECISION_TREE = 5
    RANDOM_FOREST = 6
    ADA_BOOST = 7
    GBOOST = 8


class Regressor(MachineLearning):
    
    
    def __init__(self, model, loss_metric=metrics.mean_absolute_error, output_size=1, verbose=False, save_path="./results/tmp/",
                 bayes_param = [1e-05,1e-05,1e-05,1e-05], 
                 SGD_param = ['squared_error', 'l1',0.0001,'optimal'], 
                 SVM_param = ["rbf",1,0.1,0.5,'auto'], 
                 KNN_param = [3,1], 
                 tree_param = ["absolute_error",500], 
                 forest_param = ["squared_error",50,100], 
                 AdaBoost_param = [None, 50, 1, "linear"], 
                 GBoost_param = [50, 0.5, "squared_error"]):
        """
        Initialise the model with desired algorithm
        
        Args:
            model (Regression_Models enum): algorithm selection
            loss_metric (sklearn.metrics): metrics used for loss calculation when evaluating fit or predict. It is also used for gridSearch. List on https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            verbose (bool): Put to true to print details during training. Default to False.
            output_size (int): Number of output to predict with the regression model. Default to 1
            *args: Parameters of each specific algorithm
        """
        MachineLearning.__init__(self, model, loss_metric, output_size, save_path=save_path)

        if model == Regression_Models.BAYES_LINEAR: #Bayesian regression
            self.model_name = "bayesian_regression"
            for i in range(self.output_size):
                self.model.append(BayesianRidge(alpha_1=bayes_param[0], alpha_2=bayes_param[1], lambda_1=bayes_param[2], lambda_2=bayes_param[3], verbose=verbose))
        elif model == Regression_Models.SGD: #Stochastic Gradient Descent
            self.model_name = "SGD_regression"
            for i in range(self.output_size):
                self.model.append(SGDRegressor(loss=SGD_param[0], penalty=SGD_param[1], alpha=SGD_param[2],  learning_rate=SGD_param[3], verbose=verbose)) 
        elif model == Regression_Models.SVM: #Kernel support vector machines
            self.model_name = "SVM_regression"
            for i in range(self.output_size):
                self.model.append(svm.SVR(kernel=SVM_param[0], C=SVM_param[1], epsilon=SVM_param[2], coef0=SVM_param[3], gamma=SVM_param[4],verbose=verbose)) 
        elif model == Regression_Models.KNN: #K-Nearest Neighbors
            self.model_name = "KNN_regression"
            self.model.append(neighbors.KNeighborsRegressor(n_neighbors=KNN_param[0], p=KNN_param[1]))
        elif model == Regression_Models.DECISION_TREE: #Decision Tree
            self.model_name = "decision_tree_regression"
            self.model.append(tree.DecisionTreeRegressor(criterion=tree_param[0], max_depth=tree_param[1]))
        elif model == Regression_Models.RANDOM_FOREST: #Random forest
            self.model.append(ensemble.RandomForestRegressor(criterion=forest_param[0], max_depth=forest_param[1], n_estimators=forest_param[2],verbose=verbose))
            self.model_name = "random_forest_regression"
        elif model == Regression_Models.ADA_BOOST: #AdaBoost
            self.model_name = "ADA_boost_regression"
            for i in range(self.output_size):
                self.model.append(ensemble.AdaBoostRegressor(estimator=AdaBoost_param[0],n_estimators=AdaBoost_param[1], learning_rate=AdaBoost_param[2], loss=AdaBoost_param[3]))
        elif model == Regression_Models.GBOOST: #Gradient tree boosting GBoost
            self.model_name = "GBoost_regression"
            for i in range(self.output_size):
                self.model.append(ensemble.GradientBoostingRegressor(n_estimators=GBoost_param[0], learning_rate=GBoost_param[1], loss=GBoost_param[2],verbose=verbose))
        
        
        
        
