"""
 Parameters for machine learning applications in the blob example
 Author: Adrien Dorise (adrien.dorise@hotmail.com) - LR Technologies
 Created: May 2024 
 Last updated: Adrien Dorise - May 2024
"""

from dragonflai.model.machine_learning_architectures.classification import Classification, Classification_Models



# Path parameters
save_path = r"examples/machine_learning/mnist_grid_search/outputs/"
data_path = r"examples/machine_learning/mnist_grid_search/data/"

# Grid search parameters

SGD_parameters = {'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'alpha' : [0.001,0.0005,0.0001],
                    'learning_rate': ['constant', 'optimal', 'invscaling']}
SVM_parameters = {'kernel' : ('linear', 'rbf', 'sigmoid'),
                    'C' : [1,5,10],
                    'coef0' : [0.01,10,0.5],
                    'gamma' : ('auto','scale')}
KNN_parameters = {"n_neighbors": range(1,10),
                    "p": [1,2]}
tree_parameters = {"criterion": ["gini", "entropy", "log_loss"],
                    "max_depth": [None, 100,1000, 5000]}
forest_parameters = {"n_estimators": [10,50,100],
                    "criterion": ["gini", "entropy", "log_loss"],
                    "max_depth": [None, 100,1000]}
adaBoost_parameters = {"estimator": [None],
                        "n_estimators": [10,50,100],
                        "learning_rate": [0.85,1,1.25]}
GBoost_parameters = {"n_estimators": [10,50,100],
                    "learning_rate": [0.85,1,1.25]}

# Be careful as the order in the classification models must be the same as in gridsearch_parameters
classification_models = [Classification(models) for models in Classification_Models]
gridsearch_parameters = [SGD_parameters, SVM_parameters, KNN_parameters, tree_parameters, forest_parameters, adaBoost_parameters, GBoost_parameters]

#After grid search, we select the best model
SVM_param = ["rbf", 5, 0.01, 'scale']
ML_model = Classification(Classification_Models.SVM, SVM_param=SVM_param)


