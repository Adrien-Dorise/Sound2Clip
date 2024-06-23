'''
 Main file to launch a a machine learning example using blobs synthetic data.
 Author: Adrien Dorise (adrien.dorise@hotmail.com) - LR Technologies
 Created: May 2024
 Last updated: Adrien Dorise - May 2024
'''

from config import *
from experiment import Experiment_Grid_Search, Experiment
from data import create_loaders


if __name__ == "__main__":
	from dragonflai.model.machine_learning_architectures.classification import Classification, Classification_Models
    
	train_set, test_set = create_loaders(data_path=data_path)
	experiment = Experiment_Grid_Search(model_list = classification_models,
										param_list = gridsearch_parameters,
										train_set = train_set,
										save_path = save_path
										)
	experiment.grid_search()
	
	
 	# After grid search
	experiment = Experiment(ML_model = ML_model,
							train_set = train_set,
							test_set = test_set,
							save_path = save_path
							)

	experiment.fit()
	results = experiment.predict()
	experiment.visualise()

   