'''
 Main file to launch a a machine learning example using blobs synthetic data.
 Author: Adrien Dorise (adrien.dorise@hotmail.com) - LR Technologies
 Created: May 2024
 Last updated: Adrien Dorise - May 2024
'''

from config import *
from experiment import Experiment, Model_Type
from data import create_loaders, Paradigm


def practice_experiment(ML_model, model_type, train_set, test_set):
	"""Perform an experiment. The experiment can be a classification, clustering or regression
	An experiment consist on:
		- Create the experiment object
		- Detail the model used
		- Train the model
		- Visualise the results

	Args:
		ML_model (MachineLearning object): ML model used in the experiment
		model_type (Model_Type enum): Set the paradigm of the experiment. The list can be found in the experiment file.
		train_set (DataLoader): Train DataLoader
		test_set (DataLoader): Test DataLoader
	"""
	experiment = Experiment( 
				model = ML_model,
				train_set = train_set, 
				test_set = test_set,
				model_type = model_type,
				save_path = save_path
				)
	
	experiment.model.print_architecture()
	experiment.fit()
	if(model_type is not Model_Type.CLUSTERING):
		results = experiment.predict()
	experiment.visualise()


if __name__ == "__main__":
	
	#Classification
	if True:
		train_set, test_set = create_loaders(list_N, list_centroid, paradigm=Paradigm.CLASSIFICATION)
		for model in classifications_to_test:
			practice_experiment(model, Model_Type.CLASSIFICATION, train_set, test_set)
	
	#Clustering
	if True:
		train_set, test_set = create_loaders(list_N, list_centroid, paradigm=Paradigm.CLUSTERING)
		for model in clusterings_to_test:
			practice_experiment(model, Model_Type.CLUSTERING, train_set, test_set)

	#Regression
	if True:	
		train_set, test_set = create_loaders(list_N, list_centroid, paradigm=Paradigm.REGRESSION)
		for model in regression_to_test:
			practice_experiment(model, Model_Type.REGRESSION, train_set, test_set)
   