'''
 S2V main file
 Author: Adrien Dorise (Law Tech Productions)
 Date created: June 2024
 Last updated: Adrien Dorise - June 2024
'''

if __name__ == "__main__":
    
	from dragonflai.preprocess import data_loader
	from dragonflai.utils.utils_model import modelType, taskType
	from src.experiment import *
	import src.preprocess.data_loader as loader
	import src.config.data_config as data_config
	import src.config.NN_config as NN_config
	import src.dataset.extract_video as video

	dataset = loader.S2C_Dataset(data_config.train_audio, data_config.train_frame, (NN_config.output_shape,NN_config.output_shape))
	train = loader.create_loader(dataset,NN_config.batch_size,True)



	experiment = Experiment(model = NN_config.NN_model,
							train_set = train,
							validation_set = train,
							test_set = train,
							visualisation_set = train,
							criterion = NN_config.crit,
							lr = NN_config.lr,
							optims = NN_config.optimizer,
							n_epochs = NN_config.num_epoch,
							scheduler = NN_config.scheduler,
							kwargs = NN_config.kwargs,
							batch_size = NN_config.batch_size,
							save_path = data_config.save_path
                         	)

	experiment.model.print_architecture((1,743))
	experiment.fit()
	experiment.predict()
	experiment.visualise(data_config.train_audio, video.video2framerate(data_config.train_video))
	#experiment.save(f"{data_config.save_path}experiment")
	#experiment = Experiment.load(f"{data_config.save_path}experiment")
	#experiment.visualise()
