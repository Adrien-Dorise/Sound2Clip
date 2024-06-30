'''
 S2V main file
 Author: Adrien Dorise (Law Tech Productions)
 Date created: June 2024
 Last updated: Adrien Dorise - June 2024
'''

if __name__ == "__main__":
	from src.experiment import Experiment
	import src.preprocess.data_loader as loader
	import src.config.data_config as data_config
	import src.config.NN_config as NN_config
	import src.dataset.extract_video as video


	# Train a new neural network
	if True:
		dataset = loader.S2C_Dataset_Folder(data_config.train_audio_folder, data_config.train_frame, (NN_config.output_shape,NN_config.output_shape))
		train = loader.create_loader(dataset,NN_config.batch_size,True)
		
		dataset_test = loader.S2C_Dataset_Folder(data_config.test_audio_folder, data_config.test_frame, (NN_config.output_shape,NN_config.output_shape))
		test = loader.create_loader(dataset_test,NN_config.batch_size,False)
		
		experiment = Experiment(model = NN_config.NN_model,
								train_set = train,
								validation_set = test,
								test_set = test,
								visualisation_set = test,
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
		experiment.save(f"{data_config.save_path}experiment")
		experiment.visualise(data_config.test_audio, video.video2framerate(data_config.test_video))
	

	# Create a new clip
	if False:
		dataset_overfit = loader.S2C_Dataset(data_config.overfit_audio, data_config.overfit_frame, (NN_config.output_shape,NN_config.output_shape))
		test_overfit = loader.create_loader(dataset_overfit,NN_config.batch_size,False)
		
		dataset_test = loader.S2C_Dataset_Folder(data_config.test_audio_folder, data_config.test_frame, (NN_config.output_shape,NN_config.output_shape))
		test = loader.create_loader(dataset_test,NN_config.batch_size,False)

		experiment = Experiment.load(f"./results/tmp/experiment")
		experiment.model.load_model(f"./results/tmp/epoch200_1")
		experiment.visualisation_set = test_overfit
		experiment.visualise(data_config.overfit_audio, video.video2framerate(data_config.overfit_video))
