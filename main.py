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

	dataset = loader.S2C_Dataset(data_config.train_audio, data_config.train_frame, (NN_config.output_shape,NN_config.output_shape))
	train = loader.create_loader(dataset,NN_config.batch_size,True)
	visualisation = loader.create_loader(dataset,NN_config.batch_size,False)



	experiment = Experiment(model = NN_config.NN_model,
							train_set = train,
							validation_set = train,
							test_set = train,
							visualisation_set = visualisation,
							criterion = NN_config.crit,
							lr = NN_config.lr,
							optims = NN_config.optimizer,
							n_epochs = NN_config.num_epoch,
							scheduler = NN_config.scheduler,
							kwargs = NN_config.kwargs,
							batch_size = NN_config.batch_size,
							save_path = data_config.save_path
                         	)

	#experiment.model.print_architecture((1,743))
	experiment.fit()
	experiment.save(f"{data_config.save_path}experiment")
	experiment.visualise(data_config.train_audio, video.video2framerate(data_config.train_video))
	
	#experiment = Experiment.load(f"./results/tmp/AoT/experiment")
	#experiment.model.load_model(f"./results/tmp/dummy_experiment/epoch1000_1")

