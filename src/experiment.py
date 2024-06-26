'''
Example experiment class for DragonflAI.
This example takes care of classificaiton example on tabular data.
Author: Adrien Dorise - Law Tech Productions
Created: June 2023
Last updated: Adrien Dorise - June 2024

'''
import dragonflai.utils.utils_model as utils_model
from dragonflai.utils.utils_path import create_file_path
import src.postprocess.visualisation as visu

import pickle





class Experiment():
    def __init__(self, model,
                train_set,
                test_set,
                validation_set,
                visualisation_set,
                criterion,
                batch_size,
                n_epochs,
                lr,
                scheduler,
                kwargs,
                optims,
                save_path,
                ):
        
        #Paths
        self.save_path = save_path

        #Dataset
        self.train_set = train_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.visualisation_set = visualisation_set

        #Model parameters  
        self.model = model
        self.criterion = criterion
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.optims = optims
        self.scheduler = scheduler
        self.kwargs = kwargs
         
    def fit(self):
        """Train the model using the data available in the train set.
        """
        self.model._compile(self.train_set, self.test_set, self.criterion, lr=self.lr, opts=self.optims, scheduler=self.scheduler, batch_size=self.batch_size, epochs=self.n_epochs, kwargs=self.kwargs)
        history = self.model.fit(self.train_set, valid_set=self.validation_set, criterion=self.criterion, epochs=self.n_epochs)

        self.model.plot_learning_curve(history.loss_train,history.loss_val, f"{self.save_path}loss_history")
        

    def predict(self):          
        """Model prediction on the samples available in the test set

        """
        loss, output, (_,target) = self.model.predict(self.test_set)
        
        return output


    def visualise(self):
        """Visualisation of the first picture of the visualisation set.
        """
        loss, output, (_,target) = self.model.predict(self.visualisation_set)
        idx = 0
        for img in output:
            visu.plot_cv2(img,f"{self.save_path}generated_frames/{idx}.jpg")
            idx += 1
        

    
    def save(self, path):
        """Save the whole experiment class as a pickle object.

        Args:
            path (string): Path to save the experiment status
        """

        #Check if folder exists
        create_file_path(path)

        with open(path, 'wb') as file:
            try:
                pickle.dump(self, file)
            except EOFError:
                raise Exception("Error in save experiment: Pickle was not able to save the file.")

    @classmethod
    def load(self, path):
        """Load a pickle object to an Experiment class Python variable
        This is a class method. It means that a reference to the class is NOT necessary to call this method. Simply type <your_experiment = Experiment.load(filename)>

        Args:
            filename (string): Path to the pickle saved object.
        """
        with open(path, 'rb') as file:
            try:
               return pickle.load(file)
            except EOFError:
                raise Exception("Error in load experiment: Pickle was not able to retrieve the file.")
