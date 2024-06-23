'''
This package references all neural network classes used in the application.
Author: Adrien Dorise - Edouard Villain ({adorise, evillain}@lrtechnologies.fr) - LR Technologies
Created: September 2023
Last updated: Adrien Dorise - September 2023

'''
import torch 
import pickle 
import cv2 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Experiment():
    def __init__(self, model,
                train_loader, test_loader, 
                num_epoch     = 50,
                batch_size    = 32,
                learning_rate = 1e-03,
                weight_decay  = 1e-03,
                optimizer     = torch.optim.Adam,
                scheduler     = None,
                kwargs        = {},
                criterion     = torch.nn.L1Loss(),
                nb_workers    = 0,
                save_path = "./examples/neural_network/VIT_MNIST/results/"):
        #Model parameters  
        self.model         = model
        self.num_epoch     = num_epoch
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.kwargs        = kwargs
        self.criterion     = criterion
        self.nb_workers    = nb_workers
        self.train_loader  = train_loader
        self.test_loader   = test_loader

        # Path parameter
        self.save_path = save_path

    def fit(self):
        """Train the model using the data available in the train and validation folder path.
        """
        self.model._compile(self.train_loader, self.test_loader, 
                            self.criterion, self.learning_rate, 
                            self.optimizer, self.scheduler, 
                            self.batch_size, self.num_epoch, **self.kwargs)
        
        history = self.model.fit(self.train_loader,
        valid_set=self.test_loader,
        epochs=self.num_epoch, 
        criterion=self.criterion)

        self.model.plot_learning_curve(history.loss_train,history.loss_val, f"{self.save_path}loss_history")
        self.model.plot_learning_curve(history.acc_train,history.acc_val, f"{self.save_path}accuracy_history")
        self.model.plot_learning_rate(history.lr, f"{self.save_path}learning_rate_history")

    def predict(self):          
        """Model prediction on the samples available in the test folder path
        """
        # !!! Data loading !!!
        _, _, _ = self.model.predict(self.test_loader,self.criterion)



    def visualise(self):
        """Visualisation of the first picture of the test set.
        The input + predicted images are both shown.
        """
        # !!! Data loading !!!
        self.model.history.verbosity = 0
        _, pred, (_, target) = self.model.predict(self.test_loader,self.criterion)
        print('\nCONFUSION MATRIX : \n\n')
        cm = confusion_matrix(target, torch.argmax(torch.tensor(pred), dim=1))
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot().figure_.savefig('{}/CM.png'.format(self.model.save_path))
        
        
    
    def save(self, filename):
        """Save the whole experiment class as a pickle object.

        Args:
            filename (string): Path to save the experiment status
        """
        with open(filename, 'wb') as file:
            try:
                pickle.dump(self, file)
            except EOFError:
                raise Exception("Error in save experiment: Pickle was not able to save the file.")

    @classmethod
    def load(self, filename):
        """Load a pickle object to an Experiment class Python variable
        This is a class method. It means that a reference to the class is NOT necessary to call this method. Simply type <your_experiment = Experiment.load(filename)>

        Args:
            filename (string): Path to the pickle saved object.
        """
        with open(filename, 'rb') as file:
            try:
               return pickle.load(file)
            except EOFError:
                raise Exception("Error in load experiment: Pickle was not able to retrieve the file.")
