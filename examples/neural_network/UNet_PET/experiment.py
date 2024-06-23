'''
This package references all neural network classes used in the application.
Author: Adrien Dorise - Edouard Villain ({adorise, evillain}@lrtechnologies.fr) - LR Technologies
Created: Avril 2024
Last updated: Adrien Dorise - Avril 2024

'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torch import nn


class Experiment():
    def __init__(self, model,
                train_loader, test_loader, 
                num_epoch     = 10,
                batch_size    = 32,
                learning_rate = 1e-03,
                weight_decay  = 1e-03,
                optimizer     = torch.optim.Adam,
                scheduler     = None,
                kwargs        = {},
                criterion     = torch.nn.L1Loss(),
                nb_workers    = 0,
                numberOfImagesToDisplay = 5,
                save_path = './examples/neural_network/UNet_PET/results/'):
        # Model parameters
        self.model                   = model
        self.num_epoch               = num_epoch
        self.batch_size              = batch_size
        self.learning_rate           = learning_rate
        self.weight_decay            = weight_decay
        self.optimizer               = optimizer
        self.scheduler               = scheduler
        self.kwargs                  = kwargs
        self.criterion               = criterion
        self.nb_workers              = nb_workers
        self.train_loader            = train_loader
        self.test_loader             = test_loader
        self.numberOfImagesToDisplay = numberOfImagesToDisplay

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
        self.numberImagesToDisplay = 3
        self.model.history.verbosity = 0
        _, output, (input, target) = self.model.predict(self.test_loader,self.criterion)

        pred = nn.Softmax(dim=1)(torch.from_numpy(output))
        pred_labels = pred.argmax(dim=1)
        pred_labels = pred_labels.unsqueeze(1)
        prediction = pred_labels.to(torch.float)

        all_imgs = []
        # Fetch test data
        for batch in self.test_loader:
            img, _ = batch
            all_imgs.extend(img)

        for _, (inp, targ, predic) in enumerate(zip(all_imgs[:self.numberImagesToDisplay], target[:self.numberImagesToDisplay], prediction[:self.numberImagesToDisplay])):
            _, axes = plt.subplots(1, 3)

            def convert_tensor2opencv(image):
                image = image.squeeze() * 127
                if isinstance(image, torch.Tensor):
                    image = image.numpy()
                img = image.astype(np.uint8)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            inp_cv = cv2.cvtColor(np.transpose((inp.numpy()*255).astype(np.uint8), (1,2,0)), cv2.COLOR_RGB2BGR)

            targ_cv = cv2.applyColorMap(convert_tensor2opencv(targ), cv2.COLORMAP_JET)
            predic_cv = cv2.applyColorMap(convert_tensor2opencv(predic), cv2.COLORMAP_JET)

            overlay_targ = cv2.addWeighted(inp_cv, 0.5, targ_cv, 0.5, 0)  # Adjust the weights as needed
            overlay_pred = cv2.addWeighted(inp_cv, 0.5, predic_cv, 0.5, 0)

            axes[0].imshow(cv2.cvtColor(inp_cv, cv2.COLOR_BGR2RGB))
            axes[0].set_title('RGB Image')
            axes[0].axis('off')

            axes[1].imshow(cv2.cvtColor(overlay_targ, cv2.COLOR_BGR2RGB))
            axes[1].set_title('RGB Image + Target')
            axes[1].axis('off')

            axes[2].imshow(cv2.cvtColor(overlay_pred, cv2.COLOR_BGR2RGB))
            axes[2].set_title('RGB Image + Prediction')
            axes[2].axis('off')

            plt.show()

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
