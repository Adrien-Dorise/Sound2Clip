"""
 data loader creation for a neural network generative model
 Author: Adrien Dorise (adrien.dorise@hotmail.com) - Law Tech Productions
 Created: June 2024
 Last updated: Adrien Dorise - June 2024
"""

import src.preprocess.video2data as video2data

import random
import cv2
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def image_data_augmentation(image):
    # Randomly flip the image horizontally
    if random.random() < 0.5:
        image = F.hflip(image)
   
    # Randomly adjust brightness, contrast, saturation, and hue
    factor = random.uniform(0.7, 1.3)
    image = F.adjust_brightness(image, brightness_factor=factor)
    image = F.adjust_contrast(image, contrast_factor=factor)
    image = F.adjust_saturation(image, saturation_factor=factor)

    factor = random.uniform(0.1, -0.1) 
    image = F.adjust_hue(image, hue_factor=factor)
    
    # Randomly rotate the image up to 20 degrees
    angle = random.uniform(-20, 20)
    image = F.rotate(image, angle)
    
    # Randomly translate the image
    translate_x = random.uniform(-0.1, 0.1)
    translate_y = random.uniform(-0.1, 0.1)
    image = F.affine(image, angle=0, translate=(translate_x, translate_y), scale=1, shear=0)
    
    # Normalize the image
    #image = F.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return image

class S2C_Dataset(Dataset):
    '''
    Class used to store the dataset handled by pytorch.
    It takes tabular data as input, and images as target.
    '''
    def __init__(self, audio_file, frame_folder, shape=(256,256)):
        '''
        S2C_Dataset class constructor.
        Parameters
        ----------
        audio_folder:
            folder containing the WAV file for a single video clip. The audio can be extracted by using extract_video.py
        frame_folder: 
            folder containing all frames from a single video clip. The frame can be extracted by using extract_video.py
        Returns
        ----------
        None
        '''
        self.target_path = frame_folder
        self.feature_path = audio_file

        self.targets = video2data.frames2data(self.target_path)
        framecount = len(self.targets)
        self.features = video2data.sync_audio(self.feature_path, framecount)
        if len(self.features) == 0 or (self.targets) == 0:
            raise Exception("No img file found")
        if(len(self.features) != len(self.targets)):
            raise Exception("Not the same number of features and targets! Must be an error in your dataset")
        
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),  # convert from [0, 255] to [0.0, 0.1]
            ])
        self.shape = shape

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        target = self.targets[idx]
        target = cv2.resize(target, dsize=(self.shape[0], self.shape[1]), interpolation=cv2.INTER_CUBIC)
        target = self.to_tensor(target)
        return self.features[idx], target 
    

def create_loaders(features, targets, batch_size=None, test_size=0.2, shuffle=True):
    """Create a PyTorch DataLoader object that can handles the data processing when performing training or prediction of a ML/NN model.
    It is possible to create a test set automatically with the test_size parameter.
    
    Args:
        features (list or np.array): List of features
        targets (list or np.array): List of targets
        batch_size (None or int, optional): Size of the batch used during training. When at None, only one batch is created. 
                                            Note, that ML models can't handle multiple batches. Defaults to None.
        test_size (float, optional): Proportion of samples used for the test set between ]0,1[. Defaults to 0.2.
        shuffle (bool, optional): Set to true to shuffle the data before creating the loader. Defaults to True.

    Returns:
        DataLoader, DataLoader: Train and test DataLoader objects
    """
    
    #Divide between train set and test set
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=test_size, shuffle=shuffle, random_state=948401971)
    
    # Create Dataset objects
    train_dataset = Dataset(features_train,targets_train)
    test_dataset = Dataset(features_test, targets_test)
    
    # Define batch size if not specified in the parameters
    if batch_size is None:
        batch_size = len(train_dataset)
    
    # Create loader objects
    train_loader = DataLoader(dataset=train_dataset,
                            shuffle=False,
                            batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset,
                            shuffle=False,
                            batch_size=int(batch_size*test_size))
    return train_loader, test_loader

if __name__ == "__main__":
    import src.preprocess.sound as sound
    import matplotlib.pyplot as plt
    
    frame_folder = "./data/dummy/frames/dummy_clip/"
    wav_folder = "./data/dummy/audio/dummy_audio.wav"
    
    # Display feature and target on a newly created S2C dataset
    if True:
        dataset = S2C_Dataset(wav_folder,frame_folder)
        feat, targ = dataset[0]
        
        sound.plot_fourier(feat[0],feat[1])
        targ = targ.numpy().transpose(1,2,0)
        targ = cv2.cvtColor(targ, cv2.COLOR_BGR2RGB) 
        plt.imshow(targ)
        plt.show()