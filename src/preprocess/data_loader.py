"""
 data loader creation for a neural network generative model
 Author: Adrien Dorise (adrien.dorise@hotmail.com) - Law Tech Productions
 Created: June 2024
 Last updated: Adrien Dorise - June 2024
"""

import src.preprocess.video2data as video2data
import src.postprocess.visualisation as visu

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
    

def create_loader(dataset, batch_size=None, shuffle=True):
    """Create a PyTorch DataLoader object that can handles the data processing when performing training or prediction of a ML/NN model.
    
    Args:
        dataset (Dataset): Initisialised "Dataset" object.
        batch_size (None or int, optional): Size of the batch used during training. When at None, only one batch is created. 
                                            Note, that ML models can't handle multiple batches. Defaults to None.
        shuffle (bool, optional): Set to true to shuffle the data before creating the loader. Defaults to True.

    Returns:
        loader: Initialised PyTorch DataLoader object
    """
        
    # Define batch size if not specified in the parameters
    if batch_size is None:
        batch_size = len(dataset)
    
    # Create loader objects
    loader = DataLoader(dataset=dataset,
                        shuffle=shuffle,
                        batch_size=batch_size)
    return loader

if __name__ == "__main__":
    import src.preprocess.sound as sound
    import matplotlib.pyplot as plt
    import numpy as np
    
    wav_folder = "./data/dummy/audio/dummy_audio.wav"
    frame_folder = "./data/dummy/frames/dummy_clip/"
    
    # Display feature and target on a newly created S2C dataset
    if True:
        dataset = S2C_Dataset(wav_folder,frame_folder)
        feat, targ = dataset[0]
        
        sound.plot_fourier(feat[0],feat[1])
        visu.plot_cv2(targ)

    # Create DataLoader
    if True:
        dataset = S2C_Dataset(wav_folder,frame_folder)
        loader = create_loader(dataset,1,False)
        feat, targ = next(iter(loader))
    
        sound.plot_fourier(feat[0][0],feat[1][0])
        visu.plot_cv2(targ[0])    
    
