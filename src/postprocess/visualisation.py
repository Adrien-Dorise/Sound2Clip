"""
 Display the results of a S2C model.
 Author: Adrien Dorise (adrien.dorise@hotmail.com) - Law Tech Productions
 Created: June 2024
 Last updated: Adrien Dorise - June 2024
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import dragonflai.utils.utils_path as utils
import src.utils.utils_s2c as utils_s2c

def plot_cv2(cv2_image, save_path):
        if type(cv2_image) is not np.ndarray: 
            cv2_image = cv2_image.numpy()
        if(cv2_image.shape[0] <= 4):
            cv2_image = cv2_image.transpose(1,2,0)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB) 
        plt.imshow(cv2_image)
        utils.create_file_path(save_path)
        plt.savefig(save_path)


def frames2video(frame_folder, wav_file, save_path, fps, shape=128):
    video = cv2.VideoWriter(save_path,-1,fps,(shape,shape))
    frames = utils_s2c
     

if __name__ == "__main__":
    image_path = "./data/dummy/frames/dummy_clip/0.jpg"
    save_path = "./data/dummy/results/dummy_result.jpg"
    
    # Plot a CV2 image
    if True:
        image = cv2.imread(image_path)
        plot_cv2(image, save_path)