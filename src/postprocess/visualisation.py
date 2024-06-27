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
import src.preprocess.video2data as video2data
import src.dataset.extract_video as video
import moviepy.editor as mp

def plot_cv2(cv2_image, save_path):
    """Export a cv2 image into a file.
    Matplotlib is used for the export. 

    Args:
        cv2_image (cv2 object): open-cv image
        save_path (string): File path of the futur saved picture
    """
    if type(cv2_image) is not np.ndarray: 
        cv2_image = cv2_image.numpy()
    if(cv2_image.shape[0] <= 4):
        cv2_image = cv2_image.transpose(1,2,0)
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB) 
    plt.imshow(cv2_image)
    utils.create_file_path(save_path)
    plt.savefig(save_path)


def frames2video(frame_folder, wav_file, save_path, fps, shape=128):
    """Create a video from a frame folder and an audio file.
    Open-cv is used to create the video, and moviepy is used to add the audio.

    Args:
        frame_folder (string): Path to the folder containing all the frames
        wav_file (string): Path to the WAV file
        save_path (string): Name of the exported video file. WARNING: Add the .mp4 extension
        fps (int): Framerate of the video
        shape (int, optional): Shape of the image. Only squared images are supported right now. Defaults to 128.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(save_path,fourcc,fps,(shape,shape))
    
    frames = video2data.frames2data(frame_folder)
    for f in frames:
         f = cv2.resize(f,(shape,shape))
         video.write(f)

    cv2.destroyAllWindows()
    video.release()

    video = mp.VideoFileClip(save_path)
    audio = mp.AudioFileClip(wav_file)
    video = video.set_audio(audio)
    video.write_videofile(save_path)

if __name__ == "__main__":
    
    # Plot a CV2 image
    if True:
        image_path = "./data/dummy/frames/dummy_clip/0.jpg"
        save_path = "./data/dummy/results/dummy_result.jpg"
        image = cv2.imread(image_path)
        plot_cv2(image, save_path)

    # Create a video from frames and audio
    if True:
         video_path = "./data/dummy/raw_clip/dummy_clip.mp4"
         frame_folder = "./data/dummy/frames/dummy_clip/"
         wav_file = "./data/dummy/audio/dummy_audio.wav"
         save_path = "./data/dummy/results/dummy_generated_video.mp4"

         fps = video.video2framerate(video_path)
         frames2video(frame_folder, wav_file, save_path, fps, 128)