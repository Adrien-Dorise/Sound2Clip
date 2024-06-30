"""
 Display the results of a S2C model.
 Author: Adrien Dorise (adrien.dorise@hotmail.com) - Law Tech Productions
 Created: June 2024
 Last updated: Adrien Dorise - June 2024
"""

import cv2
import PIL
import numpy as np
import os
import dragonflai.utils.utils_path as utils
import src.preprocess.video2data as video2data
import src.dataset.extract_video as video
import moviepy.editor as mp

def save_cv2(cv2_image, save_path, plot=False):
    """Export a cv2 image into a file.
    Matplotlib is used for the export. 

    Args:
        cv2_image (cv2 object): open-cv image
        save_path (string): File path of the futur saved picture
        plot (bool): Set to True to plot a blocking window of the image.
    """
    
    # Transform image in numpy array
    if type(cv2_image) is not np.ndarray: 
        cv2_image = cv2_image.numpy()
    
    # Verifies that image not in range [-1,1] or <0
    if(np.min(cv2_image) < 0):
        raise Warning("ERROR in save_cv2: Image range must be strictly positive ([0,1] or [0, 255])")


    # Case where image shape is (channel, size, size) -> Transform to (size, size, channel)
    if(cv2_image.shape[0] <= 4):
        cv2_image = cv2_image.transpose(1,2,0)

    # Case where images range in [0,1] -> Transform to [0,255]
    if np.max(cv2_image) < 2: 
        cv2_image = np.array(cv2_image*255, dtype=np.uint8)
    
    utils.create_file_path(save_path)
    cv2.imwrite(save_path, cv2_image)
    
    if plot:
        cv2.imshow("S2C Image", cv2_image)
        cv2.waitKey(0)


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
    if(fps < 1):
        raise Warning(f"ERROR in frames2video: fps parameter incorrect: fps={fps}")
    utils.create_file_path(save_path)
    tmp_path = f"{save_path[0:-4]}_tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    tmp_video = cv2.VideoWriter(tmp_path,fourcc,fps,(shape,shape))
    
    frames = video2data.frames2data(frame_folder)
    for f in frames:
         f = cv2.resize(f,(shape,shape))
         tmp_video.write(f)

    cv2.destroyAllWindows()
    tmp_video.release()
    
    video = mp.VideoFileClip(tmp_path)
    audio = mp.AudioFileClip(wav_file)
    video = video.set_audio(audio)
    video.write_videofile(save_path)
    video.close()
    os.remove(tmp_path)

if __name__ == "__main__":
    
    # Plot a CV2 image
    if True:
        image_path = "./data/dummy/frames/dummy_clip/0.jpg"
        save_path = "./data/dummy/results/dummy_result.jpg"
        image = cv2.imread(image_path)
        save_cv2(image, save_path)

    # Create a video from frames and audio
    if True:
         video_path = "data/raw_clip/Attack on Titan Season 2 - Opening _ Shinzou wo Sasageyo!.mp4"
         frame_folder = "data/frames/attack_on_titan_s2/"
         wav_file = "data/audio/attack_on_titan_s2/audio.wav"
         save_path = "./data/dummy/results/dummy_generated_video_AOT.mp4"

         fps = video.video2framerate(video_path)
         frames2video(frame_folder, wav_file, save_path, fps, shape=128)