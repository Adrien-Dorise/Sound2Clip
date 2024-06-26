"""
 extract_video.py references all methods used extract audio and images features from a video file
 Author: Adrien Dorise (adrien.dorise@hotmail.com) - Law Tech Productions
 Created: June 2024
 Last updated: Adrien Dorise - June 2024
"""

import numpy as np
import cv2
import os
from audio_extract import extract_audio
from dragonflai.utils.utils_path import create_file_path


def video2image(video_path, save_path):
    """Extract all the frames of a video file, and save each frame in a folder

    Args:
        video_path (string): Path of the video file
        save_path (string): Path to the folder in which the frames are saved
    """
    try: 
        if not os.path.exists(save_path): 
            os.makedirs(save_path)     
    except OSError: 
        print ('Error: Creating directory of data') 
    
    cam = cv2.VideoCapture(video_path)
    current_frame = 0
    while(True): 
        ret,frame = cam.read() 
        if ret: 
            name = f"{save_path}/{str(current_frame)}.jpg"
            cv2.imwrite(name, frame) 
            current_frame += 1
        else: 
            break
    cam.release() 
    cv2.destroyAllWindows()    

def video2sound(video_path, save_path):
    """Extract the audio of video file and save it in a folder

    Args:
        video_path (string): Path of the video file
        save_path (string): Complete filename of the saved audio file
        """
    if os.path.exists(save_path):
        os.remove(save_path)
    extract_audio(input_path=video_path, output_path=save_path, output_format="wav")

def video2framerate(video_path):
    """Return the framerate of a video file.

    Args:
        video_path (string): Path of the video file
    
    Returns:
        fps (float): Framerate of the video file
    """   
    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    cam.release() 
    cv2.destroyAllWindows() 
    return fps

def video2framecount(video_path):
    """Return the number of frames of a video file.

    Args:
        video_path (string): Path of the video file
    
    Returns:
        fps (int): Number of frames of the video file
    """   
    cam = cv2.VideoCapture(video_path)
    framecount = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    cam.release() 
    cv2.destroyAllWindows() 
    return int(framecount)




if __name__ == "__main__":
    
    video_path = "./data/dummy/raw_clip/dummy_clip.mp4"
    fps = video2framerate(video_path)
    print(f"video framerate: {fps}")
    
    # Extract frames from video
    if True:
        save_path = "./data/dummy/frames/dummy_clip/"
        video2image(video_path, save_path)
    
    # Extract audio from video
    if True:
        save_path = "./data/dummy/audio/dummy_audio.wav"
        video2sound(video_path, save_path)