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
        save_path (string): Path to the folder in which the audio is saved
        """
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
    
    video_path = "./data/raw/Attack on Titan Season 2 - Opening _ Shinzou wo Sasageyo!.mp4"
    
    fps = video2framerate(video_path)
    print(f"video framerate: {fps}")
    if False:
        save_path = "./data/images/attack_on_titan_s2/"
        video2image(video_path, save_path)
    if True:
        save_path = "./data/audio/attack_on_titan_s2/audio.wav"
        video2sound(video_path, save_path)