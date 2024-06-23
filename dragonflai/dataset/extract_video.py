import numpy as np
import cv2
import os
from audio_extract import extract_audio
from dragonflai.utils.utils_path import create_file_path


def video2image(video_path, save_path):
    
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
            name = f"{save_path}/frame{str(current_frame)}.jpg"
            cv2.imwrite(name, frame) 
            current_frame += 1
        else: 
            break
    cam.release() 
    cv2.destroyAllWindows()    


def video2sound(video_path, save_path):
    extract_audio(input_path=video_path, output_path=save_path)



if __name__ == "__main__":
    
    video_path = "./data/raw/Attack on Titan Season 2 - Opening _ Shinzou wo Sasageyo!.mp4"
    if False:
        save_path = "./data/images/attack_on_titan_s2/"
        video2image(video_path, save_path)
    if True:
        save_path = "./data/audio/attack_on_titan_s2/audio.mp3"
        video2sound(video_path, save_path)