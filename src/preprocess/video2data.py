"""
 video2data.py references all methods used to extract relevant data from a video clip.
 It includes synchronisation of the fourier transform for each frame of a clip.
 Author: Adrien Dorise (adrien.dorise@hotmail.com) - Law Tech Productions
 Created: June 2024
 Last updated: Adrien Dorise - June 2024
"""

import src.preprocess.sound as sound
import src.dataset.extract_video as vid
import cv2
import os


def sync_audio(wav_path, framecount):
    data, sample_rate = sound.wav2sound(wav_path)
    data = sound.stereo2mono(data)
    window_length = int(len(data) / framecount)
    windows = sound.sound2window(data, window_length)
    fouriers = sound.window2fourier(windows, sample_rate)
    return fouriers[0:framecount]
    

def frames2data(frames_path):
    frames = []

    # We sort frames by numerical order before importing them with cv2.
    # By doing so, we respect the sequence order of the video clip.
    frames_name = os.listdir(frames_path)
    extension = f".{frames_name[0].split('.')[1]}"
    frames_name = [name[0:-len(extension)] for name in frames_name]
    frames_name = sorted(frames_name, key=int)

    for file_name in frames_name:
        frames.append(cv2.imread(f"{frames_path}/{file_name}{extension}"))

    return frames
    

if __name__ == "__main__":
    audio_file = "./data/dummy/audio/dummy_audio.wav"
    video_path = "./data/dummy/raw_clip/dummy_clip.mp4"
    
    # Plot Fourier transform of the 10th frame's audio
    fouriers = sync_audio(audio_file, video_path)
    sound.plot_fourier(fouriers[10][0], fouriers[10][1])

    # Extract frames as python data
    frame_folder = "./data/dummy/frames/dummy_clip/"
    frames = frames2data(frame_folder)
