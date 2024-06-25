"""
 video2data.py references all methods used to extract relevant data from a video clip.
 It includes synchronisation of the fourier transform for each frame of a clip.
 Author: Adrien Dorise (adrien.dorise@hotmail.com) - Law Tech Productions
 Created: June 2024
 Last updated: Adrien Dorise - June 2024
"""

import src.preprocess.sound as sound
import src.dataset.extract_video as vid


def sync_audio(wav_path, video_path):
    data, sample_rate = sound.wav2sound(wav_path)
    data = sound.stereo2mono(data)
    framecount = vid.video2framecount(video_path)
    window_length = int(len(data) / framecount)
    windows = sound.sound2window(data, window_length)
    fouriers = sound.window2fourier(windows, sample_rate)
    return fouriers[0:framecount]
    
    

if __name__ == "__main__":
    audio_file = "./data/dummy/audio/dummy_audio.wav"
    video_path = "./data/dummy/raw_clip/dummy_clip.mp4"
    
    # Plot Fourier transform of the 10th frame's audio
    fouriers = sync_audio(audio_file, video_path)
    sound.plot_fourier(fouriers[10][0], fouriers[10][1])
