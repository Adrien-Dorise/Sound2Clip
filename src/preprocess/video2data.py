"""
 video2data.py references all methods used to extract relevant data from a video clip.
 It includes synchronisation of the fourier transform for each frame of a clip.
 Author: Adrien Dorise (adrien.dorise@hotmail.com) - Law Tech Productions
 Created: June 2024
 Last updated: Adrien Dorise - June 2024
"""

import src.preprocess.sound as sound
import src.dataset.extract_video as video
import src.utils.utils_s2c as utils_s2c
import cv2


def sync_audio(wav_path, framecount):
    """Create windowed Fouriers transform from a WAV file.
    The number of windows is based on the given framecount.
    Therefore, the WAV file is divided into windows that matches the each frame of a video for a similar duration.

    Args:
        wav_path (string): Path ot the WAV file
        framecount (int): Number of frame in the related video. It is used to select the number and length of the sound windows.

    Returns:
        fouriers (tuple of shape (framecount, (xf, yf), window_length)): Results of the Fourier Transform over all created windows.
    """
    data, sample_rate = sound.wav2sound(wav_path)
    data = sound.stereo2mono(data)
    window_length = int(len(data) / framecount)
    windows = sound.sound2window(data, window_length)
    fouriers = sound.window2fourier(windows, sample_rate)
    return fouriers[0:framecount]
    

def frames2data(frame_folder_path):
    """Convert frame file to open-cv objects

    Args:
        frame_folder_path (string): Path to the folder containning all the frames

    Returns:
        frames (list of cv2 objects): open-cv frames.
    """

    # We sort frames by numerical order before importing them with cv2.
    # By doing so, we respect the sequence order of the video clip.
    frames_path = utils_s2c.get_frames_in_folder(frame_folder_path)

    frames= []
    for f in frames_path:
        frames.append(cv2.imread(f))

    return frames
    

if __name__ == "__main__":
    audio_file = "./data/dummy/audio/dummy_audio.wav"
    video_path = "./data/dummy/raw_clip/dummy_clip.mp4"
    
    # Plot Fourier transform of the 10th frame's audio
    if True:
        framecount = video.video2framecount(video_path)
        fouriers = sync_audio(audio_file, framecount)
        sound.plot_fourier(fouriers[10][0], fouriers[10][1])

    # Extract frames as python data
    if True:
        frame_folder = "./data/dummy/frames/dummy_clip/"
        frames = frames2data(frame_folder)
