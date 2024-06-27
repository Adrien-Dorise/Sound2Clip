"""
This package references all utils for the S2C project
Author: Adrien Dorise (adrien.dorise@hotmail.com) - Law Tech Productions
Created: June 2024
Last updated: Adrien Dorise - June 2024
"""

import os

def get_frames_in_folder(folder_path):
    """Get all the frames in the folder and list them in numerical ascendant order

    Args:
        folder_path (string): Path to the folder containing all the frames

    Returns:
        frames_path (list of strings): Path to all the frames in a folder in numerical ascendant order.
    """
    if folder_path[-1] is not '/':
        folder_path = f"{folder_path}/"
    frames_name = os.listdir(folder_path)
    extension = f".{frames_name[0].split('.')[1]}"
    frames_name = [name[0:-len(extension)] for name in frames_name]
    frames_name = sorted(frames_name, key=int)
    frames_path = [f"{folder_path}{frame}{extension}" for frame in frames_name] 
    return frames_path


if __name__ == "__main__":

    # Get all the frames in the dummy folder
    if True:
        frame_folder_path = r"./data/dummy/frames/dummy_clip/"
        frames = get_frames_in_folder(frame_folder_path)
        for f in frames:
            print(f)