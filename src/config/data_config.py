"""
 Data related (creation, preprocessing, paths...) parameters
 Author: Adrien Dorise (adrien.dorise@hotmail.com) - Law Tech Productions
 Created: June 2024
 Last updated: Adrien Dorise - June 2024
"""

# Paths
save_path = r"results/tmp/"

train_frame = r"./data/train/frames/"
train_audio_folder = r"./data/train/audio/"
train_video = "./data/train/raw_clip/konosuba_s3.mp4"
train_audio = "./data/train/audio/konosuba_s3.wav"

overfit_frame = r"./data/train/frames/konosuba_s3"
overfit_audio_folder = r"./data/train/audio/konosuba_s3.wav"
overfit_video = r"./data/train/raw_clip/konosuba_s3.mp4"
overfit_audio = r"./data/train/audio/konosuba_s3.wav"


test_frame = r"./data/test/frames/"
test_audio_folder = r"./data/test/audio/"
test_video = r"./data/test/raw_clip/ranking_of_kings.mp4"
test_audio = r"./data/test/audio/ranking_of_kings.wav"

dummy_video = r"./data/dummy/raw_clip/dummy_clip.mp4"
dummy_frame = r"./data/dummy/frames/dummy_clip/"
dummy_audio = r"./data/dummy/audio/dummy_clip.wav"