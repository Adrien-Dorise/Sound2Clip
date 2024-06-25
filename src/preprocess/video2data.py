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
    audio_file = "./data/audio/attack_on_titan_s2/audio.wav"
    video_path = "./data/raw/Attack on Titan Season 2 - Opening _ Shinzou wo Sasageyo!.mp4"
    
    fouriers = sync_audio(audio_file, video_path)
    sound.plot_fourier(fouriers[2000][0], fouriers[2000][1])
