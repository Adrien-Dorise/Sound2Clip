import numpy as np
import scipy.io.wavfile as wav
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
import os


def generate_sin(freq, sample_rate, duration, save_path):
    """Generate a sinusoid and save it as a WAV file

    Args:
        freq (int): Frequency of the sinusoïd
        sample_rate (int): Sample rate of the sinsoïd
        duration (int): Duration of the WAV file
        save_path (string): Folder in which the WAV file will be saved

    Returns:
        x (np.array of float): abscisse values of the signal
        y (np.array of float): ordinate values of the signal
    """
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    
    try: 
        if not os.path.exists(save_path): 
            os.makedirs(save_path)     
    except OSError: 
        print ('Error: Creating directory of data') 
    wav.write(f"{save_path}sinwave.wav", sample_rate, y)
    return x, y

def wav2sound(wav_path):
    """Load a sound signal from a WAV file into a python data array

    Args:
        wav_path (string): Path to the WAV file

    Returns:
        data (np.array of float): Data corresponsding to the sound signal
        sample_rate (int): Sample rate of the file
    """
    sample_rate, data = wav.read(wav_path) 
    return data, sample_rate


def sound2fourier(data, sample_rate):
    """Fourier transform on a signal array

    Args:
        data (np.array of float): Signal data
        sample_rate (int): Sample rate of the sound signal

    Returns:
        xf (np.array of float): abscisse values of the fourier transform
        yf (np.array of float): ordinate values of the fourier transform
    """
    yf = fft(data)
    xf = fftfreq(len(data), 1 / sample_rate)
    return xf, yf

def sound2window(data, window_length):
    """Apply a sliding window to a sound signal.
    The sliding windows are overlayed together.

    Args:
        data (np.array of float): Signal data
        window_length (int): Desired length of the windows

    Returns:
        list of np.array of float: List containing all the windows
    """
    window_shift = int(window_length) #Can be divided for window overlay
    return [data[i:i+window_length] for i in range(0,len(data)-window_length+1,window_shift)]

def window2fourier(windows, sample_rate):
    """Apply Fourier Transform on multiple data windows

    Args:
        windows ([[float]]): List containing all the windows
        sample_rate (int): Sample rate of the original sound signal

    Returns:
        [[xf,yf]]: List containing the fourier abscisse (xf) and ordinate (yf) for each window
    """
    fouriers = []
    for data in windows:
        fouriers.append(sound2fourier(data,sample_rate))
    return fouriers

def stereo2mono(stereo_data):
    if(len(np.shape(stereo_data)) == 1):
        return stereo_data
    
    mono_data = np.mean(stereo_data,axis=1)
    return mono_data

def plot_fourier(xf, yf):
    """Plot the fourier transform of a signal on a graph

    Args:
        xf (np.array of float): abscisse values of the fourier transform
        yf (np.array of float): ordinate values of the fourier transform
    """
    half_len = int(len(xf)/2)
    plt.plot(xf[0:half_len], np.abs(yf[0:half_len]))
    plt.grid(True)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Fourier Transform of a WAV file")
    plt.show()
    


if __name__ == "__main__":
    save_path = "data/dummy/"
    freq = 1000
    sample_rate = 44100
    duration = 5
    generate_sin(freq, sample_rate, duration, save_path)
    
    data, sample_rate = wav2sound(f"{save_path}/sinwave.wav")
    windows = sound2window(data, 500)
    fouriers = window2fourier(windows, sample_rate)
    
    if False:
        plot_fourier(fouriers[0][0], fouriers[0][1])
        plot_fourier(fouriers[250][0], fouriers[250][1])