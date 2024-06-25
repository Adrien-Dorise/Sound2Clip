from scipy.io.wavfile import write
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
import os


def generate_sin(freq, sample_rate, duration, save_path):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    
    try: 
        if not os.path.exists(save_path): 
            os.makedirs(save_path)     
    except OSError: 
        print ('Error: Creating directory of data') 
    write(f"{save_path}sinwave.mp3", sample_rate, y)
    return x, y



def sound2fourier(mp3_path, sample_rate, duration):
    N = sample_rate * duration

    yf = fft(duration)
    xf = fftfreq(N, 1 / sample_rate)

    plt.plot(xf, np.abs(yf))
    plt.show()


if __name__ == "__main__":
    save_path = "data/dummy/"
    freq = 1000
    sample_rate = 44100
    duration = 5
    generate_sin(freq, sample_rate, duration, save_path)
    
    sound2fourier(f"{save_path}/sinwave.mp3", sample_rate, duration)
    
    