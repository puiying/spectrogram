import sys
import wave
import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftfreq, fftshift
import matplotlib.pyplot as plt

# function to read .wav files
def read_wav_file(audio_filename, win_size, step_size):
    obj = wave.open(audio_filename, 'r')
    # obj = wave.open('sa1.wav','r')
    # Sampling Rate
    sampling_rate = obj.getframerate()
    # Mono or Stero
    obj_channel = obj.getnchannels()
    # Sample Width (In Terms of Bytes)
    sample_width = obj.getsampwidth()
    # Number of Frames
    n_frames = obj.getnframes()
    # Length of audio file (in ms)
    len_audio_file = (n_frames / sampling_rate) * 1000
    # Raw Byte Data
    data = obj.readframes(n_frames)
    # Close the Wave File.
    obj.close()
    # Decoding byte data into dec data.  
    dec_data = decode_bit_to_dec(data, sample_width)

    return dec_data, sampling_rate

# create a function to perform fourier transform
def FFT(dec_data, sampling_rate, win_size=25, step_size=10):
    """
    win_size (in ms)
    step_size (in ms)
    """
    # Placeholder for windowed signals
    tmp = []
    # Window size in terms of samples
    win_size_in_samples = int(sampling_rate * (win_size / 1000))
    # Step size in terms of samples
    step_size_in_samples = int(sampling_rate * (step_size / 1000))
    # Number of windows 
    n_windows = int(1 + np.floor((len(dec_data) - (win_size_in_samples - 1)) / step_size_in_samples))
    # Placeholder for spectrogram data after FFT
    spectro = np.zeros((win_size_in_samples, n_windows))
    # Frequency values of FFT
    fftfreqs = fftshift(fftfreq(win_size_in_samples, 1 / sampling_rate))
    # Slicing dec_data into small windows
    end_idx = 0
    # Define Hamming Window Function
    hamming_window = 0.54 - 0.46 * np.cos(np.array(range(win_size_in_samples)) * 2 * np.pi / win_size_in_samples)
    # Repeat for each window
    for i in range(n_windows):
        if i == (n_windows - 1):
            # Last sample, and make sure same duration.
            sliced_data = dec_data[-win_size_in_samples:]
            tmp.append(sliced_data)
        else:
            start_idx = int(i * step_size_in_samples)
            end_idx = int((i * step_size_in_samples) + win_size_in_samples)
            # Indexing
            sliced_data = dec_data[start_idx:end_idx]
            tmp.append(sliced_data)
        # Perform FFT of the sliced data 
        sliced_data_0 = hamming_window * sliced_data
        fft_data = fft(sliced_data_0)
        # Conversion of real and imaginary to absolute magnitude
        fft_data_mag = np.absolute(fft_data)
        # Convert to log scale
        fft_data_mag = 10 * np.log(fft_data_mag)
        # Append into spectogram data
        spectro[:, i] = fftshift(fft_data_mag)

    return spectro, fftfreqs

# function to visualize the spectrogram
def visualize_spectrogram(spectrogram_data, sampling_rate):
    """
    White meaning zero intensity
    Black meaning max intensity
    Each pixel is 10ms -> Along the x-axis.
    """
    n_freq_windows, n_time_window = spectrogram_data.shape

    plt.figure(figsize=(15, 6))
    # After fftshift, -ve freq is at the top row. Zero to +ve freq is at the bottom
    plt.imshow(spectrogram_data[int(n_freq_windows / 2):], origin='lower', cmap='Greys',
               extent=(0, 53863 / sampling_rate, 0, sampling_rate / 2 / 1000), aspect='auto')
    plt.colorbar()
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Time (s)')
    plt.title('Spectrogram of Audio Signal')
    plt.show()

# function to visualize the audio
def visualize_audiodata(raw_data, sampling_rate):
    np_dec_data = raw_data #np.asarray(np.asarray(raw_data) - 32768, dtype=np.int16)
    t = len(np_dec_data) / sampling_rate
    # get timestamps
    timestamps = np.arange(0, t, (t / len(np_dec_data)))
    # plot audio wave
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, np_dec_data)
    plt.xlabel('Time (s)')
    plt.ylabel('16-Bit Digital Value (Zero-Centered)')
    plt.title('Raw Audio File')
    plt.show()


# function to decode to dec
def decode_bit_to_dec(data, sample_width):
    """
    Sample Width - Number of byte to 1 data point.
    """
    tmp = []
    len_data = int(len(data) / sample_width)
    for i in range(len_data):
        val = int.from_bytes(data[((i + 1) * 2) - 2:(i + 1) * 2], byteorder='little', signed="True")
        tmp.append(val)
    return tmp


def main():
    audio_filename = sys.argv[1]
    win_size = sys.argv[2]
    step_size = sys.argv[3]
    # Reading the Wav. File
    dec_data, sampling_rate = read_wav_file(audio_filename, int(win_size), int(step_size))
    # FFT and obtaining intensity map of FFT.
    spectrogram_data, fftfreqs = FFT(dec_data, sampling_rate, int(win_size), int(step_size))
    # Visualizing Spectrogram
    visualize_spectrogram(spectrogram_data, sampling_rate)



if __name__ == "__main__":
    main()
