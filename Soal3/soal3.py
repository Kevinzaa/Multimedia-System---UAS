import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import os

# wav = "guitar.wav"
# mp3 = "guitar(mp3).mp3"

# try:
#     audio = AudioSegment.from_wav(wav)
#     audio.export(mp3, format="mp3") 

# except FileNotFoundError:
#     print(f"Error: File '{wav}' tidak ditemukan. Pastikan file ada di direktori yang sama.")
# except Exception as e:
#     print(f"Terjadi kesalahan: {e}")


input_wav_file = "guitar.wav"
output_mp3_file = "guitar.mp3"

print("\n--- Perbandingan Ukuran File ---")
try:
    size_wav = os.path.getsize(input_wav_file) / (1024 * 1024) # MB
    size_mp3 = os.path.getsize(output_mp3_file) / (1024 * 1024) # MB
    print(f"Ukuran file '{input_wav_file}': {size_wav:.2f} MB")
    print(f"Ukuran file '{output_mp3_file}': {size_mp3:.2f} MB")
    print(f"Penghematan ukuran: {((size_wav - size_mp3) / size_wav * 100):.2f}%")
except FileNotFoundError:
    print("Pastikan kedua file (WAV dan MP3) ada di direktori yang sama.")
    exit()

def load_audio_data(filepath):
    if filepath.endswith('.wav'):
        sample_rate, data = wavfile.read(filepath)
    elif filepath.endswith('.mp3'):
        audio = AudioSegment.from_mp3(filepath)
        data = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            data = data.reshape((-1, 2))
            data = data[:, 0]
        sample_rate = audio.frame_rate
    else:
        raise ValueError("Format file tidak didukung. Gunakan .wav atau .mp3")
    return sample_rate, data

print("\n--- Memuat Data Audio untuk Analisis ---")
sr_wav, data_wav = load_audio_data(input_wav_file)
sr_mp3, data_mp3 = load_audio_data(output_mp3_file)

if sr_wav != sr_mp3:
    print(f"Peringatan: Sample rate WAV ({sr_wav}) dan MP3 ({sr_mp3}) berbeda. Mungkin ada konversi otomatis.")

max_samples_for_plot = int(10 * sr_wav) 
if len(data_wav) > max_samples_for_plot:
    data_wav_plot = data_wav[:max_samples_for_plot]
    time_wav_plot = np.arange(len(data_wav_plot)) / sr_wav
else:
    data_wav_plot = data_wav
    time_wav_plot = np.arange(len(data_wav_plot)) / sr_wav

if len(data_mp3) > max_samples_for_plot:
    data_mp3_plot = data_mp3[:max_samples_for_plot]
    time_mp3_plot = np.arange(len(data_mp3_plot)) / sr_mp3
else:
    data_mp3_plot = data_mp3
    time_mp3_plot = np.arange(len(data_mp3_plot)) / sr_mp3


print("\n--- Visualisasi Gelombang (Waveform) ---")
plt.figure(figsize=(15, 6))

plt.subplot(2, 1, 1)
plt.plot(time_wav_plot, data_wav_plot, color='blue')
plt.title(f'Waveform Asli ({input_wav_file})')
plt.xlabel('Waktu (s)')
plt.ylabel('Amplitudo')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_mp3_plot, data_mp3_plot, color='red')
plt.title(f'Waveform Terkompresi ({output_mp3_file})')
plt.xlabel('Waktu (s)')
plt.ylabel('Amplitudo')
plt.tight_layout()
plt.show()

print("\n--- Spektrum Frekuensi ---")

def plot_spectrum(data, sample_rate, title, color):
    N = len(data)
    yf = np.fft.fft(data)
    xf = np.fft.fftfreq(N, 1 / sample_rate)
    
    plt.plot(xf[:N//2], 2.0/N * np.abs(yf[:N//2]), color=color)
    plt.title(title)
    plt.xlabel('Frekuensi (Hz)')
    plt.ylabel('Amplitudo')
    plt.grid(True)
    plt.xlim(0, sample_rate / 2) 
    plt.ylim(bottom=0)

plt.figure(figsize=(15, 6))

plt.subplot(2, 1, 1)
plot_spectrum(data_wav, sr_wav, f'Spektrum Frekuensi Asli ({input_wav_file})', 'blue')

plt.subplot(2, 1, 2)
plot_spectrum(data_mp3, sr_mp3, f'Spektrum Frekuensi Terkompresi ({output_mp3_file})', 'red')

plt.tight_layout()
plt.show()