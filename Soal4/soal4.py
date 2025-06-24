import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as write_wav, read as read_wav
import sounddevice as sd
from scipy.signal import correlate

SAMPLE_RATE = 48000  
DURASI = 5          
FREKUENSI = 440      
AMPLITUDO = 0.5      

SEED_MONTH = 8
SEED_DATE = 27
WATERMARK_SEED = SEED_MONTH * 100 + SEED_DATE 

WEIGHT_LOW = 0.005 
WEIGHT_HIGH = 0.02 


def create_sine_wave(frekuensi, sample_rate, durasi, amplitudo):
    t = np.linspace(0, durasi, int(sample_rate * durasi), endpoint=False)
    audio_signal = amplitudo * np.sin(2 * np.pi * frekuensi * t)
    return audio_signal

main_signal = create_sine_wave(FREKUENSI, SAMPLE_RATE, DURASI, AMPLITUDO)
main_signal_int16 = np.int16(main_signal * 32767)

original_wav_path = "sinusoida_440hz.wav"
write_wav(original_wav_path, SAMPLE_RATE, main_signal_int16)


def create_pn_sequence(length, seed):
    np.random.seed(seed) 
    pn_sequence = np.random.choice([-1, 1], size=length)
    return pn_sequence

pn_sequence = create_pn_sequence(len(main_signal), WATERMARK_SEED)

def embed_watermark(host_signal, pn_sequence, weight):
    if len(host_signal) != len(pn_sequence):
        raise ValueError("Panjang host_signal dan pn_sequence harus sama")

    watermarked_signal = host_signal + (weight * pn_sequence)

    watermarked_signal = np.clip(watermarked_signal, -1.0, 1.0)
    return watermarked_signal


watermarked_signal_low_w = embed_watermark(main_signal, pn_sequence, WEIGHT_LOW)
watermarked_signal_low_w_int16 = np.int16(watermarked_signal_low_w * 32767)
low_w_wav_path = f"low_watermark({WEIGHT_LOW}).wav"
write_wav(low_w_wav_path, SAMPLE_RATE, watermarked_signal_low_w_int16)

watermarked_signal_high_w = embed_watermark(main_signal, pn_sequence, WEIGHT_HIGH)
watermarked_signal_high_w_int16 = np.int16(watermarked_signal_high_w * 32767)
high_w_wav_path = f"high_watermark({WEIGHT_HIGH}).wav"
write_wav(high_w_wav_path, SAMPLE_RATE, watermarked_signal_high_w_int16)


def detect_watermark(received_signal, original_pn_sequence):
    correlation = correlate(received_signal, original_pn_sequence, mode='valid')
    return correlation[0] / len(original_pn_sequence) 


print("\nDeteksi Watermark")

correlation_original = detect_watermark(main_signal, pn_sequence)
print(f"Korelasi pada sinyal ASLI (tanpa watermark): {correlation_original:.6f}")

correlation_low_w = detect_watermark(watermarked_signal_low_w, pn_sequence)
print(f"Korelasi pada sinyal dengan watermark bobot RENDAH ({WEIGHT_LOW}): {correlation_low_w:.6f}")

correlation_high_w = detect_watermark(watermarked_signal_high_w, pn_sequence)
print(f"Korelasi pada sinyal dengan watermark bobot TINGGI ({WEIGHT_HIGH}): {correlation_high_w:.6f}")

fake_pn_sequence = create_pn_sequence(len(main_signal), 999) 
correlation_fake_pn = detect_watermark(watermarked_signal_high_w, fake_pn_sequence)
print(f"Korelasi pada sinyal ber-watermark (bobot tinggi) dengan PN PALSU: {correlation_fake_pn:.6f}")

THRESHOLD = 0.001
print(f"\nAmbang batas deteksi (THRESHOLD) = {THRESHOLD}")

print(f"Deteksi Watermark pada Sinyal Asli: {'Terdeteksi' if abs(correlation_original) > THRESHOLD else 'Tidak Terdeteksi'}")
print(f"Deteksi Watermark pada Sinyal Low Weight: {'Terdeteksi' if abs(correlation_low_w) > THRESHOLD else 'Tidak Terdeteksi'}")
print(f"Deteksi Watermark pada Sinyal High Weight: {'Terdeteksi' if abs(correlation_high_w) > THRESHOLD else 'Tidak Terdeteksi'}")
print(f"Deteksi Watermark pada Sinyal Ber-Watermark (High Weight) dengan PN PALSU: {'Terdeteksi' if abs(correlation_fake_pn) > THRESHOLD else 'Tidak Terdeteksi'}")


print("\n--- Menampilkan Grafik Waveform ---")

plot_duration = 0.05 
num_samples_to_plot = int(plot_duration * SAMPLE_RATE)

if num_samples_to_plot > len(main_signal):
    num_samples_to_plot = len(main_signal)

time_plot = np.linspace(0, plot_duration, num_samples_to_plot, endpoint=False)


plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(time_plot, main_signal[:num_samples_to_plot], color='blue') 
plt.title(f'Sinyal Asli - Sinus {FREKUENSI} Hz (Tampilan Dekat)')
plt.xlabel('Waktu (s)')
plt.ylabel('Amplitudo')
plt.grid(True)
plt.ylim(-1.1, 1.1)
plt.xlim(0, plot_duration) 

plt.subplot(3, 1, 2)
plt.plot(time_plot, watermarked_signal_low_w[:num_samples_to_plot], color='green') 
plt.title(f'Sinyal dengan Watermark (Bobot {WEIGHT_LOW}) (Tampilan Dekat)')
plt.xlabel('Waktu (s)')
plt.ylabel('Amplitudo')
plt.grid(True)
plt.ylim(-1.1, 1.1)
plt.xlim(0, plot_duration)


plt.subplot(3, 1, 3)
plt.plot(time_plot, watermarked_signal_high_w[:num_samples_to_plot], color='red') 
plt.title(f'Sinyal dengan Watermark (Bobot {WEIGHT_HIGH}) (Tampilan Dekat)')
plt.xlabel('Waktu (s)')
plt.ylabel('Amplitudo')
plt.grid(True)
plt.ylim(-1.1, 1.1)
plt.xlim(0, plot_duration)

plt.tight_layout()
plt.show()