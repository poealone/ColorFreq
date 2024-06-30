import pyaudio
import numpy as np
import matplotlib.pyplot as plt

# Constants
CHUNK = 1024  # Number of audio samples per frame
RATE = 44100  # Sampling rate in Hz

# Initialize PyAudio
p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Function to map frequency to wavelength in nm (using octave method)
def frequency_to_wavelength_octave(frequency):
    if frequency < 20 or frequency > 20000:
        return 0  # Out of audible range
    octave = int(np.log2(frequency / 20))
    base_frequency = 20 * 2**octave
    min_wavelength = 380 + (700 - 380) / 10 * octave
    max_wavelength = 380 + (700 - 380) / 10 * (octave + 1)
    norm_frequency = (frequency - base_frequency) / base_frequency
    wavelength = min_wavelength + norm_frequency * (max_wavelength - min_wavelength)
    return wavelength

# Function to map frequency to wavelength in nm (simple linear mapping)
def frequency_to_wavelength_simple(frequency):
    min_freq = 20
    max_freq = 20000
    min_wavelength = 380
    max_wavelength = 700
    norm_frequency = (frequency - min_freq) / (max_freq - min_freq)
    wavelength = max_wavelength - norm_frequency * (max_wavelength - min_wavelength)
    return wavelength

# Function to map frequency to wavelength in nm (starting from 440 Hz)
def frequency_to_wavelength_440hz(frequency):
    base_frequency = 440  # Starting frequency
    min_wavelength = 380
    max_wavelength = 700
    if frequency < base_frequency:
        return min_wavelength
    elif frequency > base_frequency * 2:
        return max_wavelength
    else:
        norm_frequency = (frequency - base_frequency) / (base_frequency)
        wavelength = min_wavelength + norm_frequency * (max_wavelength - min_wavelength)
        return wavelength

# Function to convert wavelength to RGB color
def wavelength_to_rgb(wavelength):
    gamma = 0.8
    intensity_max = 255
    factor = 0.0
    R = G = B = 0

    if (wavelength >= 380) and (wavelength < 440):
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif (wavelength >= 440) and (wavelength < 490):
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif (wavelength >= 490) and (wavelength < 510):
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif (wavelength >= 510) and (wavelength < 580):
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif (wavelength >= 580) and (wavelength < 645):
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif (wavelength >= 645) and (wavelength <= 700):
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = G = B = 0.0

    if (wavelength >= 380) and (wavelength < 420):
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif (wavelength >= 420) and (wavelength < 645):
        factor = 1.0
    elif (wavelength >= 645) and (wavelength <= 700):
        factor = 0.3 + 0.7 * (700 - wavelength) / (700 - 645)
    else:
        factor = 0.0

    R = round(intensity_max * (R * factor)**gamma)
    G = round(intensity_max * (G * factor)**gamma)
    B = round(intensity_max * (B * factor)**gamma)

    return (R / 255.0, G / 255.0, B / 255.0)

# Function to map frequency to color using the selected method
def frequency_to_color(frequency, method):
    if method == "octave":
        wavelength = frequency_to_wavelength_octave(frequency)
    elif method == "440hz":
        wavelength = frequency_to_wavelength_440hz(frequency)
    else:
        wavelength = frequency_to_wavelength_simple(frequency)
    color = wavelength_to_rgb(wavelength)
    return color

def main():
    print("Choose a method to map frequencies to colors:")
    print("1. Simple linear mapping")
    print("2. Octave-based harmonic mapping")
    print("3. 440 Hz-based mapping")
    choice = input("Enter 1, 2, or 3: ")

    if choice == "1":
        method = "simple"
    elif choice == "2":
        method = "octave"
    elif choice == "3":
        method = "440hz"
    else:
        print("Invalid choice. Using simple linear mapping.")
        method = "simple"

    try:
        print("Listening...")
        while True:
            # Read audio data
            data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
            # Perform FFT
            fft_data = np.fft.fft(data)
            freqs = np.fft.fftfreq(len(fft_data), 1/RATE)
            # Get the magnitude of the FFT
            magnitudes = np.abs(fft_data)
            # Find the peak frequency
            peak_freq = abs(freqs[np.argmax(magnitudes)])
            # Map the peak frequency to a color
            color = frequency_to_color(peak_freq, method)
            # Display the color
            plt.imshow([[color]])
            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.01)
            plt.clf()

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
