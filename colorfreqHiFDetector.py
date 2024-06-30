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

# Function to map frequency to wavelength in nm
def frequency_to_wavelength(frequency):
    # Visible light spectrum ranges from 380 nm (violet) to 700 nm (red)
    # Audio frequencies range from 20 Hz to 20 kHz
    min_freq = 20
    max_freq = 20000
    min_wavelength = 380
    max_wavelength = 700
    # Normalize the frequency to a 0-1 range
    norm_frequency = (frequency - min_freq) / (max_freq - min_freq)
    # Invert the normalized frequency to match the wavelength spectrum
    wavelength = max_wavelength - norm_frequency * (max_wavelength - min_wavelength)
    return wavelength

# Function to convert wavelength to RGB color
def wavelength_to_rgb(wavelength):
    # This function converts a wavelength in nm to an RGB color
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

    # Let the intensity fall off near the vision limits
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

    return (R, G, B)

# Function to map frequency to color
def frequency_to_color(frequency):
    wavelength = frequency_to_wavelength(frequency)
    color = wavelength_to_rgb(wavelength)
    return color

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
        color = frequency_to_color(peak_freq)
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
