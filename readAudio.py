import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from IPython.display import Audio
from numpy.fft import fft, ifft
import sounddevice as sd


fs = 44100  # Sample rate
seconds = 3  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording)  # Save as WAV file 



data = myrecording
data = data[:,0]

plt.figure()
plt.plot(data)
plt.xlabel("sample Index")
plt.ylabel("Amplitude")
plt.title("Audio signal ")
plt.show()