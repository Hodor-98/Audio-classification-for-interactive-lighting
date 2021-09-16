import librosa.display
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Read urban sound dataset

data = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
data.head(5)

# Select sound samples larger than 3 seconds
valid_data = data[['slice_file_name', 'fold', 'classID', 'class']][data['end']-data['start'] >= 3]

# Librosa spectrogram example
y, sr = librosa.load('UrbanSound8K/audio/fold5/6508-9-0-1.wav', duration=2.97)
ps = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(ps, y_axis='mel', x_axis='time')
plt.show()

y, sr = librosa.load('UrbanSound8K/audio/fold5/43787-3-0-0.wav', duration=2.97)
ps = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(ps, y_axis='mel', x_axis='time')
plt.show()