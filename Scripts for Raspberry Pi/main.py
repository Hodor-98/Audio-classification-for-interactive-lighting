import sounddevice as sd
import numpy as np
import scipy.signal
import timeit
import python_speech_features
import pandas as pd

from tflite_runtime.interpreter import Interpreter

df = pd.read_csv (r'yamnet_class_map.csv')


# Parameters
debug_time = 1
debug_acc = 1
debug_type = 1
word_threshold = 0.5
rec_duration = 0.5
window_stride = 0.5
sample_rate = 48000
resample_rate = 16000
num_channels = 1
#model_path = 'wake_word_stop_lite.tflite' #to be changed to YAMNET
#model_path = 'lite-model_yamnet_classification_tflite_1.tflite'
model_path = 'yamnet.tflite'
word_flag = 0


# Sliding window
window = np.zeros(int(rec_duration * resample_rate) * 2)

# Load model (interpreter)
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

# Decimate (filter and downsample)
def decimate(signal, old_fs, new_fs):

    # Check to make sure we're downsampling
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs

    # We can only downsample by an integer factor
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only decimate by integer factor")
        return signal, old_fs

    # Do decimation
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))

    return resampled_signal, new_fs

# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):

    global word_flag

    # Start timing for testing
    start = timeit.default_timer()

    # Notify if errors
    if status:
        print('Error:', status)

    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)

    # Resample
    rec, new_fs = decimate(rec, sample_rate, resample_rate)

    # Save recording onto sliding window
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec

    # Ajust lenght
    waveform = window[:15600]

    waveform = [element*10 for element in waveform]
#    print(waveform)    
#    print(max(waveform))

    # Make prediction from model
    in_tensor = np.expand_dims(np.array(waveform, dtype=np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    if debug_type:
        max_index_col = np.argmax(output_data, axis=1)
        print(df.iloc[max_index_col])


    if max_index_col == 349:
        print('Doorbell recognized')
        word_flag = 1

    if debug_acc:
        print(np.amax(output_data))

    if debug_time:
        print(timeit.default_timer() - start)

# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while word_flag == 0:
        pass
    print("Done!")
 
