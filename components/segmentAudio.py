import torch
import silero_vad
import cv2
import io
import numpy as np
import pandas as pd
import pyaudio
import wave
import opensmile
import threading

from silero_vad import (
    load_silero_vad,
    read_audio,
    get_speech_timestamps,
    save_audio,
    VADIterator,
    collect_chunks,
)

record = True


def stop():
    print("\n")
    input('Press "q" to stop recording.')
    global record
    record = False


# Outfile param, FIX: make parametric to the CLI
OUTFILE_NAME = "output.wav"
CHANNELS = 1  # Mono audio


# yoinked from https://github.com/snakers4/silero-vad/blob/master/examples/pyaudio-streaming/pyaudio-streaming-examples.ipynb
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound


# Siler vad parameters

SAMPLING_RATE = 16000
USE_PIP = True

USE_ONNX = False  # for later when we use ONNX runtime

model = load_silero_vad(onnx=USE_ONNX)


# PyAudio setup
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = int(SAMPLING_RATE / 10)

audio = pyaudio.PyAudio()


# setting up recording

data = []
audio_confidences = []


n_samples = 512

frame_to_record = 50

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLING_RATE,
    input=True,
    frames_per_buffer=CHUNK,
)


stop_listener = threading.Thread(target=stop)
stop_listener.start()


while record:
    audio_chunk = stream.read(n_samples)

    data.append(audio_chunk)

    audio_int16 = np.frombuffer(audio_chunk, np.int16)
    audio_float32 = int2float(audio_int16)

    confidence = model(torch.from_numpy(audio_float32), SAMPLING_RATE).item()
    audio_confidences.append(confidence)


stream.stop_stream()
stream.close()
audio.terminate()


print("Recording Stopped.")

with wave.open(OUTFILE_NAME, "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(SAMPLING_RATE)
    wf.writeframes(b"".join(data))

# Transcribing

# OpenSmile Docs: https://audeering.github.io/opensmile-python/

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)
y = smile.process_file(OUTFILE_NAME)  # saves as pandas df

y.to_csv("output_df.csv", index=False)
# print(f"Transcription: {}")
print(y)
