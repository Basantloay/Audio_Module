import moviepy.editor
from tkinter.filedialog import *
# from scipy.io import wavfile
# import noisereduce as nr
import os
import torch
import librosa
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from IPython.display import Audio

from transformers import AutoTokenizer, AutoConfig, Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import transformers as tfs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def video_to_audio_converter(video_name):
    # video = askopenfilename()
    video_name1 = video_name + '.mp4'
    video = moviepy.editor.VideoFileClip(video_name1)
    audio = video.audio

    audio.write_audiofile(video_name + '.wav')
    print('DONE!!!')
    return video_name + '.wav'


# noise reduction step
# rate, data = wavfile.read(file_name+'.wav')
# reduced_noise = nr.reduce_noise(y=data, sr=rate)
# wavefile.write("reduced_noise"+file_name+'.wav', rate, reduced_noise)

# Load tokenizer and model

#tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
#config = AutoConfig.from_pretrained("facebook/wav2vec2-base-960h")
#model = tfs.AutoModelForCTC.from_config(config)
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

file_name = 'video1'
audio_name = video_to_audio_converter(file_name)
Audio(audio_name)

data = wavfile.read(audio_name)
framerate = data[0]
sounddata = data[1]
time = np.arange(0, len(sounddata)) / framerate
input_audio, _ = librosa.load(audio_name, sr=16000)
input_values = tokenizer(input_audio, return_tensors="pt").input_values
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)[0]
transcription
