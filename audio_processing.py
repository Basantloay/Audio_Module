import moviepy.editor

import os
import torch
import librosa as lb
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from IPython.display import Audio

from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
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


# Load tokenizer and model

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained('./')
file_name = 'video1'
audio_name = video_to_audio_converter(file_name)

# Read the sound file
waveform, rate = lb.load(audio_name, sr=16000)

# Tokenize the waveform
input_values = tokenizer(waveform, return_tensors='pt').input_values

# Retrieve logits from the model
logits = model(input_values).logit

# Take argmax value and decode into transcription
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)

# Print the output
print(transcription)
