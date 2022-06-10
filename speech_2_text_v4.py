import io
import os
import sys
import subprocess
import wave
import aifc
import math
import audioop
import collections
import json
import base64
import threading
import platform
import stat
import hashlib
import hmac
import time
import uuid

from speech_recognition import AudioData, RequestError, UnknownValueError
import moviepy.editor
import speech_recognition as sr

try:  # attempt to use the Python 2 modules
    from urllib import urlencode
    from urllib2 import Request, urlopen, URLError, HTTPError
except ImportError:  # use the Python 3 modules
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError


true=True
false=False
def recognize_google_version(audio_data, key=None, language="en-US", pfilter=0, show_all=False):
    """
    Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the Google Speech Recognition API.
    The Google Speech Recognition API key is specified by ``key``. If not specified, it uses a generic key that works out of the box. This should generally be used for personal or testing purposes only, as it **may be revoked by Google at any time**.
    To obtain your own API key, simply following the steps on the `API Keys <http://www.chromium.org/developers/how-tos/api-keys>`__ page at the Chromium Developers site. In the Google Developers Console, Google Speech Recognition is listed as "Speech API".
    The recognition language is determined by ``language``, an RFC5646 language tag like ``"en-US"`` (US English) or ``"fr-FR"`` (International French), defaulting to US English. A list of supported language tags can be found in this `StackOverflow answer <http://stackoverflow.com/a/14302134>`__.
    The profanity filter level can be adjusted with ``pfilter``: 0 - No filter, 1 - Only shows the first character and replaces the rest with asterisks. The default is level 0.
    Returns the most likely transcription if ``show_all`` is false (the default). Otherwise, returns the raw API response as a JSON dictionary.
    Raises a ``speech_recognition.UnknownValueError`` exception if the speech is unintelligible. Raises a ``speech_recognition.RequestError`` exception if the speech recognition operation failed, if the key isn't valid, or if there is no internet connection.
    """
    assert isinstance(audio_data, AudioData), "``audio_data`` must be audio data"
    assert key is None or isinstance(key, str), "``key`` must be ``None`` or a string"
    assert isinstance(language, str), "``language`` must be a string"
    punctuation = "True"
    flac_data = audio_data.get_flac_data(
        convert_rate=None if audio_data.sample_rate >= 8000 else 8000,  # audio samples must be at least 8 kHz
        convert_width=2  # audio samples must be 16-bit
    )
    if key is None:
        key = "AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw"

    url = "http://www.google.com/speech-api/v2/recognize?{}".format(urlencode({
        "client": "chromium",
        "key": key,
        "lang": "en-US",
        "pFilter": pfilter

    }))

    request = Request(url, data=flac_data,
                      headers={"Content-Type": "audio/x-flac; rate={}".format(audio_data.sample_rate)})

    # obtain audio transcription results
    try:
        response = urlopen(request, timeout=None)

    except HTTPError as e:
        raise RequestError("recognition request failed: {}".format(e.reason))
    except URLError as e:
        raise RequestError("recognition connection failed: {}".format(e.reason))
    response_text = response.read().decode("utf-8")
    #print(response_text)
    # ignore any blank blocks
    actual_result = []
    for line in response_text.split("\n"):
        print(line)
        if not line: continue
        result = json.loads(line)["result"]
        if len(result) != 0:
            actual_result = result[0]
            break

    # return results
    if show_all: return actual_result
    if not isinstance(actual_result, dict) or len(actual_result.get("alternative", [])) == 0: raise UnknownValueError()

    if "confidence" in actual_result["alternative"]:
        # return alternative with highest confidence score
        best_hypothesis = max(actual_result["alternative"], key=lambda alternative: alternative["confidence"])
    else:
        # when there is no confidence available, we arbitrarily choose the first hypothesis.
        best_hypothesis = actual_result["alternative"][0]
    if "transcript" not in best_hypothesis: raise UnknownValueError()
    return best_hypothesis["transcript"]


"""
https://geekscoders.com/python-speech-recognition-tutorial-for-beginners/
"""




def video_to_audio_converter(video_name):
    # video = askopenfilename()
    video_name1 = video_name + '.mp4'
    video = moviepy.editor.VideoFileClip(video_name1)
    audio = video.audio

    audio.write_audiofile(video_name + '.wav')
    print('DONE!!!')
    return video_name + '.wav'


def speech_to_text_converter(file_name):
    r = sr.Recognizer()
    audio_name = video_to_audio_converter(file_name)

    with sr.WavFile(audio_name) as source:
        r.adjust_for_ambient_noise(source)

        audio = r.record(source)

        # recognize speech using google

        try:
            output_text = recognize_google_version(audio)
            # print(transcription)
            print("Done EL7MDULLAAAH \n ")


        except Exception as e:
            print("Error :  " + str(e))

        # write transcription
        with open(file_name + '_transcription.txt', "w") as f:
            f.write(output_text)


if __name__ == "__main__":
    speech_to_text_converter(file_name='video1_Trim')
