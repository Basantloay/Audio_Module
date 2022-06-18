"""
    References :
        https://cloud.google.com/speech-to-text/docs/async-recognize#speech_transcribe_async_gcs-python
        https://cloud.google.com/storage/docs/creating-buckets#storage-create-bucket-code_samples
        https://cloud.google.com/storage/docs/streaming#code-samples

"""
from datetime import datetime
import requests
from speech_recognition import AudioData, RequestError, UnknownValueError
import moviepy.editor
import speech_recognition as sr
import os
from google.cloud import storage
from google.cloud import speech

credential = 'D:\GP\\affable-hall-343712-f18d4d6f8c88.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'D:\GP\\affable-hall-343712-f18d4d6f8c88.json'

GCS_URI = 'gs://pgraderbucket11'


def recognize_google_cloud(gcs_uri=GCS_URI):
    """Asynchronously transcribes the audio file specified by the gcs_uri."""
    from google.cloud import speech

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        audio_channel_count=2,
        language_code="en-US",
        enable_automatic_punctuation=True
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation of converting speech to text using GOOGLE API to complete...")
    response = operation.result(timeout=1000)

    transcript = []
    for result in response.results:
        transcript.append(result.alternatives[0].transcript.strip())
    return transcript


def video_to_audio_converter(video_name):
    # video = askopenfilename()
    video_name1 = video_name + '.mp4'
    video = moviepy.editor.VideoFileClip(video_name1)
    audio = video.audio

    audio.write_audiofile(video_name + '.wav')
    print('DONE!!!')
    return video_name + '.wav'


def upload_to_cloud(file_path):
    """
        saves a file in the google storage. As Google requires audio files greater than 60 seconds to be saved on cloud before processing
        It always saves in 'audio-files-bucket' (folder)
        Input:
            Path of file to be saved
        Output:
            URI of the saved file
    """
    print("Uploading to cloud...")
    client = storage.Client().from_service_account_json(credential)
    bucket = client.get_bucket('pgraderbucket11')
    file_name = str(file_path).split('\\')[-1]
    print(file_name)
    blob = bucket.blob(file_name)

    ## For slow upload speed
    storage.blob._DEFAULT_CHUNKSIZE = 5 * 1024 * 1024  # 100 MB
    storage.blob._MAX_MULTIPART_SIZE = 5 * 1024 * 1024  # 100 MB

    with open(file_path, 'rb') as f:
        blob.upload_from_file(f)
    print("uploaded at: ", "gs://pgraderbucket11/{}".format(file_name))
    return "gs://pgraderbucket11/{}".format(file_name)


def speech_to_text_converter(file_name,flag):
    requests.get('http://google.com', timeout=(10, 2000))
    audio_name = video_to_audio_converter(file_name)

    r = sr.Recognizer()
    with sr.WavFile(audio_name) as source:
        r.adjust_for_ambient_noise(source)

        audio = r.record(source)

        flac_data = audio.get_flac_data(
            convert_rate=None if 8000 <= audio.sample_rate <= 48000 else max(8000, min(audio.sample_rate, 48000)),
            # audio sample rate must be between 8 kHz and 48 kHz inclusive - clamp sample rate into this range
            convert_width=2  # audio samples must be 16-bit
        )

        audio = speech.RecognitionAudio(content=flac_data)

    audio_path = os.path.join(os.path.dirname(__file__), audio_name)
    print(audio_path)
    # upload_blob_from_stream('pgraderbucket11',audio_path,audio_name)
    t1 = datetime.now()
    if flag==0:
        upload_to_cloud(audio_path)
        print('Uploaded on Cloud Bucket')
    # recognize speech using google
    t2 = datetime.now()
    if flag==0:

        print('Total Time Taken to upload wav file on bucket in seconds : ', (t2 - t1).total_seconds())
    output_text = recognize_google_cloud(GCS_URI + '/' + audio_name)
    # print(transcription)
    # print("Done EL7MDULLAAAH \n ")

    print(output_text)
    print('Total Time get convert speech to text in seconds : ', (datetime.now() - t2).total_seconds())

    # write transcription
    with open(file_name + '_transcription.txt', "w") as f:
        f.write(output_text[0])
    return output_text


if __name__ == "__main__":
    speech_to_text_converter(file_name='video1')
