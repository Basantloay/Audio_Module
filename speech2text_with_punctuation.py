"""
    References :
        https://cloud.google.com/speech-to-text/docs/async-recognize#speech_transcribe_async_gcs-python
        https://cloud.google.com/storage/docs/creating-buckets#storage-create-bucket-code_samples
        https://cloud.google.com/storage/docs/streaming#code-samples

"""
import requests
from speech_recognition import AudioData, RequestError, UnknownValueError
import moviepy.editor
import speech_recognition as sr
import os
from google.cloud import storage

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'D:\GP\\affable-hall-343712-f18d4d6f8c88.json'

GCS_URI='gs://pgraderbucket11'
def create_bucket_class_location(bucket_name):
    """
    Create a new bucket in the US region with the coldline storage
    class
    """
    # bucket_name = "your-new-bucket-name"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    bucket.storage_class = "COLDLINE"
    new_bucket = storage_client.create_bucket(bucket, location="us")

    print(
        "Created bucket {} in {} with storage class {}".format(
            new_bucket.name, new_bucket.location, new_bucket.storage_class
        )
    )
    return new_bucket


def recognize_google_cloud(gcs_uri=GCS_URI):
    """Asynchronously transcribes the audio file specified by the gcs_uri."""
    from google.cloud import speech

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
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


def upload_blob_from_stream(bucket_name='pgraderbucket11', file_obj=None, destination_blob_name="audiofile"):

    storage.blob._DEFAULT_CHUNKSIZE = 100 * 1024 * 1024  # 100 MB
    storage.blob._MAX_MULTIPART_SIZE = 100 * 1024 * 1024  # 100 MB
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Rewind the stream to the beginning. This step can be omitted if the input
    # stream will always be at a correct position.

    # Upload data from the stream to your bucket.
    blob.upload_from_filename(file_obj)

    print(
        f"Stream data uploaded to {destination_blob_name} in bucket {bucket_name}."
    )


def speech_to_text_converter(file_name):
    requests.get('http://google.com', timeout=(10, 200))
    audio_name = video_to_audio_converter(file_name)
    audio_path = os.path.join(os.path.dirname(__file__), audio_name)
    print(audio_path)
    upload_blob_from_stream('pgraderbucket11',audio_path,audio_name)
    print('Uploaded on Cloud Bucket')
    # recognize speech using google
    output_text = recognize_google_cloud(GCS_URI+'/'+audio_name)
            # print(transcription)
    print("Done EL7MDULLAAAH \n ")

    print(output_text)
        # write transcription
        #with open(file_name + '_transcription.txt', "w") as f:
         #   f.write(output_text)


if __name__ == "__main__":
    speech_to_text_converter(file_name='video1')
