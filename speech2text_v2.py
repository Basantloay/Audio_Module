"""
https://geekscoders.com/python-speech-recognition-tutorial-for-beginners/
"""

import moviepy.editor
import speech_recognition as sr


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
            transcription = r.recognize_google(audio)
            # print(transcription)
            print("Done EL7MDULLAAAH \n ")


        except Exception as e:
            print("Error :  " + str(e))

        # write transcription
        with open(file_name + '_transcription.txt', "w") as f:
            f.write(transcription)


if __name__ == "__main__":
    speech_to_text_converter(file_name='video1')
