"""
References:
    https://towardsdatascience.com/text-preprocessing-in-natural-language-processing-using-python-6113ff5decd8
    https://www.geeksforgeeks.org/part-speech-tagging-stop-words-using-nltk-python/
    https://www.tutorialspoint.com/natural_language_processing/natural_language_processing_part_of_speech_tagging.htm
    https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python/
    https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-language-model-nlp-python-code/
    https://nlpforhackers.io/language-models/

"""

# from datasets import load_dataset
# dataset = load_dataset("jfleg")
from datetime import datetime

from algo import grammar_checking

if __name__ == '__main__':

    sentences = [
        'How is you doing ?',
        'We are on the supermarket .',
        'Hello you be in school for lecture .']
    t1 = datetime.now()
    grammar_checking(sentences)
    print('Total Time Taken to check Grammar in seconds: ', (datetime.now() - t1).total_seconds())