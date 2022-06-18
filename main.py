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
from Statistical_Grammar_Checker.n_gram_corpus import TrigramCorpus
from Statistical_Grammar_Checker.algo import grammar_checking, grammar_checking_tagged


from speech2text_with_punctuation import *
import re

if __name__ == '__main__':
    text1 = speech_to_text_converter(file_name='video1',flag=1)
    text1[0] = re.split(r'\?|;|\.', text1[0])
    print(len(text1[0]))
    print(text1[0])
    tc = TrigramCorpus(1)
    sentences = [
        'How are you doing ?',
        'We are on the supermarket .',
        'Hello you be in school for lecture .']
    t1 = datetime.now()
    print(grammar_checking(text1[0], tc))
    print(grammar_checking_tagged(text1[0], tc))
    print('Total Time Taken to check Grammar in seconds: ', (datetime.now() - t1).total_seconds())
