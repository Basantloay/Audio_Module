"""
References:
    https://towardsdatascience.com/text-preprocessing-in-natural-language-processing-using-python-6113ff5decd8
    https://www.geeksforgeeks.org/part-speech-tagging-stop-words-using-nltk-python/
    https://www.tutorialspoint.com/natural_language_processing/natural_language_processing_part_of_speech_tagging.htm
    https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python/
    https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-language-model-nlp-python-code/
    https://nlpforhackers.io/language-models/

"""
import nltk
from n_gram_corpus import n_gram_probability

sentences = [
    'How is you doing?',
    'We is on the supermarket.',
    'Hello you be in school for lecture.']
# Tokenization and remove capital characters
# tag each word in sentence using NLTK
words_array = []
for sentence in sentences:
    words_array.append(nltk.pos_tag((sentence.lower()).split()))

print(words_array)


#generated_corpus=create_corpus()
