import nltk
from n_gram_corpus import TrigramCorpus
from nltk import trigrams


def grammar_checking(sentences,tc):
    # Tokenization and remove capital characters
    # tag each word in sentence using NLTK
    words_array = []

    for sentence in sentences:

        prob_multiplication = 1.0
        for w1, w2, w3 in trigrams((sentence.lower()).split(), pad_right=True, pad_left=True):
            if prob_multiplication == 0.0:
                break
            prob_multiplication *= tc.extract_probability(w3, (w1, w2))
            print((w3, (w1, w2)), tc.extract_probability(w3, (w1, w2)))

        words_array.append(prob_multiplication)

    print(words_array)


def grammar_checking_tagged(sentences,tc):
    # Tokenization and remove capital characters
    # tag each word in sentence using NLTK
    words_array = []

    for sentence in sentences:
        tags = nltk.pos_tag((sentence.lower()).split())
        tags_list = [x[1] for x in tags]
        prob_multiplication = 1.0
        for w1, w2, w3 in trigrams(tags_list, pad_right=True, pad_left=True):
            if prob_multiplication == 0.0:
                break
            prob_multiplication *= tc.extract_tagged_probability(w3, (w1, w2))
            print((w3, (w1, w2)), tc.extract_tagged_probability(w3, (w1, w2)))

        words_array.append(prob_multiplication)

    print(words_array)
