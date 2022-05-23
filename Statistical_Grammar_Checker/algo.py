import nltk
from n_gram_corpus import TrigramCorpus
from nltk import trigrams


def grammar_checking(sentences):
    # Tokenization and remove capital characters
    # tag each word in sentence using NLTK
    words_array = []
    tc = TrigramCorpus()
    for sentence in sentences:
        tags = nltk.pos_tag((sentence.lower()).split())
        tags_list = [x[1] for x in tags]
        prob_multiplication = 1.0
        for w1, w2, w3 in trigrams(tags_list):
            if prob_multiplication == 0.0:
                break
            prob_multiplication *= tc.extract_probability(w3, (w1, w2))
            print((w3, (w1, w2)),tc.extract_probability(w3, (w1, w2)))

        words_array.append(prob_multiplication)

    print(words_array)
