import json
import os
from nltk import pos_tag
from nltk.corpus import reuters
from nltk import trigrams
from collections import defaultdict

from datetime import datetime


class TrigramCorpus:
    probability_corpus_array = []

    def __init__(self):
        self.generate_n_gram_probability()

    def generate_n_gram_probability(self):

        probability_corpus = defaultdict(lambda: defaultdict(lambda: 0))

        # from sentence calculate freq of pos tags
        for sent in reuters.sents():
            tags = pos_tag(sent)
            y = 1
            tags_list = [x[y] for x in tags]
            # print(tags_list)

            for w1, w2, w3 in trigrams(tags_list, pad_right=True, pad_left=True):
                # print(w1, w2, w3)
                probability_corpus[(w1, w2)][w3] += 1

        # print('**********************************************************')
        # print(probability_corpus[1])

        prob_ngram = os.path.join(os.path.dirname(__file__), "prob_ngram.json")
        with open(prob_ngram, 'w') as out:
            # by dividing freq by total frequencies we get probability
            for i in probability_corpus:
                summation = float(sum(probability_corpus[i].values()))
                for w3 in probability_corpus[i]:
                    probability_corpus[i][w3] /= summation
                    str1 = i, w3, probability_corpus[i][w3]
                    self.probability_corpus_array.append(str1)
                    out.write(json.dumps(str1, indent=1))
                    # out.write(json.dumps('\n'))
        print(self.probability_corpus_array)

    def extract_probability(self):
        return


if __name__ == "__main__":
    tc = TrigramCorpus()
    t1 = datetime.now()
    print('Total Time Taken To Generate Probability corpus in seconds: ', (datetime.now() - t1).total_seconds())

