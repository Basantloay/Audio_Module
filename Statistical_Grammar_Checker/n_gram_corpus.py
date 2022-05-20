import json
import os
from nltk import pos_tag
from nltk.corpus import reuters
from nltk import trigrams
from collections import defaultdict


def n_gram_probability():
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

    #print('**********************************************************')
    #print(probability_corpus[1])

    prob_ngram = os.path.join(os.path.dirname(__file__), "prob_ngram.json")
    with open(prob_ngram, 'w') as out:
        # by dividing freq by total frequencies we get probability
        for i in probability_corpus:
            summation = float(sum(probability_corpus[i].values()))
            for w3 in probability_corpus[i]:
                probability_corpus[i][w3] /= summation
                str1 = i, w3, probability_corpus[i][w3]
                out.write(json.dumps(str1,indent=1))
                #out.write(json.dumps('\n'))

    return probability_corpus


if __name__ == "__main__":
    output = n_gram_probability()
