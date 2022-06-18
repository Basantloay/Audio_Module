import json
import os
from nltk import pos_tag
from nltk.corpus import reuters
from nltk import trigrams
from collections import defaultdict
import csv
from datetime import datetime
import pandas as pd


class TrigramCorpus:
    tagged_probability_corpus = {}
    probability_corpus = {}

    def __init__(self, used_training_dataset=2):
        """
            used_training_dataset = 1 for nltk.corpus.reuters
            used_training_dataset = 2 for CoLA
        """
        self.probability_corpus = defaultdict(lambda: defaultdict(lambda: 0))

        if not self.read_corpus_from_file():
            print('Generating probability of Trigram Corpus ')
            self.generate_n_gram_probability(used_training_dataset)

        self.tagged_probability_corpus = defaultdict(lambda: defaultdict(lambda: 0))

        if not self.read_corpus_from_tagged_file():
            print('Generating probability of TAGGED Trigram Corpus ')
            self.generate_n_gram_tagged_probability(used_training_dataset)

    def generate_n_gram_probability(self, used_training_dataset):
        # from sentence calculate freq of pos tags
        if used_training_dataset == 1:
            sents = reuters.sents()
        else:
            df = pd.read_csv("../cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None,
                             names=['sentence_source', 'label', 'label_notes', 'sentence'])

            sents = df.sentence.values
        #print(sents)
        for i in sents:
            if used_training_dataset==1:
                sent=i
            else:
                sent=i.split()
            sent = [x.lower() for x in sent]
            # print(sent)
            # print(tags_list)

            for w1, w2, w3 in trigrams(sent, pad_right=True, pad_left=True):
                # print(w1, w2, w3)
                self.probability_corpus[(w1, w2)][w3] += 1

        # print('**********************************************************')
        # print(self.probability_corpus[1])

        prob_ngram = os.path.join(os.path.dirname(__file__), "prob_ngram.csv")

        with open(prob_ngram, 'w', encoding='UTF8', newline='') as out:
            writer = csv.writer(out)
            # writer.writerow(['Tuple[0]','Tuple[1]', 'Prior', 'Prob'])
            # by dividing freq by total frequencies we get probability
            for i in self.probability_corpus:
                summation = float(sum(self.probability_corpus[i].values()))
                for w3 in self.probability_corpus[i]:
                    self.probability_corpus[i][w3] /= summation
                    str1 = [i[0], i[1], w3, self.probability_corpus[i][w3]]
                    # self.probability_corpus_array.append(str1)
                    # json.dump(str1, out)
                    # json.dump('\\n', out)
                    # pickle.dump(str1, out)
                    writer.writerow(str1)
            # out.write(json.dumps('\n'))
        # out.write(json.dumps(self.probability_corpus))
        # print(self.probability_corpus_array)

    def extract_probability(self, likelihood, posterior_tuple):

        return self.probability_corpus[posterior_tuple][likelihood]

    def read_corpus_from_file(self):
        print('Loading probability of trigram corpus from CSV')
        prob_ngram = os.path.join(os.path.dirname(__file__), "prob_ngram.csv")

        try:

            with open(prob_ngram, encoding="utf8") as reading:
                df = csv.reader(reading)
                count = 0

                for i in df:
                    count += 1
                    for j in range(2):
                        if i[j] == '':
                            i[j] = None
                    # print(i)
                    self.probability_corpus[(i[0], i[1])][i[2]] = float(i[3])
                if count < 2:
                    print('Probability Corpus is EMPTY !!!')
                    return False
                # print(self.probability_corpus)
                return True

        except IOError:
            print("File IS NOT FOUND ")
            return False

    def generate_n_gram_tagged_probability(self, used_training_dataset):
        # from sentence calculate freq of pos tags
        if used_training_dataset == 1:
            sents = reuters.sents()
        else:
            df = pd.read_csv("../cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None,
                             names=['sentence_source', 'label', 'label_notes', 'sentence'])
            sents = df.sentence.values
        for i in sents:
            if used_training_dataset==1:
                sent=i
            else:
                sent=i.split()
            tags = pos_tag(sent)
            y = 1
            tags_list = [x[y] for x in tags]
            # print(tags_list)

            for w1, w2, w3 in trigrams(tags_list, pad_right=True, pad_left=True):
                # print(w1, w2, w3)
                self.tagged_probability_corpus[(w1, w2)][w3] += 1

        prob_ngram = os.path.join(os.path.dirname(__file__), "prob_ngram_tagged.csv")

        with open(prob_ngram, 'w', encoding='UTF8', newline='') as out:
            writer = csv.writer(out)

            # by dividing freq by total frequencies we get probability
            for i in self.tagged_probability_corpus:
                summation = float(sum(self.tagged_probability_corpus[i].values()))
                for w3 in self.tagged_probability_corpus[i]:
                    self.tagged_probability_corpus[i][w3] /= summation
                    str1 = [i[0], i[1], w3, self.tagged_probability_corpus[i][w3]]

                    writer.writerow(str1)

    def extract_tagged_probability(self, likelihood, posterior_tuple):

        return self.tagged_probability_corpus[posterior_tuple][likelihood]

    def read_corpus_from_tagged_file(self):
        print('Loading probability of tagged trigram corpus from CSV')
        prob_ngram = os.path.join(os.path.dirname(__file__), "prob_ngram_tagged.csv")

        try:

            with open(prob_ngram, encoding="utf8") as reading:
                df = csv.reader(reading)
                count = 0

                for i in df:
                    count += 1
                    for j in range(3):
                        if i[j] == '':
                            i[j] = None
                    # print(i)
                    self.tagged_probability_corpus[(i[0], i[1])][i[2]] = float(i[3])
                if count < 2:
                    print('Tagged Probability Corpus is EMPTY !!!')
                    return False

                return True

        except IOError:
            print("File IS NOT FOUND ")
            return False


if __name__ == "__main__":
    t1 = datetime.now()
    tc = TrigramCorpus()

    print('Total Time Taken To Generate Probability corpus in seconds: ', (datetime.now() - t1).total_seconds())
    print(tc.extract_tagged_probability("CC", (None, None)))  # 0.026758787081208532
    print(tc.tagged_probability_corpus[("NN", '.')][None])
