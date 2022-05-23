"""
    References:
        https://huggingface.co/docs/transformers/model_doc/bert
        https://analyticsindiamag.com/how-to-use-bert-transformer-for-grammar-checking/
        https://github.com/amilanpathirana/Grammar-Check-using-BERT
"""
import torch

from transformers import BertModel, BertConfig, BertTokenizer,BertForSequenceClassification,
import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from tqdm import tqdm , trange



def grammar_check_BERT(phrases):
    # call BERT model configuration
    configur = BertConfig()

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load the model
    model = BertModel(configur)

    # save model
    # model.save('./')

    # Tokenize text
    id_array = []
    mask_array = []
    df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None,
                     names=['sentence_source', 'label', 'label_notes', 'sentence'])

    sentences = df.sentence.values
    labels = df.label.values

    for i in sentences:
        encoded_dict = tokenizer.encode_plus(
            i,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=64,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        id_array.append(encoded_dict['input_ids'])
        mask_array.append(encoded_dict['attention_mask'])

    print('*' * 100)
    mask_array = torch.cat(mask_array, dim=0)
    id_array = torch.cat(id_array, dim=0)
    labels = torch.tensor(labels)


# Incorrect phrases
phrases = [
    'How is you doing?',
    'We is on the supermarket.',
    'Hello you be in school for lecture.']

if __name__ == "__main__":
    grammar_check_BERT(phrases)
