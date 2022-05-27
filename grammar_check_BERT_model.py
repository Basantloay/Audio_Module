"""
    References:
        https://huggingface.co/docs/transformers/model_doc/bert
        https://analyticsindiamag.com/how-to-use-bert-transformer-for-grammar-checking/
        https://github.com/amilanpathirana/Grammar-Check-using-BERT
        https://towardsdatascience.com/checking-grammar-with-bert-and-ulmfit-1f59c718fe75

"""
"""
import torch

from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification
import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from tqdm import tqdm, trange


def grammar_check_BERT(sentences):
    print(torch.cuda.is_available())
    # call BERT model configuration
    configur = BertConfig()

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # save model
    # model.save('./')

    # Tokenize text
    id_array = []
    mask_array = []
    training_dataset = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None,
                                   names=['sentence_source', 'label', 'label_notes', 'sentence'])

    #sentences = training_dataset.sentence.values

    #labels = training_dataset.label.values
    predictions = []
    #true_labels = []
    modelGED = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                             num_labels=2)

    # restore model
    #modelGED.load_state_dict(torch.load('bert-based-uncased-GED.pth'))
    modelGED.eval()
    for i in sentences:
        encoded_dict = tokenizer.encode_plus(
            i,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=128,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        id_array.append(encoded_dict['input_ids'])
        mask_array.append(encoded_dict['attention_mask'])

    print('*' * 100)
    prediction_inputs = torch.tensor(id_array)
    prediction_masks = torch.tensor(mask_array)
    #prediction_labels = torch.tensor(labels)
    with torch.no_grad():
        logits = modelGED(prediction_inputs, token_type_ids=None,
                          attention_mask=prediction_masks)

    logits = logits.detach().cpu().numpy()

    predictions.append(logits)

    flat_predictions = [item for sublist in predictions for item in sublist]

    prob_vals = flat_predictions
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    # flat_true_labels = [item for sublist in true_labels for item in sublist]
    #   print(flat_predictions)
    return flat_predictions, prob_vals


# Incorrect phrases
phrases = [
    'How is you doing?',
    'We is on the supermarket.',
    'Hello you be in school for lecture.']

if __name__ == "__main__":
    grammar_check_BERT(phrases)
"""

import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
import keras
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange

import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Check to confirm that GPU is available
device_name = tf.test.gpu_device_name()

print('Found GPU at: {}'.format(device_name))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

# Load the dataset into a pandas dataframe.
df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None,
                 names=['sentence_source', 'label', 'label_notes', 'sentence'])

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))

# Display 10 random rows from the data.
df.sample(10)

sentences = df.sentence.values
labels = df.label.values

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=64,  # Pad & truncate all sentences.

        padding='max_length',
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# print('Token IDs:', input_ids[0])


# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(
    train_dataset,  # The training samples.
    sampler=RandomSampler(train_dataset),  # Select batches randomly
    batch_size=32  # Trains with this batch size.
)

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
    val_dataset,  # The validation samples.
    sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
    batch_size=32  # Evaluate with this batch size.
)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=2,  # The number of output labels--2 for binary classification.
    # You can increase this for multi-class tasks.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)
params = list(model.named_parameters())
optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                  )

total_steps = len(train_dataloader) * 4

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

"""
keras.preprocessing.sequence.pad_sequences()
"""
