# Imports
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def grammar_check(phrases):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")

    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")

    # save model
    # model.save('./')

    # Tokenize text
    tokenized_phrases = tokenizer(phrases, return_tensors='pt', padding=True)

    # Perform corrections and decode the output
    corrections = model.generate(**tokenized_phrases)
    corrections = tokenizer.batch_decode(corrections, skip_special_tokens=True)

    # Print correction
    for i in range(len(corrections)):
        original, correction = phrases[i], corrections[i]
        print(f'[Phrase] {original}')
        print(f'[Suggested phrase] {correction}')
        print('~' * 100)


# Incorrect phrases
phrases = [
    'How is you doing?',
    'We is on the supermarket.',
    'Hello you be in school for lecture.']

if __name__ == "__main__":
    grammar_check(phrases)
