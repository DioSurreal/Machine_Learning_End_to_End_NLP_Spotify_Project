from nltk.tokenize import word_tokenize
import re
import json 
import numpy as np
from flask import jsonify

with open('model/vocab.json', 'r') as f:
    vocab = json.load(f)

def text_cleaning(text):
    text = re.sub(r'Ã[\x80-\xBF]+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    pattern = r"[^a-zA-Z0-9\s,']"
    text = re.sub(pattern, '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text.lower()


def tokenize_text(text):
    token = word_tokenize(text)
    return token


def tokens_to_indices(tokens, vocabs):
    tokens_indices = []
    for token in tokens: 
        if token in vocabs:
            tokens_indices.append(vocabs[token])
        else:
            vocabs[token] = len(vocabs)
            tokens_indices.append(vocabs[token])
            with open('vocab.json', 'w') as f:
                json.dump(vocabs, f) 
    return tokens_indices



def pad_features(reviews_idx, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    if len(reviews_idx) <= seq_length:
        zeroes = list(np.zeros(seq_length-len(reviews_idx)))
        new = zeroes+reviews_idx
        return np.array(new)
    elif len(reviews_idx) > seq_length:
        new = reviews_idx[0:seq_length]
        return np.array(new)



def preprocess_lstm(text):
    try:
        text_cleaned = text_cleaning(text)
        token  = tokenize_text(text_cleaned)
        token_idx = tokens_to_indices(token,vocab)
        features =pad_features(token_idx,287)
        return features 
    except Exception as e:
        return jsonify({"error": str(e)})   

# print(pad_features([1, 2, 3, 4, 5, 6, 7, 8, 9, 4, 10, 6, 11, 12, 13, 14, 15, 16, 9, 17, 18],287))