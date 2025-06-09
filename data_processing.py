import os
import re
import urllib.request
import zipfile
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

CONFIG = {
    'max_samples': 100000,
    'batch_size': 32,
    'epochs': 100,
    'embedding_dim': 256,
    'max_vocab_size': 10000,
    'max_seq_length': 30,
    'num_filters': 128,
    'kernel_sizes': [3, 5, 7]
}

def download_and_prepare_data():
    url = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
    zip_path = "ai_chat_platform/cornell_movie_dialogs_corpus.zip"
    extract_dir = "ai_chat_platform/cornell_movie_dialogs_corpus"
    
    if not os.path.exists(extract_dir):
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, zip_path)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        os.remove(zip_path)

    movie_lines_path = os.path.join(extract_dir, "cornell movie-dialogs corpus", "movie_lines.txt")
    movie_conversations_path = os.path.join(extract_dir, "cornell movie-dialogs corpus", "movie_conversations.txt")

    id2line = {}
    with open(movie_lines_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.strip().split(' +++$+++ ')
            if len(parts) == 5:
                id2line[parts[0]] = parts[4]

    conversations = []
    with open(movie_conversations_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.strip().split(' +++$+++ ')
            if len(parts) == 4:
                line_ids = eval(parts[3])
                for i in range(len(line_ids) - 1):
                    q = id2line.get(line_ids[i], '').lower()
                    a = id2line.get(line_ids[i+1], '').lower()
                    if q and a:
                        conversations.append((q, a))

    return conversations[:CONFIG['max_samples']]


def preprocess_text(text):
    """Enhanced and safe text preprocessing."""
    text = text.lower()

    contractions = {
        r"\bi'm\b": "i am", r"\bhe's\b": "he is", r"\bshe's\b": "she is",
        r"\bit's\b": "it is", r"\bthat's\b": "that is", r"\bwhat's\b": "what is",
        r"\bwhere's\b": "where is", r"\bwon't\b": "will not", r"\bcan't\b": "cannot",
        r"n't\b": " not", r"'re\b": " are", r"'ll\b": " will",
        r"'ve\b": " have", r"'d\b": " would"
    }

    for pattern, repl in contractions.items():
        text = re.sub(pattern, repl, text)

    # Replace numbers and clean punctuation
    text = re.sub(r'\d+', ' <num> ', text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_data(conversations):
    input_texts = []
    target_texts = []

    for input_text, target_text in conversations:
        input_text = preprocess_text(input_text)
        target_text = preprocess_text(target_text)

        if len(input_text.split()) <= CONFIG['max_seq_length'] and \
           len(target_text.split()) <= CONFIG['max_seq_length']:
            input_texts.append(input_text)
            target_texts.append(f"<start> {target_text} <end>")

    return input_texts, target_texts


def tokenize_texts(input_texts, target_texts):
    tokenizer = Tokenizer(
        num_words=CONFIG['max_vocab_size'],
        filters='',
        lower=True,
        oov_token='<unk>'
    )
    tokenizer.fit_on_texts(input_texts + target_texts)

    # Ensure <start> and <end> are included
    for token in ['<start>', '<end>']:
        if token not in tokenizer.word_index:
            tokenizer.word_index[token] = len(tokenizer.word_index) + 1

    tokenizer.index_word = {v: k for k, v in tokenizer.word_index.items()}

    input_sequences = tokenizer.texts_to_sequences(input_texts)
    target_sequences = tokenizer.texts_to_sequences(target_texts)

    input_sequences = pad_sequences(
        input_sequences,
        maxlen=CONFIG['max_seq_length'],
        padding='post',
        truncating='post'
    )
    target_sequences = pad_sequences(
        target_sequences,
        maxlen=CONFIG['max_seq_length'] + 1,
        padding='post',
        truncating='post'
    )

    return input_sequences, target_sequences, tokenizer

