from data_processing import *
from model import create_conv1d_model, prepare_training_data
from inference import create_inference_model, chat_with_bot
import numpy as np
import tensorflow as tf
import random

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


def main():
    # Data preparation
    conversations = download_and_prepare_data()
    input_texts, target_texts = prepare_data(conversations)
    input_sequences, target_sequences, tokenizer = tokenize_texts(input_texts, target_texts)
    vocab_size = min(CONFIG['max_vocab_size'], len(tokenizer.word_index) + 1)
    
    # Create and train model
    print("Creating model...")
    model = create_conv1d_model(vocab_size)
    model.summary()
    
    print("Preparing training data...")
    train_data, train_labels = prepare_training_data(input_sequences, target_sequences)
    
    print("Training model...")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
    mc = tf.keras.callbacks.ModelCheckpoint('ai_chat_platform/model.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    model.fit(
        train_data,
        train_labels,
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        validation_split=0.2, callbacks=[mc,early_stopping]
    )


    model.load_weights('ai_chat_platform/model.keras')
    # Create inference models
    encoder_model, decoder_model = create_inference_model(model)
    
    # Start chatting
    print("Chat with the bot (type 'exit' or 'quit' to end)")
    chat_with_bot(encoder_model, decoder_model, tokenizer)

if __name__ == "__main__":
    main()
