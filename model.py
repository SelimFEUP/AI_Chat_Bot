import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, Dense, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.models import Model
import numpy as np
from data_processing import CONFIG
from tensorflow.keras.layers import Lambda


def create_conv1d_model(vocab_size):
    # Encoder
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(vocab_size, CONFIG['embedding_dim'])(encoder_inputs)
    
    conv_branches = []
    for kernel_size in CONFIG['kernel_sizes']:
        conv = Conv1D(
            filters=CONFIG['num_filters'],
            kernel_size=kernel_size,
            activation='relu',
            padding='same'
        )(encoder_embedding)
        conv = GlobalMaxPooling1D()(conv)
        conv_branches.append(conv)
    
    encoder_output = Concatenate()(conv_branches) if len(conv_branches) > 1 else conv_branches[0]
    
    # Decoder
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(vocab_size, CONFIG['embedding_dim'])(decoder_inputs)
    
    encoder_output_expanded = Lambda(lambda x: tf.expand_dims(x, 1))(encoder_output)
    encoder_output_tiled = Lambda(lambda x: tf.tile(x[0], [1, tf.shape(x[1])[1], 1]))([encoder_output_expanded, decoder_embedding])
    decoder_combined = Concatenate()([decoder_embedding, encoder_output_tiled])
    
    decoder_conv = Conv1D(
        filters=CONFIG['num_filters'],
        kernel_size=3,
        activation='relu',
        padding='same'
    )(decoder_combined)
    
    decoder_output = Dense(vocab_size, activation='softmax')(decoder_conv)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

def prepare_training_data(input_sequences, target_sequences):
    decoder_input_data = target_sequences[:, :-1]
    decoder_target_data = target_sequences[:, 1:]
    return [input_sequences, decoder_input_data], np.expand_dims(decoder_target_data, -1)
