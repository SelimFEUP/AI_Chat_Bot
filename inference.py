import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from data_processing import CONFIG, preprocess_text
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Lambda

def create_inference_model(model):
    # Encoder inference model
    encoder_inputs = model.get_layer('input_layer').output
    encoder_embedding = model.get_layer('embedding')(encoder_inputs)
    
    # Reconstruct encoder branches
    conv_branches = []
    conv_layers = [
        model.get_layer('conv1d'),
        model.get_layer('conv1d_1'),
        model.get_layer('conv1d_2')
    ]
    pooling_layers = [
        model.get_layer('global_max_pooling1d'),
        model.get_layer('global_max_pooling1d_1'),
        model.get_layer('global_max_pooling1d_2')
    ]
    
    for conv_layer, pooling_layer in zip(conv_layers, pooling_layers):
        conv_output = conv_layer(encoder_embedding)
        pooled_output = pooling_layer(conv_output)
        conv_branches.append(pooled_output)
    
    encoder_output = model.get_layer('concatenate')(conv_branches) if len(conv_branches) > 1 else conv_branches[0]
    encoder_model = Model(encoder_inputs, encoder_output)
    
    # Decoder inference model
    decoder_inputs = Input(shape=(None,))
    decoder_state_input = Input(shape=(encoder_output.shape[-1],))
    
    # Get decoder embedding
    decoder_embedding = model.get_layer('embedding_1')(decoder_inputs)
    
    # Wrap TensorFlow operations in Lambda layers
    expand_layer = Lambda(lambda x: tf.expand_dims(x, 1))
    encoder_output_expanded = expand_layer(decoder_state_input)
    
    tile_layer = Lambda(lambda x: tf.tile(x[0], [1, tf.shape(x[1])[1], 1]))
    encoder_output_tiled = tile_layer([encoder_output_expanded, decoder_embedding])
    
    # Use the original concatenate layer
    decoder_combined = model.get_layer('concatenate_1')([decoder_embedding, encoder_output_tiled])
    
    # Get decoder conv layer
    decoder_conv = model.get_layer('conv1d_3')(decoder_combined)
    
    # Final output
    decoder_output = model.get_layer('dense')(decoder_conv)
    
    decoder_model = Model(
        inputs=[decoder_inputs, decoder_state_input],
        outputs=[decoder_output, decoder_state_input]
    )
    
    return encoder_model, decoder_model

def chat_with_bot(encoder_model, decoder_model, tokenizer):
    print("\n=== DEBUG MODE ===")
    print("Available tokens:", list(tokenizer.word_index.items())[:10], "...")
    
    while True:
        input_sentence = input("\nYou: ")
        if input_sentence.lower() in ['exit', 'quit']:
            break
        
        # Tokenization debug
        input_seq = tokenizer.texts_to_sequences([input_sentence])
        print(f"Tokenized: {input_seq}")
        
        if not input_seq or not any(input_seq[0]):
            print("Warning: Empty sequence after tokenization!")
            print("Try simpler input or check tokenizer")
            continue
            
        input_seq = pad_sequences(input_seq, maxlen=CONFIG['max_seq_length'], padding='post')
        print(f"Padded: {input_seq}")
        
        # Encoder debug
        states_value = encoder_model.predict(input_seq)
        print(f"Encoder output: min={np.min(states_value):.4f}, max={np.max(states_value):.4f}")
        
        # Decoder process
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = tokenizer.word_index['<start>']
        decoded_sentence = []
        
        for i in range(CONFIG['max_seq_length']):
            output_tokens, states_value = decoder_model.predict([target_seq, states_value])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            
            # Get the word
            sampled_word = next(
                (word for word, idx in tokenizer.word_index.items() if idx == sampled_token_index),
                '<UNK>'
            )
            
            print(f"Step {i}: Predicted '{sampled_word}' (idx={sampled_token_index})")
            
            if sampled_word == '<end>':
                break
                
            if sampled_word not in ['<start>', '<end>', '<unk>']:
                decoded_sentence.append(sampled_word)
                
            target_seq[0, 0] = sampled_token_index
        
        print("\nBot:", ' '.join(decoded_sentence) if decoded_sentence else "<no meaningful response>")
