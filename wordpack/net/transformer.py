"""Author: Brandon Trabucco, Copyright 2019
Word Pack."""


import tensorflow as tf
from wordpack.net.multi_head_attention import MultiHeadAttention


class Transformer(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding_size, num_heads, hidden_sizes, output_sizes, 
            fc_hidden_sizes, fc_output_sizes, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE):
            self.embeddings_map = tf.get_variable("embeddings_map", shape=[
                vocab_size, embedding_size], dtype=tf.float32)
        self.attention_layers = [MultiHeadAttention(a, b, c) for a, b, c in zip(
            num_heads, hidden_sizes, output_sizes)]
        self.fc_hidden_layers = [tf.keras.layers.Dense(x) for x in fc_hidden_sizes]
        self.fc_output_layers = [tf.keras.layers.Dense(x) for x in fc_output_sizes]
    
    def __call__(self, x):
        x = tf.nn.embedding_lookup(self.embeddings_map, x)
        for attention, hidden, output in zip(self.attention_layers, self.fc_hidden_layers,
                self.fc_output_layers):
            x = output(tf.nn.relu(hidden(attention(x, x, x))))
        return x
        
    @property
    def trainable_variables(self):
        layer_variables = [self.embeddings_map]
        for layer in self.attention_layers:
            layer_variables += layer.trainable_variables
        for layer in self.fc_hidden_layers:
            layer_variables += layer.trainable_variables
        for layer in self.fc_output_layers:
            layer_variables += layer.trainable_variables
        return layer_variables
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        layer_variables = [self.embeddings_map]
        for layer in self.attention_layers:
            layer_variables += layer.variables
        for layer in self.fc_hidden_layers:
            layer_variables += layer.variables
        for layer in self.fc_output_layers:
            layer_variables += layer.variables
        return layer_variables
    
    @property
    def weights(self):
        return self.variables