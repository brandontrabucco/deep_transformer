"""Author: Brandon Trabucco, Copyright 2019
Word Pack."""


import tensorflow as tf
from wordpack.net.transformer import Transformer


class BidirectionalTransformer(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding_size, num_heads, hidden_sizes, output_sizes, 
            fc_hidden_sizes, fc_output_sizes, **kwargs):
        super(BidirectionalTransformer, self).__init__(**kwargs)
        self.fw_transformer = Transformer(vocab_size, embedding_size, num_heads, hidden_sizes, output_sizes, 
            fc_hidden_sizes, fc_output_sizes, **kwargs)
        self.bw_transformer = Transformer(vocab_size, embedding_size, num_heads, hidden_sizes, output_sizes, 
            fc_hidden_sizes, fc_output_sizes, **kwargs)
        self.final_layer = tf.keras.layers.Dense(vocab_size)
    
    def __call__(self, x):
        return self.final_layer(tf.concat([
            self.fw_transformer(x), self.bw_transformer(tf.reverse(x, [1]))], 2))
        
    @property
    def trainable_variables(self):
        layer_variables = (self.fw_transformer.trainable_variables + 
            self.bw_transformer.trainable_variables[1:] + 
            self.final_layer.trainable_variables)
        return layer_variables
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        layer_variables = (self.fw_transformer.variables + 
            self.bw_transformer.variables[1:] + 
            self.final_layer.variables)
        return layer_variables
    
    @property
    def weights(self):
        return self.variables