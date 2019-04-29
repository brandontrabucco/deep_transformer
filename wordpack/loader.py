"""Author: Brandon Trabucco, Copyright 2019
Word Pack."""


import tensorflow as tf
import nltk
import numpy as np


class Loader(object):

    def __init__(self, text_corpora, vocabulary, window_size=100, batch_size=32, num_epochs=1, random=True):
        tokenized_corpora = [nltk.tokenize.word_tokenize(corpus) for corpus in text_corpora]
        ids_corpora = [vocabulary.word_to_id(tokens) for tokens in tokenized_corpora]
        splits = np.zeros([0, 3], dtype=np.int32)
        for i, ids in enumerate(ids_corpora):
            splits = np.concatenate([splits, np.stack([np.full([len(ids) - window_size], i), 
                np.arange(len(ids) - window_size), 
                window_size + np.arange(len(ids) - window_size)], 1)], 0)
        self.iterator = self.get_iterator(splits, ids_corpora, window_size, batch_size, num_epochs, random)

    def get_iterator(self, splits, ids_corpora, window_size, batch_size, num_epochs, random):
        def split_to_ids_python(split):
            return np.array(ids_corpora[split[0]][split[1]:split[2]])
        def split_to_ids(x):
            return {"ids": tf.reshape(tf.py_function(split_to_ids_python, [x["splits"]], tf.int32), [window_size])}
        dataset = tf.data.Dataset.from_tensor_slices({"splits": splits})
        dataset = dataset.map(split_to_ids, num_parallel_calls=batch_size)
        if random:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(batch_size * 100, count=num_epochs))
        else:
            dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size).apply(tf.data.experimental.prefetch_to_device("/gpu:0", buffer_size=2))
        return dataset.make_initializable_iterator()

    def reset(self, sess):
        sess.run(self.iterator.initializer)

    def fetch(self):
        return self.iterator.get_next()
