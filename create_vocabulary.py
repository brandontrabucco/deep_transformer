"""Author: Brandon Trabucco, Copyright 2019
Word Pack."""


from wordpack.vocabulary import Vocabulary
import tensorflow as tf
import numpy as np
import os
import nltk


tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_string("corpora_dir", "./", "Path to the text files.")
tf.flags.DEFINE_string("output_dir", "./", "Output data directory.")
tf.flags.DEFINE_integer("min_word_frequency", 5, "Min word occurences to be in the vocabulary.")
FLAGS = tf.flags.FLAGS


if __name__ == "__main__":

    if not tf.gfile.IsDirectory(FLAGS.corpora_dir):
        tf.gfile.MakeDirs(FLAGS.corpora_dir)
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    corpora_raw = []
    for filename in tf.gfile.Glob(os.path.join(FLAGS.corpora_dir, "*.txt")):
        with tf.gfile.FastGFile(filename, "r") as f:
            corpora_raw.append(f.read().strip().lower())
    corpora_words = [nltk.tokenize.word_tokenize(corpus) for corpus in corpora_raw]

    word_frequencies = {}
    for corpus in corpora_words:
        for word in corpus:
            if word not in word_frequencies:
                word_frequencies[word] = 0
            word_frequencies[word] += 1
            
    unique_words, counts = list(zip(*list(sorted(
        word_frequencies.items(), key=(lambda x: x[1]), reverse=True))))

    split_index = 0
    for i, c in enumerate(counts):
        if c < FLAGS.min_word_frequency:
            split_index = i - 1
            break

    reverse_vocab = unique_words[:split_index]
    with tf.gfile.FastGFile(os.path.join(FLAGS.output_dir, "word.vocab"), "w") as f:
        for word in reverse_vocab:
            f.write("{}\n".format(word))