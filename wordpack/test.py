"""Author: Brandon Trabucco, Copyright 2019
Word Pack."""


from wordpack.loader import Loader
from wordpack.vocabulary import Vocabulary
import tensorflow as tf
import numpy as np
import os


def test(model, ckpt_dir, test_dir, window_size, batch_size):

    test_corpora_raw = []
    for filename in tf.gfile.Glob(os.path.join(test_dir, "*.txt")):
        with tf.gfile.FastGFile(filename, "r") as f:
            test_corpora_raw.append(f.read().strip().lower())
    with tf.gfile.FastGFile(os.path.join(ckpt_dir, "word.vocab"), "r") as f:
        reverse_vocab = [line.strip().lower() for line in f.readlines()]
        vocab = Vocabulary(reverse_vocab + ["<s>", "</s>", "<unk>"], "<s>", "</s>", "<unk>")

    sess = tf.Session()

    data_test = Loader(test_corpora_raw, vocab, window_size=window_size, batch_size=batch_size, 
        num_epochs=1, random=False)
    x_test = data_test.fetch()
    ids_test = x_test["ids"]

    logits_test = tf.clip_by_value(model(ids_test), -1e1, 1e1)

    tensor_indices = tf.stack([
        tf.tile(tf.expand_dims(tf.range(tf.shape(ids_test)[0]), 1), [1, tf.shape(ids_test)[1]]),
        tf.tile(tf.expand_dims(tf.range(tf.shape(ids_test)[1]), 0), [tf.shape(ids_test)[0], 1])], axis=2)

    selected_logits = tf.gather_nd(logits_test, tf.concat([tensor_indices, ids_test], 2))

    saver = tf.train.Saver(var_list=model.trainable_variables)
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

    logits = np.zeros([0, window_size])
    data_test.reset(sess)
    while True:
        try:
            logits = np.concatenate([logits, sess.run(selected_logits)], 0) 
        except tf.errors.OutOfRangeError:
            return logits
        