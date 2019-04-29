"""Author: Brandon Trabucco, Copyright 2019
Word Pack."""


from wordpack.loader import Loader
from wordpack.vocabulary import Vocabulary
import tensorflow as tf
import numpy as np
import os


def train(model, ckpt_dir, train_dir, val_dir, window_size, batch_size, num_epochs):

    train_corpora_raw = []
    for filename in tf.gfile.Glob(os.path.join(train_dir, "*.txt")):
        with tf.gfile.FastGFile(filename, "r") as f:
            train_corpora_raw.append(f.read().strip().lower())
    val_corpora_raw = []
    for filename in tf.gfile.Glob(os.path.join(val_dir, "*.txt")):
        with tf.gfile.FastGFile(filename, "r") as f:
            val_corpora_raw.append(f.read().strip().lower())
    with tf.gfile.FastGFile(os.path.join(ckpt_dir, "word.vocab"), "r") as f:
        reverse_vocab = [line.strip().lower() for line in f.readlines()]
        vocab = Vocabulary(reverse_vocab + ["<s>", "</s>", "<unk>"], "<s>", "</s>", "<unk>")

    sess = tf.Session()

    data_train = Loader(train_corpora_raw, vocab, window_size=window_size, batch_size=batch_size, 
        num_epochs=num_epochs, random=True)
    x_train = data_train.fetch()
    ids_train = x_train["ids"]

    data_val = Loader(val_corpora_raw, vocab, window_size=window_size, batch_size=batch_size, 
        num_epochs=1, random=False)
    x_val = data_val.fetch()
    ids_val = x_val["ids"]

    logits_train = tf.clip_by_value(model(ids_train), -1e1, 1e1)
    logits_val = tf.clip_by_value(model(ids_val), -1e1, 1e1)

    loss_train = tf.losses.sparse_softmax_cross_entropy(ids_train, logits_train)
    loss_val = tf.losses.sparse_softmax_cross_entropy(ids_val, logits_val)

    mask_train = tf.where(tf.equal(ids_train, tf.argmax(logits_train, axis=2, output_type=tf.int32)),
        tf.ones([tf.shape(ids_train)[0], tf.shape(ids_train)[1]]), 
        tf.zeros([tf.shape(ids_train)[0], tf.shape(ids_train)[1]]))
    mask_val = tf.where(tf.equal(ids_val, tf.argmax(logits_val, axis=2, output_type=tf.int32)),
        tf.ones([tf.shape(ids_val)[0], tf.shape(ids_val)[1]]), 
        tf.zeros([tf.shape(ids_val)[0], tf.shape(ids_val)[1]]))

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(0.00001, global_step, 1000, 1.0)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    learning_step = tf.group(optimizer.minimize(loss_train, global_step=global_step, 
        var_list=model.trainable_variables), tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    sess.run(tf.variables_initializer(optimizer.variables()))
    saver = tf.train.Saver(var_list=model.trainable_variables)
    if not tf.gfile.IsDirectory(ckpt_dir):
        tf.gfile.MakeDirs(ckpt_dir)
    model_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if model_ckpt is not None:
        saver.restore(sess, model_ckpt)
    else:
        sess.run(tf.variables_initializer(model.trainable_variables + [global_step]))

    def validate_model():
        correct, count = 0.0, 0.0
        data_val.reset(sess)
        while True:
            try:
                accuracy_mask = sess.run(mask_val)
                correct += float(np.sum(accuracy_mask))
                count += float(np.size(accuracy_mask))
            except tf.errors.OutOfRangeError:
                return correct / count

    losses, accuracies = [], []
    best_accuracy, best_iteration = validate_model(), sess.run(global_step)
    data_train.reset(sess)
    while True:
        try:
            iteration, loss, _step = sess.run([global_step, loss_train, learning_step])
        except tf.errors.OutOfRangeError:
            return losses, accuracies, best_accuracy, best_iteration
        losses.append(float(loss))
        print("Iteration {0} train loss was {1:.5f}".format(iteration, loss))
        if iteration % 100 == 0:
            accuracy = validate_model()
            accuracies.append(float(accuracy))
            if accuracy > best_accuracy:
                best_accuracy, best_iteration = accuracy, iteration
                saver.save(sess, os.path.join(ckpt_dir, "model.ckpt"), global_step=global_step)
            print("Iteration {0} best accuracy was {1:.5f} at iteration {2}".format(iteration, best_accuracy, best_iteration))
        elif iteration - best_iteration > 5000:
            return losses, accuracies, best_accuracy, best_iteration
        