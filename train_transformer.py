"""Author: Brandon Trabucco, Copyright 2019
Word Pack."""


from wordpack.train import train
from wordpack.net.bidirectional_transformer import BidirectionalTransformer
import tensorflow as tf
import numpy as np


tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_string("ckpt_dir", "./", "Output data directory.")
tf.flags.DEFINE_string("train_dir", "./train_data", "Path to the text files.")
tf.flags.DEFINE_string("val_dir", "./val_data", "Path to the text files.")
tf.flags.DEFINE_integer("window_size", 100, "Size of the temporal window for the model.")
tf.flags.DEFINE_integer("batch_size", 32, "Number of examples to process at once.")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of train loops through all corpora.")
FLAGS = tf.flags.FLAGS


if __name__ == "__main__":


    model = BidirectionalTransformer(7410, 512, [8, 8, 8], [32, 32, 32], [512, 512, 512], [2048, 2048, 2048], [512, 512, 512])
    train(model, FLAGS.ckpt_dir, FLAGS.train_dir, FLAGS.val_dir, FLAGS.window_size, FLAGS.batch_size, FLAGS.num_epochs)