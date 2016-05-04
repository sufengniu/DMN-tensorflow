from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import os
import random
import sys
import time
import re

import numpy as np

import tensorflow as tf
from tensorflow.contrib import skflow

import data_utils


flags=tf.app.flags

# model configuration
flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("dropout", 0.8, "dropout rates")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
flags.DEFINE_integer("batch_size", 4, "Batch size to use during training.")
flags.DEFINE_integer("max_len", 100, "sequence length longer than this will be ignored.")
flags.DEFINE_integer("embedding_size", 16, "Size of each model layer.")
flags.DEFINE_integer("depth", 1, "Number of layers in the model.")
flags.DEFINE_integer("vocab_size", 80000, "vocabulary size.")
flags.DEFINE_string("data_dir", "bAbI_data/en", "Data directory")
flags.DEFINE_string("train_dir", "data", "Training directory.")
flags.DEFINE_string("model_types", "GRU", "RNN model types (LSTM, GRU)")
flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
flags.DEFINE_integer("steps_per_checkpoint", 10, "How many training steps to do per checkpoint.")
flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
flags.DEFINE_boolean("validation", False, "Run validation if true")

FLAGS = flags.FLAGS



def create_model():



def train():



def test():



def main(_):
	if FLAGS.train:




if __name__ == '__main__':
	tf.app.run()




