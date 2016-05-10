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
import models
import tensorflow as tf
from tensorflow.contrib import skflow

import data_utils


flags=tf.app.flags

# model configuration
flags.DEFINE_integer("vocab_size", 80000, "vocabulary size.")
flags.DEFINE_integer("embedding_size", 300, "Size of each model layer.")
flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("learning_rate_decay_op", 0.99, "Learning rate decay.")
flags.DEFINE_float("dropout_rate", 0.5, "dropout rates")
# flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
# flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
# flags.DEFINE_integer("max_len", 100, "sequence length longer than this will be ignored.")
# flags.DEFINE_integer("depth", 1, "Number of layers in the model.")
flags.DEFINE_string("data_dir", "bAbI_data/en", "Data directory")
flags.DEFINE_string("train_dir", "bAbI_data/en", "Training directory.")
flags.DEFINE_boolean("use_lstm", False, "Set True using LSTM, or False using GRU")
# flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
flags.DEFINE_integer("steps_per_checkpoint", 50, "How many training steps to do per checkpoint.")
# flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")

FLAGS = flags.FLAGS



def create_model(session, forward_only):

	model = dmn(FLAGS.vocab_size, FLAGS.embedding_size, FLAGS.learning_rate, 
		FLAGS.learning_rate_decay_op, FLAGS.dropout_rate, use_lstm=FLAGS.use_lstm)
	ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
	if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
		print("[*] Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("[*] Created model with fresh parameters.")
		session.run(tf.initialize_all_variables())
	return model

def train():
	train_data_path = glob.glob('%s/qa*_*_train.txt' % data_dir)
	print("[*] Preparing bAbI data ... in %s" % FLAGS.data.dir)
	bAbI_data, dictionary, rev_dictionary = data_utils.parse_babi_task(train_data_path, False)
	


	
	with tf.Session() as sess:
		


def test():
	"""Test the DMN model."""
	with tf.session() as sess:
		print("Self-test for neural translation model.")
		# Create model with vocab=10, embedding=10
		model = DMN.DMN(10,10,0.5,0.99,0.5)
		sess.run(tf.initialize_all_variables())

		# Fake data set
		data_set = 


def main(_):
	if FLAGS.train:
		train()



if __name__ == '__main__':
	tf.app.run()




