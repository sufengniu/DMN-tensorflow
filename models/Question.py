from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange 
import tensorflow as tf

class Question(object):
	#def __init__(self, config, forward_only=False):
	def init_Question(self, config, forward_only=False):
		self.vocab_size = config.vocab_size
		self.batch_size = config.batch_size
		self.num_steps = 
		self.size = 
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.learning_rate_decay_op = self.learning_rate.assign(
			self.learning_rate * learning_rate_decay_factor)
		self.global_step = tf.Variable(0, trainable=False, name='global_step')

		# placeholder
		self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
		

		if config.rnn_types is 'GRU':
			single_cell = tf.nn.rnn_cell.GRUCell(size)
		elif config.rnn_types is 'LSTM':
			single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
		else:
			print("model types not found!")
			sys.exit(1)

		if not forward_only and dropout < 1:
			single_cell = tf.nn.rnn_cell.DropoutWrapper(
				single_cell, output_keep_prob=dropout)
		cell = single_cell
		cell = tf.nn.rnn_cell.MultiRNNCell([single_cell])

		with tf.variable_scope("", reuse=True):
			embedding = tf.get_variable("embedding", [config.vocab_size, self.size])
			inputs = tf.nn.embedding_lookup(embedding, self._input_data)



	def step(self, session, config, forward_only=True):
		
