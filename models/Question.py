from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange 
import tensorflow as tf

class Question(object):
	"""
		Question module:
		Args:
			embedding_size: RNN input size

		Return:
			
	"""
	#def __init__(self, config, forward_only=False):
	def init_Question(self, embedding_size, num_steps, forward_only=False):

		self.num_steps = num_steps
		self.size = embedding_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.learning_rate_decay_op = self.learning_rate.assign(
			self.learning_rate * learning_rate_decay_factor)
		self.global_step = tf.Variable(0, trainable=False, name='global_step')

		# placeholder
		self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
		
		if config.rnn_types is 'GRU':
			single_cell = tf.nn.rnn_cell.GRUCell(self.size)
		elif config.rnn_types is 'LSTM':
			single_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
		else:
			print("model types not found!")
			sys.exit(1)

		if not forward_only and dropout < 1:
			single_cell = tf.nn.rnn_cell.DropoutWrapper(
				single_cell, output_keep_prob=config.dropout)
		cell = single_cell
		cell = tf.nn.rnn_cell.MultiRNNCell([single_cell])

		# embedding is shared with input module
		# with tf.variable_scope("EMBEDDING", reuse=True):
		# 	embedding = tf.get_variable("embedding", [config.vocab_size, self.size])
		# 	inputs = tf.nn.embedding_lookup(embedding, self._input_data)
			
		self._initial_state = cell.zero_state(batch_size, tf.float32)

		if not forward_only and config.dropout < 1:
			inputs = tf.nn.dropout(inputs, config.dropout)

		from tensorflow.models.rnn import rnn
		inputs = [tf.squeeze(input_, [1])
			for input_ in tf.split(1, self.num_steps, inputs)]
		state = self._initial_state
		with tf.variable_scope("RNN"):
			outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)


	def step(self, session, forward_only=True):
		
		if not forward_only:

		else:


