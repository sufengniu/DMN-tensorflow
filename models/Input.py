from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange 
import tensorflow as tf

class Input(object):
	"""
		Input module: it contains Input fusion layer (bidirectional RNN)

		Arg:

	"""
	def init_Input(self, num_layers, use_lstm, ):
		self.num_layers = config.num_layers
		self.use_lstm = False

		single_cell = tf.nn.rnn_cell.GRUCell(size)
		if use_lstm:
			single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
		if num_layers > 1:
			cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

		def seq2seq_f(encoder_inputs):
			return #todo

		self.encoder_inputs = []
		
	def step(self, session, config):



	def unsupervised_step(self, session, config):
		pass

	def supervised_step(self, session, config):
		pass