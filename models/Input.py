from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
import numpy as np
from six.moves import xrange 
import tensorflow as tf
import seq2seq

class Input(object):
	"""
		Input module: it contains Input fusion layer (bidirectional RNN)

		Arg:

	"""
	def init_Input(self, num_layers, use_lstm, n_hidden ):

		# input module
		self.num_layers = config.num_layers
		self.use_lstm = False
		# tobe defined
		dropout
		learning_rate_decay_op
		learning_rate
		num_steps
		global_step
		embedding_size
		vocab_size
		length????
		n_sentence?
		state_size??
		#sentence reader cell
		reader_cell = tf.nn.rnn_cell.GRUCell(embedding_size)
		if use_lstm:
			reader_cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size)
		if not forward_only and dropout < 1:
			reader_cell = tf.nn.rnn_cell.DropoutWrapper(
				reader_cell, output_keep_prob=dropout)
		#embed toekn into vector, feed into rnn cell return cell state


		def seq2seq_f(encoder_inputs):
			return seq2seq.sentence_embedding_rnn(
				encoder_inputs, length, vocab_size, reader_cell, embedding_size)
		
		# Sentence token placeholder
		self.story = tf.placeholder(tf.int32, [n_sentence, length])

		fusion_fw_cell = tf.nn.rnn_cell.GRUCell(embedding_size)
		fusion_bw_cell = tf.nn.rnn_cell.GRUCell(embedding_size)
		if use_lstm:
			fusion_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size)
			fusion_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size)

		if not forward_only and dropout < 1:
			fusion_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
				fusion_fw_cell, output_keep_prob=dropout)
			fusion_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
				fusion_bw_cell, output_keep_prob=dropout)


		outputs = rnn.bidirectional_rnn(fusion_fw_cell,fusion_bw_cell,
			lambda x: seq2seq_f(self.story))

		# question module
		question_cell = tf.nn.rnn_cell.GRUCell(embedding_size)
		if use_lstm:
			question_cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size)
		if not forward_only and dropout < 1:
			question_cell = tf.nn.rnn_cell.DropoutWrapper(
				question_cell, output_keep_prob=dropout)

		self.question = tf.placeholder(tf.int32,[n_question, n_length])





		



	def step(self, session, config):



	def unsupervised_step(self, session, config):
		pass

	def supervised_step(self, session, config):
		pass