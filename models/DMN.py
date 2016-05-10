from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange 
import tensorflow as tf

import Question
import Input
import Answer
import Episodic

class DMN(object):
	"""
		Dynamic Memory Network: it contains four modules: Input, Question, Anwser, Episodic Memory
		check ref: Ask Me Anything: Dynamic Memory Networks for Natural Language Processing
		Args:
			vocab_size: vocabular size
			batch_size: batch size
			learning_rate: learning rate
			embedding_size: embedding size, also the RNN first layer size
			q_depth: question module layer depth
			
			i_depth: input layer depth (not include input fusion layer)
			
		Returns:


	"""
	def __init__(self, vocab_size, embedding_size, learning_rate, learning_rate_decay_op, memory_hops,
		, dropout_rate, maximum_length=20, maximum_sentence=10, use_lstm=False, forward_only=False):

		# initialization
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.learning_rate_decay_op = tf.Variable(float(learning_rate_decay_op), trainable=False)
		self.dropout_rate = dropout_rate
		self.global_step = tf.Variable(0, trainable=False, name='global_step')
		self.q_depth = q_depth
		self.memory_hops = memory_hops	# number of episodic memory pass
		self.facts =???	# inputs from Input modeule
		self.num_steps = num_sentences

		self.m_size = episodic_m_size # memory cell size
		self.m_depth = episodic_m_depth # memory cell depth
		self.ep_size # episodic cell size
		self.ep_depth
		self._ep_initial_state = cell.zero_state(batch_size, tf.float32)


		print("[*] Creating Dynamic Memory Network ...")
		

		# question module
		def seq2seq_f(encoder_inputs):
			return seq2seq.sentence_embedding_rnn(
				encoder_inputs, maximum_length, vocab_size, reader_cell, embedding_size)
		# attention gate in episodic
		def feedfoward_nn(l1_input):
			l2_input = tf.tanh(tf.matmul(l1_input * l1_weights) + l1_biases)
			logits = tf.matmul(l2_input * l2_weights) + l2_biases
			gate_prediction = tf.nn.softmax(logits)
			return gate_prediction


		# Sentence token placeholder
		self.story = tf.placeholder(tf.int32, [maximum_sentence, maximum_length])

		



		#------------ question module ------------
		single_cell = tf.nn.rnn_cell.GRUCell(embedding_size)
		if use_lstm:
			single_cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size)
		if not forward_only and dropout < 1:
			single_cell = tf.nn.rnn_cell.DropoutWrapper(
				single_cell, output_keep_prob=dropout)
		if q_depth > 1:
			question_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell])
		self.question = 
		#self.question = tf.placeholder(tf.int32,[n_question, n_length])


		#------------ Input module ------------
		reader_cell = tf.nn.rnn_cell.GRUCell(self.embedding_size)
		if use_lstm:
			reader_cell = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)
		if not forward_only and dropout < 1:
			reader_cell = tf.nn.rnn_cell.DropoutWrapper(
				reader_cell, output_keep_prob=dropout_rate)
		#embed toekn into vector, feed into rnn cell return cell state
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


		self.facts = rnn.bidirectional_rnn(fusion_fw_cell,fusion_bw_cell,
			lambda x: seq2seq_f(self.story))


		
		#------------ episodic memory module ------------
		# configuration of attention gate
		input_size = 
		l1_size = 
		l2_size = 
		mem_input_size = 
		with tf.variable_scope("episodic"):
			# parameters of attention gate
			l1_weights = tf.Variable(tf.truncated_normal([input_size, l1_size], -0.1, 0.1))
			l1_biases = tf.Variable(tf.zeros([l1_size]))
			l2_weights = tf.Variable(tf.truncated_normal([l1_size, l2_size], -0.1, 0.1))
			l2_biases = tf.Variable(tf.zeros([l2_size]))
			# paramters of episodic
			mem_weights = tf.Variable(tf.truncated_normal([mem_input_size, self.m_size], -0.1, 0.1))
			mem_biases = tf.Variable(tf.zeros([self.m_size]))

		# construct memory cell
		#single_cell = tf.nn.rnn_cell.BasicLSTMCell(self.mem_size)
		#single_cell = cell.MemCell(self.mem_size)
		single_cell = tf.nn.rnn_cell.GRUCell(self.mem_size)
		if m_depth > 1:
			mem_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * m_depth)

		# construct episodic_cell
		ep_cell = []
		for i in xrange(self.memory_hops):
			single_cell = cell.MGRUCell(self.ep_size)
			if ep_depth > 1:
				ep_cell.append(tf.nn.rnn_cell.MultiRNNCell([single_cell] * ep_depth))


		for hops in xrange(self.memory_hops):
			# gate attention network
			z = tf.concat(0, [tf.mul(facts, q),
				tf.mul(facts, mem_state), tf.abs(tf.sub(facts, q)), tf.abs(tf.sub(facts, mem_state))])
			episodic_gate = feedfoward_nn(z)
			# attention GRU
			output, context = cell.rnn(ep_cell[hops], facts, initial_state=self._ep_initial_state, 
				episodic_gate=episodic_gate, scope="epsodic")
			context_state.append(context)
			# memory updates
			if hops is 0:
				mem_state = question
			_, mem_state = mem_cell(context_state, question, mem_state)

		#------------ answer ------------




		loss = 

	def step(self, session, forward_only):


	def get_qns(self, data_set):
		"""Provide data set; return question and story"""
		

		
