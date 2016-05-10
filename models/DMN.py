from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange 
import tensorflow as tf

import seq2seq
import cell

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
			a_depth: answer module layer depth
			m_depth: memory cell depth
			i_depth: input layer depth (not include input fusion layer)
			memory_hops: how many hops for episodic memory module


		Returns:


	"""
	def __init__(self, vocab_size, embedding_size, learning_rate, learning_rate_decay_op, memory_hops,
		, dropout_rate, maximum_story_length=50, maximum_question_length=30, use_lstm=False, forward_only=False):

		# initialization
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.learning_rate_decay_op = tf.Variable(float(learning_rate_decay_op), trainable=False)
		self.dropout_rate = dropout_rate
		self.global_step = tf.Variable(0, trainable=False, name='global_step')
		self.q_depth = q_depth	# question RNN depth
		self.a_depth = a_depth	# answer RNN depth
		self.m_depth = episodic_m_depth # memory cell depth
		self.ep_depth = ep_depth	# episodic depth
		
		self.memory_hops = memory_hops	# number of episodic memory pass
		self.num_steps = num_sentences

		self.m_input_size = m_input_size
		self.m_size = episodic_m_size # memory cell size
		self.ep_size # episodic cell size
		self._ep_initial_state = cell.zero_state(batch_size, tf.float32)


		
		print("[*] Creating Dynamic Memory Network ...")
		# question module
		def seq2seq_f(encoder_inputs, mask=None):
			return seq2seq.sentence_embedding_rnn(
				encoder_inputs, mask, vocab_size, reader_cell, embedding_size)
		# attention gate in episodic
		def feedfoward_nn(l1_input):
			with tf.variable_scope("episodic"):
				l1_weights = get_variable("l1_weights", [input_size, l1_size])
				l1_biases = get_variable("l1_biases", [l1_size])
				l2_weights = get_variable("l2_weights", [l1_size, l2_size])
				l2_biases = get_variable("l2_biases", [l2_size])
			l2_input = tf.tanh(tf.matmul(l1_input * l1_weights) + l1_biases)
			logits = tf.matmul(l2_input * l2_weights) + l2_biases
			gate_prediction = tf.nn.softmax(logits)
			return gate_prediction


		# Sentence token placeholder
		self.story = []
		for i in range(maximum_story_length):
			self.story.append(tf.placeholder(tf.int32))
		self.story_mask = placeholder(tf.int32)
		self.question = []
		for i in range(maximum_question_length):
			self.question.append(tf.placeholder(tf.int32))

		# configuration of attention gate
		input_size = 
		l1_size = 
		l2_size =
		with tf.variable_scope("episodic"):
			# parameters of attention gate
			l1_weights = tf.Variable(tf.truncated_normal([input_size, l1_size], -0.1, 0.1), name="l1_weights")
			l1_biases = tf.Variable(tf.zeros([l1_size]), name="l1_biases")
			l2_weights = tf.Variable(tf.truncated_normal([l1_size, l2_size], -0.1, 0.1), name="l2_weights")
			l2_biases = tf.Variable(tf.zeros([l2_size]), name="l2_biases")
			# paramters of episodic
			mem_weights = tf.Variable(tf.truncated_normal([self.m_input_size, self.m_size], -0.1, 0.1), name="mem_weights")
			mem_biases = tf.Variable(tf.zeros([self.m_size]), name="mem_biases")

		with tf.variable_scope("answer"):
			softmax_weights = tf.Variable(tf.truncated_normal([, ], -0.1, 0.1), name="softmax_weights")


		


		#------------ question module ------------
		single_cell = tf.nn.rnn_cell.GRUCell(embedding_size)
		if use_lstm:
			single_cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size)
		if not forward_only and dropout < 1:
			single_cell = tf.nn.rnn_cell.DropoutWrapper(
				single_cell, output_keep_prob=dropout)
		question_cell = single_cell
		if q_depth > 1:
			question_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell]*q_depth)
		for question in	seq2seq.seq2seq_f(self.question):
			self.question_state = question
		#self.question = tf.placeholder(tf.int32,[n_question, n_length])


		#------------ Input module ------------
		reader_cell = tf.nn.rnn_cell.GRUCell(self.embedding_size)
		if use_lstm:
			reader_cell = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)
		if not forward_only and dropout < 1:
			reader_cell = tf.nn.rnn_cell.DropoutWrapper(
				reader_cell, output_keep_prob=dropout_rate)
		# Embedded toekn into vector, feed into rnn cell return cell state
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
			ep_cell.append(tf.nn.rnn_cell.MultiRNNCell([single_cell] * ep_depth))


		for hops in xrange(self.memory_hops):
			# gate attention network
			z = tf.concat(0, [tf.mul(self.facts, q), tf.mul(self.facts, mem_state), 
				tf.abs(tf.sub(self.facts, q)), tf.abs(tf.sub(self.facts, mem_state))])
			episodic_gate = feedfoward_nn(z)
			# attention GRU
			output, context = cell.rnn(ep_cell[hops], self.facts, episodic_gate, initial_state=self._ep_initial_state, 
				scope="epsodic")
			context_state.append(context)
			# memory updates
			if hops is 0:
				mem_state = self.question_state
			#_, mem_state = mem_cell(context_state, mem_state)	# GRU
			_, mem_state = mem_cell(context_state, self.question_state, mem_state, self.m_input_size, self.m_size)

		#------------ answer ------------
		# TODO: use decoder sequence to generate answer
		single_cell = tf.nn.rnn_cell.GRUCell(self.mem_size)
		answer_cell = single_cell
		if a_depth > 1:
			answer_cell =tf.nn.rnn_cell.MultiRNNCell([single_cell] * a_depth)
		
		a_state = mem_state
		for step in range(answer_steps):
			y = tf.nn.softmax(tf.matmul(softmax_weights, a_state))
			(answer, a_state) = answer_cell(tf.concat(0, [self.question_state, y]), a_state)
			#(answer, a_state) = answer_cell(tf.concat(0, [question, mem_state]), a_state)

		logits = 

		loss = 

	def step(self, session, story, story_mask, question, answer, forward_only):
		input_feed = {}
		for l in range(len(story)):
			input_feed[self.story[l].name] = story[l]
		for l in range(len(question)):
			input_feed[self.question[l].name] = question[l]
		for l in range(len(answer)):
			input_feed[self.answer[l].name] = answer[l]
		input_feed[self.story_mask.name] = story_mask

		# if not forward_only:
		# 	output_feed = [self.updates[bucket_id],	# Update Op that does SGD.
		# 								 self.gradient_norms[bucket_id],	# Gradient norm.
		# 								 self.losses[bucket_id]]	# Loss for this batch.
		# else:
		# 	output_feed = [self.losses[bucket_id]]	# Loss for this batch.


	def get_qns(self, data_set):
		"""Provide data set; return question and story"""
