from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange 
import tensorflow as tf

import cell

class Episodic(object):
	"""
		Episodic Memory module:
		Args:

		Return:


	"""
	def episodic_init(self):

		# initialization
		self.memory_hops = ???	# number of episodic memory pass
		self.facts =???	# inputs from Input modeule
		self.num_steps = num_sentences

		self.mem_size # memory cell size
		self.mem_depth
		self.ep_size # episodic cell size
		self.ep_depth
		self._ep_initial_state = cell.zero_state(batch_size, tf.float32)

		# configuration of attention gate
		input_size = 
		l1_size = 
		l2_size = 
		mem_size = # memory cell size
		mem_input_size = 
		with tf.variable_scope("episodic"):
			# parameters of attention gate
			l1_weights = tf.Variable(tf.truncated_normal([input_size, l1_size], -0.1, 0.1))
			l1_biases = tf.Variable(tf.zeros([l1_size]))
			l2_weights = tf.Variable(tf.truncated_normal([l1_size, l2_size], -0.1, 0.1))
			l2_biases = tf.Variable(tf.zeros([l2_size]))
			# paramters of episodic
			mem_weights = tf.Variable(tf.truncated_normal([mem_input_size, mem_size], -0.1, 0.1))
			mem_biases = tf.Variable(tf.zeros([mem_size]))

		def feedfoward_nn(l1_input):
			l2_input = tf.tanh(tf.matmul(l1_input * l1_weights) + l1_biases)
			logits = tf.matmul(l2_input * l2_weights) + l2_biases
			gate_prediction = tf.nn.softmax(logits)
			return gate_prediction


		# construct memory cell
		#single_cell = tf.nn.rnn_cell.BasicLSTMCell(self.mem_size)
		#single_cell = cell.MemCell(self.mem_size)
		single_cell = tf.nn.rnn_cell.GRUCell(self.mem_size)
		if mem_depth > 1:
			mem_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * mem_depth)

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

	
		# define optimization


	def step():
		
