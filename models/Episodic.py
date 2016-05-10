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

		# construct memory cell
		#single_cell = tf.nn.rnn_cell.GRUCell(self.mem_size)
		single_cell = tf.nn.rnn_cell.BasicLSTMCell(self.mem_size)
		if mem_depth > 1:
			mem_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * mem_depth)

		# construct episodic_cell
		ep_cell = []
		for i in xrange(self.memory_hops):
			single_cell = cell.MGRUCell(self.ep_size)
			if ep_depth > 1:
				ep_cell.append(tf.nn.rnn_cell.MultiRNNCell([single_cell] * ep_depth))


		for hops in xrange(self.memory_hops):
			output, state = cell.rnn(ep_cell[hops], _inputs, initial_state=self._ep_initial_state)


		# define optimization


	def step():

