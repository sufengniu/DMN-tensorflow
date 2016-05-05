from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange 
import tensorflow as tf

class Input(object):
	"""
<<<<<<< HEAD
	This class reads story sentence by sentence, generating sentence vecter
	
	"""
	def init(self, config):
		self.vocab_size = config.vocab_size
		self.batch_size = batch_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.global_step = tf.Variable(0, trainable=False, name='global_step')

		self._input_data = tf.placeholder(tf.int32, ][)

	def step(self, session, config):

=======
		Input module: it contains Input fusion layer (bidirectional RNN)
	"""
	#def __init__(self, config):
	def init_Input(self, config):
		self.vocab_size = config.vocab_size
		self.batch_size = batch_size
		self.size = 
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.global_step = tf.Variable(0, trainable=False, name='global_step')


		with tf.variable_scope("embed", reuse=False):
			self.embedding = tf.get_variable("embedding", [config.vocab_size, self.size], 
				initializer=tf.random_normal_initializer(-1.0, 1.0))


	def step(self, session, config):




	def unsupervised_step(self, session, config):
		pass

	def supervised_step(self, session, config):
		pass
>>>>>>> f30f3d3ed568bd2376c88897dad98c029fa675c7
