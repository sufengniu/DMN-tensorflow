from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange 
import tensorflow as tf

class Question(object):
	def __init__(self, config):
		self.vocab_size = config.vocab_size
		self.batch_size = batch_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.global_step = tf.Variable(0, trainable=False, name='global_step')

		self._input_data = tf.placeholder(tf.int32, ][)

	def step(self, session, config):

