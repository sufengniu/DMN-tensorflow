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

class DMN(Question, Input, Episodic, Answer):
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
	def __init__(self, config, forward_only=False):
		self.vocab_size = config.vocab_size
		self.batch_size = config.batch_size
		self.learning_rate = tf.Variable(float(config.learning_rate), trainable=False)
		self.global_step = tf.Variable(0, trainable=False, name='global_step')

		self._input_data = tf.placeholder(tf.int32, [])

		# Question module
		try:
			self.init_question(forward_only)
		except AttributeError:
			raise NotImplementedError("question module init error!")

		# Episodic module
		try:
			self.init_episodic(forward_only)
		except AttributeError:
			raise NotImplementedError("episodic module init error!")

		# Input module
		try:
			self.init_input(forward_only)
		except AttributeError:
			raise NotImplementedError("input module init error!")

		# Answer module
		try:
			self.init_answer(forward_only)
		except AttributeError:
			raise NotImplementedError("answer module init error!")

		


	def step(self, session, forward_only):
		
