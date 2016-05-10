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
	def __init__(self, vocab_size, embedding_size, learning_rate, learning_rate_decay_op,
	dropout_rate, maximum_length=20, maximum_sentence=10, use_lstm=False, forward_only=False):
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.learning_rate_decay_op = tf.Variable(float(learning_rate_decay_op), trainable=False)
		self.dropout_rate = dropout_rate
		self.global_step = tf.Variable(0, trainable=False, name='global_step')

		print("[*] Creating Dynamic Memory Network ...")
		# Input module
		reader_cell = tf.nn.rnn_cell.GRUCell(self.embedding_size)

		if use_lstm:
			reader_cell = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)
		if not forward_only and dropout < 1:
			reader_cell = tf.nn.rnn_cell.DropoutWrapper(
				reader_cell, output_keep_prob=dropout_rate)
		#embed toekn into vector, feed into rnn cell return cell state


		def seq2seq_f(encoder_inputs):
			return seq2seq.sentence_embedding_rnn(
				encoder_inputs, maximum_length, vocab_size, reader_cell, embedding_size)
		
		# Sentence token placeholder
		self.story = tf.placeholder(tf.int32, [maximum_sentence, maximum_length])

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

		


	def step(self, session, forward_only):


	def get_qns(self, data_set):
		"""Provide data set; return question and story"""
		

		
