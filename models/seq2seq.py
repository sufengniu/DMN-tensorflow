"""
	modified from seq2seq.py in tensorflow
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip	 # pylint: disable=redefined-builtin

import tensorflow as tf
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



def sentence_embedding_rnn(_encoder_inputs, vocab_size, cell, 
	embedding_size, mask=None, dtype=dtypes.float32, scope=None, reuse_scop=None):
	"""
	
	"""
	with variable_scope.variable_scope("embedding_rnn", reuse=reuse_scop):
		# encoder_cell = rnn_cell.EmbeddingWrapper(
		# 		cell, embedding_classes=vocab_size,
		# 		embedding_size=embedding_size)
		# Divde encoder_inputs by given input_mask
		if mask != None:
			encoder_inputs = [[] for _ in mask]
			_mask = 0
			for num in range(len(_encoder_inputs)):
				encoder_inputs[_mask].append(_encoder_inputs[num])
				if num == mask[_mask]:
					_mask += 1
		else:
			encoder_inputs = []
			encoder_inputs.append(_encoder_inputs)
		encoder_state = None	 
		encoder_states = []
		for encoder_input in encoder_inputs:
			if encoder_state == []:
				_, encoder_state = rnn.dynamic_rnn(encoder_cell, encoder_input, dtype=dtype)
			else:
				_, encoder_state = rnn.dynamic_rnn(encoder_cell, encoder_input, encoder_state, dtype=dtype)
			encoder_states.append(encoder_state)
		return encoder_states



# def def_feedforward_nn(input_size, l1_size, l2_size):
# 	with tf.variable_scope("episodic"):
# 		l1_weights = tf.get_variable("l1_weights", [input_size, l1_size])
# 		l1_biases = tf.get_variable("l1_biases", [l1_size])
# 		l2_weights = tf.get_variable("l2_weights", [l1_size, l2_size])
# 		l2_biases = tf.get_variable("l2_biases", [l2_size])
#def feedforward_nn(l1_input, input_size, l1_size, l2_size):
#	with tf.variable_scope("episodic"):
#		l1_weights = tf.get_variable("l1_weights", [input_size, l1_size])
#		l1_biases = tf.get_variable("l1_biases", [l1_size])
#		l2_weights = tf.get_variable("l2_weights", [l1_size, l2_size])
#		l2_biases = tf.get_variable("l2_biases", [l2_size])
#		l2_input = tf.tanh(tf.matmul(l1_input , l1_weights) + l1_biases)
#		gate_prediction = tf.matmul(l2_input , l2_weights) + l2_biases
#		return gate_prediction

