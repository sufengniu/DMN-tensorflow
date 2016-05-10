
""" modified GRU for episodic memory based on tensorflow source code

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from six.moves import xrange

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs


from tensorflow.python.platform import tf_logging as logging



class MGRUCell(RNNCell):
	"""Modified Gated Recurrent Unit cell"""

	def __init__(self, num_units, input_size=None):
		if input_size is not None:
			logging.warn("%s: The input_size parameter is deprecated." % self)
		self._num_units = num_units

	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units

	def __call__(self, inputs, state, episodic_gate, scope=None):
		"""Gated recurrent unit (GRU) with nunits cells."""
		with vs.variable_scope(scope or type(self).__name__):	# "GRUCell"
			with vs.variable_scope("Gates"):	# Reset gate and update gate.
				# We start with bias of 1.0 to not reset and not update.
				r = rnn_cell.linear([inputs, state], self._num_units, True, 1.0))
				r = sigmoid(r)
			with vs.variable_scope("Candidate"):
				c = tanh(rnn_cell.linear([inputs, r * state], self._num_units, True))
			new_h = episodic_gate * state + (1 - episodic_gate) * c
		return new_h, new_h

def rnn(cell, inputs, initial_state=None, episodic_gate, dtype=None,
		sequence_length=None, scope=None):

	if not isinstance(cell, rnn_cell.RNNCell):
		raise TypeError("cell must be an instance of RNNCell")
	if not isinstance(inputs, list):
		raise TypeError("inputs must be a list")
	if not inputs:
		raise ValueError("inputs must not be empty")

	outputs = []
	# Create a new scope in which the caching device is either
	# determined by the parent scope, or is set to place the cached
	# Variable using the same placement as for the rest of the RNN.
	with vs.variable_scope(scope or "RNN") as varscope:
		if varscope.caching_device is None:
			varscope.set_caching_device(lambda op: op.device)

		# Temporarily avoid EmbeddingWrapper and seq2seq badness
		# TODO(lukaszkaiser): remove EmbeddingWrapper
		if inputs[0].get_shape().ndims != 1:
			(fixed_batch_size, input_size) = inputs[0].get_shape().with_rank(2)
			if input_size.value is None:
				raise ValueError(
						"Input size (second dimension of inputs[0]) must be accessible via "
						"shape inference, but saw value None.")
		else:
			fixed_batch_size = inputs[0].get_shape().with_rank_at_least(1)[0]

		if fixed_batch_size.value:
			batch_size = fixed_batch_size.value
		else:
			batch_size = array_ops.shape(inputs[0])[0]
		if initial_state is not None:
			state = initial_state
		else:
			if not dtype:
				raise ValueError("If no initial_state is provided, dtype must be.")
			state = cell.zero_state(batch_size, dtype)

		if sequence_length is not None:	# Prepare variables
			sequence_length = math_ops.to_int32(sequence_length)
			zero_output = array_ops.zeros(
					array_ops.pack([batch_size, cell.output_size]), inputs[0].dtype)
			zero_output.set_shape(
					tensor_shape.TensorShape([fixed_batch_size.value, cell.output_size]))
			min_sequence_length = math_ops.reduce_min(sequence_length)
			max_sequence_length = math_ops.reduce_max(sequence_length)

		for time, input_ in enumerate(inputs):
			if time > 0: vs.get_variable_scope().reuse_variables()
			# pylint: disable=cell-var-from-loop
			call_cell = lambda: cell(input_, state, episodic_gate)
			# pylint: enable=cell-var-from-loop
			if sequence_length is not None:
				(output, state) = rnn._rnn_step(
						time, sequence_length, min_sequence_length, max_sequence_length,
						zero_output, state, call_cell)
			else:
				(output, state) = call_cell()

			outputs.append(output)

		return (outputs, state)
