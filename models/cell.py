
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



class MGRUCell(rnn_cell.RNNCell):
	"""Modified Gated Recurrent Unit cell"""

	def __init__(self, num_units):
		self._num_units = num_units

	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units

	def __call__(self, inputs, state, episodic_gate, scope=None):
		"""Gated recurrent unit (GRU) with nunits cells."""
		
		with vs.variable_scope("MGRUCell"):  # "GRUCell"
			with vs.variable_scope("Gates"):	# Reset gate and update gate.
				# We start with bias of 1.0 to not reset and not update.
				r = rnn_cell.linear([inputs, state], self._num_units, True, 1.0, scope=scope)
				r = sigmoid(r)
			with vs.variable_scope("Candidate"):
				c = tanh(rnn_cell.linear([inputs, r * state], self._num_units, True))
			for f in xrange(len(episodic_gate)):
				new_h = episodic_gate[f] * c + (1 - episodic_gate[f]) * state
		return new_h, new_h

class MemCell(rnn_cell.RNNCell):
	""" simplified recurrent cell for memory"""

	def __init__(self, num_units):
		self._num_units = num_units

	@property
	def state_size(self):
		return self._num_units

	def __call__(self, inputs, question, state, m_input_size, m_size, scope=None):
		""" simple Recurrent cell for memory updates """

		with tf.variable_scope("episodic"):
			mem_weights = tf.get_variable("mem_weights", [m_input_size, self._num_units])
			mem_bias = tf.get_variable("mem_biases", [self._num_units])
		new_state = tf.nn.relu(tf.matmul(tf.concat(1, [state, inputs, question]), mem_weights) + mem_bias)
		return new_state


class MultiMemCell(rnn_cell.RNNCell):
	"""RNN cell composed sequentially of multiple simple cells."""

	def __init__(self, cells):
		"""Create a RNN cell composed sequentially of a number of RNNCells.
		Args:
			cells: list of RNNCells that will be composed in this order.
		Raises:
			ValueError: if cells is empty (not allowed).
		"""
		if not cells:
			raise ValueError("Must specify at least one cell for MultiRNNCell.")
		self._cells = cells

	@property
	def state_size(self):
		return sum([cell.state_size for cell in self._cells])

	@property
	def output_size(self):
		return self._cells[-1].output_size

	def __call__(self, inputs, question, state, m_input_size, m_size, scope=None):
		"""Run this multi-layer cell on inputs, starting from state."""
		with vs.variable_scope(scope or type(self).__name__):	# "MultiRNNCell"
			cur_state_pos = 0
			cur_inp = inputs
			new_states = []
			for i, cell in enumerate(self._cells):
				with vs.variable_scope("Cell%d" % i):
					cur_state = array_ops.slice(
							state, [0, cur_state_pos], [-1, cell.state_size])
					cur_state_pos += cell.state_size
					cur_inp, new_state = cell(cur_inp, question, cur_state, m_input_size, m_size)
					new_states.append(new_state)
		return cur_inp, array_ops.concat(1, new_states)


class MultiMGRUCell(rnn_cell.RNNCell):
	"""RNN cell composed sequentially of multiple simple cells."""

	def __init__(self, cells):
		"""Create a RNN cell composed sequentially of a number of RNNCells.
		Args:
			cells: list of RNNCells that will be composed in this order.
		Raises:
			ValueError: if cells is empty (not allowed).
		"""
		if not cells:
			raise ValueError("Must specify at least one cell for MultiRNNCell.")
		self._cells = cells

	@property
	def state_size(self):
		return sum([cell.state_size for cell in self._cells])

	@property
	def output_size(self):
		return self._cells[-1].output_size

	def __call__(self, inputs, state, episodic_gate, scope=None):
		"""Run this multi-layer cell on inputs, starting from state."""
		with vs.variable_scope(scope or type(self).__name__):	# "MultiRNNCell"
			cur_state_pos = 0
			cur_inp = inputs
			new_states = []
			for i, cell in enumerate(self._cells):
				with vs.variable_scope("Cell%d" % i):
					cur_state = array_ops.slice(
							state, [0, cur_state_pos], [-1, cell.state_size])
					cur_state_pos += cell.state_size
					cur_inp, new_state = cell(cur_inp, cur_state, episodic_gate)
					new_states.append(new_state)
		return cur_inp, array_ops.concat(1, new_states)

def rnn_ep(cell, inputs, episodic_gate, initial_state=None, dtype=None,
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


def rnn_mem(cell, inputs, question, state, m_input_size, m_size, initial_state=None, dtype=None,
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
			call_cell = lambda: cell(input_, question, state, m_input_size, m_size)
			# pylint: enable=cell-var-from-loop
			if sequence_length is not None:
				(output, state) = rnn._rnn_step(
						time, sequence_length, min_sequence_length, max_sequence_length,
						zero_output, state, call_cell)
			else:
				(output, state) = call_cell()

			outputs.append(output)

		return (outputs, state)

