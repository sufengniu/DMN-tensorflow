"""
	modified from seq2seq.py in tensorflow
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

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



def sentence_embedding_rnn(encoder_inputs, length, vocab_size, cell, 
    embedding_size, sentence_length, init_state=None,dtype=dtypes.float32, scope=None):
	"""

	"""
	with variable_scope.variable_scope(scope or "embedding_rnn"):
		encoder_cell = rnn_cell.EmbeddingWrapper(
				cell, embedding_class=vocab_size,
				embedding_size=embedding_size)
        for step in range(length):
		        _, encoder_state = rnn.rnn(encoder_cell, encoder_input[step], encoder_state, dtype=dtype)
			
	return encoder_state


