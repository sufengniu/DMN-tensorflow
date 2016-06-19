from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange 
import tensorflow as tf

#from tensorflow.models.rnn import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
import models.seq2seq as seq2seq
import models.cell as cell

class DMN(object):
	"""
		this is shrinked test version
		Dynamic Memory Network: it contains four modules: Input, Question, Anwser, Episodic Memory
		check ref: Ask Me Anything: Dynamic Memory Networks for Natural Language Processing
		Args:
			vocab_size: vocabular size
			batch_size: batch size
			learning_rate: learning rate
			embedding_size: embedding size, also the RNN first layer size
			q_depth: question module layer depth
			a_depth: answer module layer depth
			m_depth: memory cell depth
			i_depth: input layer depth (not include input fusion layer)
			memory_hops: how many hops for episodic memory module

		Returns:
			built model of dynamic memory network

	"""
	def __init__(self, vocab_size, embedding_size, batch_size, learning_rate, 
		learning_rate_decay_op, memory_hops, dropout_rate, 
		q_depth, a_depth, episodic_m_depth, ep_depth,
		attention_ff_l1_size, max_gradient_norm, maximum_story_length=5,
		maximum_question_length=20, use_lstm=False, forward_only=False):
	
		# initialization
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.batch_size = batch_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.learning_rate_decay_op = tf.Variable(float(learning_rate_decay_op), trainable=False)
		self.dropout_rate = dropout_rate
		self.global_step = tf.Variable(0, trainable=False, name='global_step')
		self.q_depth = q_depth	# question RNN depth
		self.a_depth = a_depth	# answer RNN depth
		self.m_depth = episodic_m_depth # memory cell depth
		self.ep_depth = ep_depth	# episodic depth
		self.max_gradient_norm = max_gradient_norm
		self.memory_hops = memory_hops	# number of episodic memory pass
		self.m_input_size = embedding_size * 3
		self.m_size = embedding_size # memory cell size	
		self.attention_ff_l1_size = attention_ff_l1_size 
		self.maximum_story_length = maximum_story_length
				
		
		print("[*] Creating Dynamic Memory Network ...")
		# Initializing word2vec
		W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]),
								trainable=False, name="W")
		self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
		self.embedding_init = W.assign(self.embedding_placeholder)

		# Sentence token placeholder
		self.story = []
		story_embedded = []
		for i in range(maximum_story_length):
			self.story.append(tf.placeholder(tf.int32, shape=[None,None], name="Story"))
			story_embedded.append(tf.nn.embedding_lookup(W, self.story[i]))
			story_embedded[i] = tf.transpose(story_embedded[i],[1,0,2])

		self.story_len = tf.placeholder(tf.int32, shape=[1], name="Story_length")

		self.question = tf.placeholder(tf.int32, shape=[None,None], name="Question")	
		question_embedded = tf.transpose(tf.nn.embedding_lookup(W, self.question), [1,0,2])
		self.answer = tf.placeholder(tf.int64, name="answer")

		# configuration of attention gate		
		answer_weights = tf.Variable(tf.truncated_normal([self.m_size, self.vocab_size], -0.1, 0.1), name="answer_weights")	
		answer_biases = tf.Variable(tf.zeros([self.vocab_size]), name="answer_biases")	

		#------------ question module ------------
		with tf.name_scope("question_embedding_rnn"):
			question_embedding_cell = tf.nn.rnn_cell.GRUCell(self.embedding_size)
			_, self.question_state = tf.nn.dynamic_rnn(question_embedding_cell, question_embedded, dtype=tf.float32, time_major=True)
		
		#------------ Input module ------------
		Story_embedding_cell = tf.nn.rnn_cell.GRUCell(self.embedding_size)
		self.story_state_array = []
		with tf.name_scope("story_embedding_rnn"):
			for i in range(maximum_story_length):
				with tf.variable_scope("embedding_rnn", reuse=True if i > 0 else None):
					_, story_states = tf.nn.dynamic_rnn(Story_embedding_cell, story_embedded[i], dtype=tf.float32, time_major=True)
					self.story_state_array.append(story_states)
		fusion_fw_cell = tf.nn.rnn_cell.GRUCell(self.embedding_size)
		fusion_bw_cell = tf.nn.rnn_cell.GRUCell(self.embedding_size)
		(self.facts, _, _) = rnn.bidirectional_rnn(fusion_fw_cell,fusion_bw_cell, self.story_state_array, dtype=tf.float32)		
		
		#------------ episodic memory module ------------	
		self.ep_size = self.embedding_size * 4 # episodic cell size
		
		z_dim = self.embedding_size * 8
		attention_ff_size = z_dim
		attention_ff_l2_size = 1 
		q_double = tf.concat(1, [self.question_state, self.question_state])

		# -------- multi-layer feedforward for multi-hop propagation -----------	
		self.facts = tf.concat(0, self.facts_)
		# ep_cell = cell.MGRUCell(self.ep_size)
		# mem_cell = cell.MemCell(self.m_size)	
		mem_weights = tf.get_variable("mem_weights", [attention_ff_size, self.m_size], initializer=tf.random_normal_initializer())
		mem_biases = tf.get_variable("mem_biases", [self.m_size], initializer=tf.random_normal_initializer())
		l1_weights = tf.get_variable("l1_weights", [attention_ff_size, attention_ff_l1_size], 
			initializer=tf.random_normal_initializer())
		l1_biases = tf.get_variable("l1_biases", [attention_ff_l1_size], 
			initializer=tf.random_normal_initializer())
		l2_weights = tf.get_variable("l2_weights", [attention_ff_l1_size, attention_ff_l2_size], 
			initializer=tf.random_normal_initializer())
		l2_biases = tf.get_variable("l2_biases", [attention_ff_l2_size], 
			initializer=tf.random_normal_initializer())
		mgru_weights = {}
		mgru_weights['ur_weights'] = tf.get_variable('ur_weights', [embedding_size, embedding_size], initializer=tf.random_normal_initializer())
		mgru_weights['wr_weights'] = tf.get_variable('wr_weights', [embedding_size, embedding_size], initializer=tf.random_normal_initializer())
		mgru_weights['uh_weights'] = tf.get_variable('uh_weights', [embedding_size, embedding_size], initializer=tf.random_normal_initializer())
		mgru_weights['wh_weights'] = tf.get_variable('wh_weights', [embedding_size, embedding_size], initializer=tf.random_normal_initializer())
		
		def MGRU(inputs, episodic_gates):
			"""	modified GRU 
				arg:

			"""
			batch_size = array_ops.shape(inputs[0])[0]
			state = tf.zeros([1,embedding_size],tf.float32)
			for time, (input_, episodic_gate_) in enumerate(zip(inputs, episodic_gates)):
				input_ = tf.reshape(input_,[1,embedding_size])
				r = tf.sigmoid(tf.matmul(input_, mgru_weights['ur_weights']) + tf.matmul(state, mgru_weights['wr_weights']))
				c = tf.tanh(tf.matmul(input_, mgru_weights['uh_weights']) + tf.matmul(tf.mul(state, r), mgru_weights['wh_weights']))
				state = tf.mul(episodic_gate_, c) + tf.mul((1 - episodic_gate_), state)
			return state
		episodic_gate_unpacked = []
		def body(mem_state_previous, hops):
					
			# attention GRU	
			# outputs, context = cell.rnn_ep(ep_cell, self.facts_, episodic_gate_unpacked, dtype=tf.float32)
			# outputs, context = ep_cell(ep_cell, self.facts_, episodic_gate_unpacked)
			context = MGRU(self.facts_, episodic_gate_unpacked)

			# memory updates
			# mem_state_current = mem_cell(mem_state_previous, self.question_state, mem_state_previous, mem_weights, mem_biases, hops)
			#question_state_next = question_state_prev
			#print (self.question_state, mem_state_previous)
			mem_state_current = tf.nn.relu(tf.matmul(tf.concat(1, [mem_state_previous, context, self.question_state]), mem_weights) + mem_biases)
			
			hops = tf.add(hops,1)
			return  mem_state_current, hops

		def condition(mem_state_previous, hops):
			mem_state_double = tf.concat(1, [mem_state_previous, mem_state_previous])
			z = tf.concat(1, [tf.mul(self.facts, q_double), tf.mul(self.facts, mem_state_double), 
				tf.abs(tf.sub(self.facts, q_double)), tf.abs(tf.sub(self.facts, mem_state_double))], name="z")
			episodic_array_reshaped = tf.reshape(tf.matmul(tf.tanh(tf.matmul(z , l1_weights) + l1_biases) , l2_weights) + l2_biases, [1,-1], name="episodic_array_reshaped")
			episodic_gate = tf.nn.softmax(episodic_array_reshaped)
			episodic_gate_unpacked = tf.unpack( tf.reshape(episodic_gate, [maximum_story_length,1]))
			argmax_ep_gate = tf.to_int32(tf.argmax(episodic_gate, 1)) #should be 1
			return tf.cond(tf.equal(hops,0),lambda: tf.equal(hops,0),
				lambda: tf.logical_and(tf.less(argmax_ep_gate,self.story_len)[0],tf.less(hops,tf.constant(self.memory_hops))))
			# return tf.logical_and(tf.less(argmax_ep_gate,self.story_len)[0],tf.less(hops,tf.constant(self.memory_hops)))

		initial_argmax_ep_gate = tf.constant(0)
		initial_hops = tf.constant(0)
			# initial_context = tf.constant([[0.5 for _ in range(50)]])
		mem_state, _ = tf.while_loop(condition,body,[self.question_state, initial_hops], back_prop=True)


		self.a_state = mem_state
				
		self.predicted_answer = tf.matmul(self.a_state, answer_weights)
		
		answer = tf.reshape(tf.one_hot(self.answer, self.vocab_size, 1.0, 0.0), [1,self.vocab_size])
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predicted_answer, answer))
		# self.loss = tf.nn.softmax_cross_entropy_with_logits(self.predicted_answer, answer)
		params = tf.trainable_variables()	

		if not forward_only:
			self.gradient_norms = []
			self.updates = []
			# optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
			optimizer = tf.train.AdamOptimizer(self.learning_rate)
			gradients = tf.gradients(self.loss, params)
	
			clipped_gradients, norm = tf.clip_by_global_norm(gradients,
				self.max_gradient_norm)
			self.gradient_norms = norm
			self.updates = optimizer.apply_gradients(
				zip(clipped_gradients, params), global_step=self.global_step)
		
		self.saver = tf.train.Saver(tf.all_variables())
	
	def step(self, session, story, story_mask, question, answer, forward_only):
		input_feed = {}
		# split story according to story_mask and pad it to maximum story length
		story_splited = [[] for i in story_mask]
		story_count = 0
		for i in range(len(story)):
			story_splited[story_count].append(story[i])
			if i == story_mask[story_count]:
				story_count += 1
		for i in range(len(story_mask),self.maximum_story_length):
			story_splited.append([0,0,0,0,0])
		for l in range(self.maximum_story_length):
			input_feed[self.story[l].name] = [story_splited[l]]

		input_feed[self.question.name] = [question]
		input_feed[self.answer.name] = answer	
		print ('correct answer: ',answer)
		input_feed[self.story_len.name] = [len(story_mask)]
		
		if not forward_only:
			output_feed = [self.gradient_norms, self.loss, self.updates]
		else:
			output_feed = [self.loss,		# Loss for this batch.
							tf.argmax(self.logits, 0),
							# debugging
							self.logits,
							self.question_state,
							self.facts,
							self.episodic_gate]

		outputs = session.run(output_feed, input_feed)
		if not forward_only:	
			return outputs#[1], outputs[2], None	# Gradient norm, loss, no outputs.
		else:
			return None, outputs[0], outputs[1]	# No gradient norm, loss, outputs.

	def init_embedding(self, sess, embedding):
		sess.run(self.embedding_init, feed_dict={self.embedding_placeholder.name: embedding})

