from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import os
import random
import sys
import time
import re

import numpy as np
import tensorflow as tf
from tensorflow.contrib import skflow

import data_utils
from models.DMN import DMN
import models.cell
import models.seq2seq


flags=tf.app.flags

# model configuration
flags.DEFINE_integer("vocab_size", 80000, "vocabulary size.")
flags.DEFINE_integer("embedding_size", 50, "Size of each model layer.")
flags.DEFINE_integer("q_depth", 1, "question module depth")
flags.DEFINE_integer("a_depth", 1, "answer module depth")
flags.DEFINE_integer("episodic_m_depth", 1, "memory update module depth")
flags.DEFINE_integer("ep_depth", 1, "episodic module depth")
flags.DEFINE_integer("attention_ff_l1_size", 50, "episodic gating neural network first layer size") # testing should be 100 originally
flags.DEFINE_integer("maximum_story_length", 15, "max story length")
flags.DEFINE_integer("maximum_attention_length", 15, "max attetion length")
flags.DEFINE_integer("maximum_question_length", 20, "max question length")
flags.DEFINE_integer("memory_hops", 5, "max memoy hops")  # testing should be 10

flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
flags.DEFINE_float("learning_rate_decay_op", 0.99, "Learning rate decay.")
flags.DEFINE_float("dropout_rate", 0.5, "dropout rates")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
# flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
# flags.DEFINE_integer("max_len", 100, "sequence length longer than this will be ignored.")
# flags.DEFINE_integer("depth", 1, "Number of layers in the model.")
flags.DEFINE_string("word2vec_dir", "./glove.6B.50d.txt", "word2vec location")
flags.DEFINE_string("data_dir", "bAbI_data/en", "Data directory")
flags.DEFINE_string("train_dir", "bAbI_data/en", "Training directory.")
flags.DEFINE_boolean("use_lstm", False, "Set True using LSTM, or False using GRU")
# flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
flags.DEFINE_integer("steps_per_checkpoint", 50, "How many training steps to do per checkpoint.")
# flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
flags.DEFINE_string("data_type", "1", "choose babi_map, check data_utils for detail")

FLAGS = flags.FLAGS



def create_model(session, vocab_size, forward_only):

	model = DMN(vocab_size, 
		FLAGS.embedding_size, 
		FLAGS.learning_rate, 
		FLAGS.learning_rate_decay_op, 
		FLAGS.memory_hops,
		FLAGS.dropout_rate,
		FLAGS.q_depth,
		FLAGS.a_depth,
		FLAGS.episodic_m_depth,
		FLAGS.ep_depth,
		FLAGS.attention_ff_l1_size,
		FLAGS.max_gradient_norm,
		FLAGS.maximum_story_length,
		FLAGS.maximum_question_length,
		# FLAGS.maximum_attention_length,
		use_lstm=FLAGS.use_lstm,
		forward_only=False)
	ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
	if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
		print("[*] Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("[*] Created model with fresh parameters.")
		session.run(tf.initialize_all_variables())
	return model

def train():
	if FLAGS.data_type == "1":
		print("[*] Preparing bAbI data ... in %s" % FLAGS.data_dir)
		babi_train_raw, babi_validation_raw = data_utils.get_babi_raw("1")
		vocab, ivocab, embedding, embedding_size = data_utils.load_glove_w2v(FLAGS.word2vec_dir)
		if embedding_size != FLAGS.embedding_size:
			raise Exception ("Embedding size of model and word2vec must match.")
		t_context, t_questions, t_answers, t_fact_counts, t_input_masks = data_utils.process_input(babi_train_raw, vocab, ivocab)
		v_context, v_questions, v_answers, v_fact_counts, v_input_masks = data_utils.process_input(babi_validation_raw, vocab, ivocab)
	else:
		raise Exception ("Only joint mode is allowed")
	with tf.Graph().as_default():
		with tf.Session() as sess:
			model = create_model(sess, len(vocab), False)
			model.init_embedding(sess, embedding)
			step_time, loss = 0.0, 0.0
			current_step = 0
			previous_losses = []
			for j in range(50):
				print ("step %i" % j)
				for i in range(999):
					# _, loss, _ = 
					# loss = 
					gindex,gvalue,gargmax,aweight,p_answer, summaries, updates, gradient_norms, loss = model.step(sess, t_context[i], t_input_masks[i], t_questions[i], t_answers[i], False)
					print ("=========step %d==========" % i)
					print ('answer weight',aweight)
					print ('gate index',gindex)
					print ('gate array', gvalue)
					print ('gate argmax', gargmax)
					print ("loss is: %.2f, gradient norm: %.2f" % (loss, gradient_norms))
					print ("predicted answer: %d" % p_answer)
					#print ("---------- a_state tensor -----------")
					#print (a_state)
				# 	for i in loss:
				# 		print (i.shape)
					# break
					if i %20 == 0:
						if i== 0:
							writer = tf.train.SummaryWriter("./tensorboard", sess.graph_def)
						else:
							writer.add_summary(summaries, i)
				break
					#print (loss[2].shape)
					# print (loss[4].shape)
					# print (loss[5].shape)
					# print (loss[6].shape)
					# print (loss[7].shape)

def test():
	"""Test the DMN model."""
	with tf.Session() as sess:
		print("Self-test for neural translation model.")
		# Create model with vocab=10, embedding=10
		model = DMN.DMN(10,10,0.5,0.99,0.5)
		sess.run(tf.initialize_all_variables())

		# Fake data set
		#data_set = 




def main(_):
	train()# if FLAGS.train:
	



if __name__ == '__main__':
	tf.app.run()




