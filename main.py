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
flags.DEFINE_integer("embedding_size", 300, "Size of each model layer.")
flags.DEFINE_integer("q_depth", 1, "question module depth")
flags.DEFINE_integer("a_depth", 1, "answer module depth")
flags.DEFINE_integer("episodic_m_depth", 1, "memory update module depth")
flags.DEFINE_integer("ep_depth", 1, "episodic module depth")
flags.DEFINE_integer("m_input_size", 300, "context vector size in episodic module")
flags.DEFINE_integer("maximum_story_length", 50, "max story length")
flags.DEFINE_integer("maximum_question_length", 20, "max question length")
flags.DEFINE_integer("memory_hops", 10, "max memoy hops")

flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("learning_rate_decay_op", 0.99, "Learning rate decay.")
flags.DEFINE_float("dropout_rate", 0.5, "dropout rates")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
# flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
# flags.DEFINE_integer("max_len", 100, "sequence length longer than this will be ignored.")
# flags.DEFINE_integer("depth", 1, "Number of layers in the model.")
flags.DEFINE_string("data_dir", "bAbI_data/en", "Data directory")
flags.DEFINE_string("train_dir", "bAbI_data/en", "Training directory.")
flags.DEFINE_boolean("use_lstm", False, "Set True using LSTM, or False using GRU")
# flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
flags.DEFINE_integer("steps_per_checkpoint", 50, "How many training steps to do per checkpoint.")
# flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
flags.DEFINE_string("data_type", "1", "choose babi_map, check data_utils for detail")
FLAGS = flags.FLAGS



def create_model(session, forward_only):

	model = DMN(FLAGS.vocab_size, 
		FLAGS.embedding_size, 
		FLAGS.learning_rate, 
		FLAGS.learning_rate_decay_op, 
		FLAGS.memory_hops,
		FLAGS.dropout_rate,
		FLAGS.q_depth,
		FLAGS.a_depth,
		FLAGS.episodic_m_depth,
		FLAGS.ep_depth,
		FLAGS.m_input_size,
		FLAGS.max_gradient_norm,
		FLAGS.maximum_story_length,
		FLAGS.maximum_question_length,
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
		t_context, t_questions, t_answers, t_fact_counts, t_input_masks, vocab, ivocab = data_utils.process_input(babi_train_raw)
		v_context, v_questions, v_answers, v_fact_counts, v_input_masks, vocab, ivocab = data_utils.process_input(babi_validation_raw, vocab, ivocab)
	else:
		raise Exception ("Only joint mode is allowed")
	
	with tf.Session() as sess:
		model = create_model(sess, False)
		step_time, loss = 0.0, 0.0
		current_step = 0
		previous_losses = []
		for i in range(500):
			_, loss, _model.step(sess, t_context[i], t_input_masks[i], t_questions[i], t_answers[i], False)
			print (loss)	


def test():
	"""Test the DMN model."""
	with tf.session() as sess:
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




