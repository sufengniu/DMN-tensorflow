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
flags.DEFINE_integer("attention_ff_l1_size", 10, "episodic gating neural network first layer size") # testing should be 100 originally
flags.DEFINE_integer("maximum_story_length", 10, "max story length")
flags.DEFINE_integer("maximum_attention_length", 15, "max attetion length")
flags.DEFINE_integer("maximum_question_length", 20, "max question length")
flags.DEFINE_integer("memory_hops", 5, "max memoy hops")  # testing should be 10
flags.DEFINE_integer("epoch", 5, "number of repeating training")
flags.DEFINE_float("learning_rate", 0.2, "Learning rate.")
flags.DEFINE_float("learning_rate_decay_op", 0.99, "Learning rate decay.")
flags.DEFINE_float("dropout_rate", 0.4, "dropout rates")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
flags.DEFINE_integer("batch_size", 8, "Batch size to use during training.")
# flags.DEFINE_integer("max_len", 100, "sequence length longer than this will be ignored.")
# flags.DEFINE_integer("depth", 1, "Number of layers in the model.")
flags.DEFINE_string("word2vec_dir", "./glove.6B.300d.txt", "word2vec location")
flags.DEFINE_string("data_dir", "bAbI_data/en", "Data directory")
flags.DEFINE_string("train_dir", "data", "Training directory.")
flags.DEFINE_boolean("use_lstm", False, "Set True using LSTM, or False using GRU")
# flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
# flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
flags.DEFINE_boolean("validation", False, "Set to True for test set validation.")
flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
flags.DEFINE_string("data_type", "1", "choose babi_map, check data_utils for detail")

FLAGS = flags.FLAGS



def create_model(session, vocab_size, forward_only):

  model = DMN(vocab_size, 
    FLAGS.embedding_size,
    FLAGS.batch_size,
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
  print("[*] Preparing bAbI data ... in %s" % FLAGS.data_dir)
  babi_train_raw, babi_validation_raw = data_utils.get_babi_raw(FLAGS.data_type)
  vocab, ivocab, embedding, embedding_size = data_utils.load_glove_w2v(FLAGS.word2vec_dir, FLAGS.embedding_size)
  if embedding_size != FLAGS.embedding_size:
    raise Exception ("Embedding size of model and word2vec must match.")
  t_context, t_questions, t_answers, t_fact_counts, t_input_masks = data_utils.process_input(babi_train_raw, vocab, ivocab)
  # v_context, v_questions, v_answers, v_fact_counts, v_input_masks = data_utils.process_input(babi_validation_raw, vocab, ivocab)
  with tf.Graph().as_default():
    with tf.Session() as sess:
      model = create_model(sess, len(vocab), False)
      model.init_embedding(sess, embedding)
      step_time, loss = 0.0, 0.0
      current_step = 0
      previous_losses = []
      for j in range(20):
        print ("step %i" % j)
        for i in range(len(t_context)):
          start_time = time.time()
          step_loss, _, _, answer, gate = model.step(sess, t_context[i], t_input_masks[i], t_questions[i], t_answers[i], False)
          print (step_loss, ivocab[int(answer)])
          print ('correct answer:%s' % ivocab[int(t_answers[i])])
          step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
          loss += step_loss / FLAGS.steps_per_checkpoint
          current_step += 1
          # Once in a while, we save checkpoint, print statistics, and run evals.
          if current_step % FLAGS.steps_per_checkpoint == 0:
            # Print statistics for the previous epoch.
            perplexity = math.exp(loss) if loss < 300 else float('inf')
            print ("global step %d learning rate %.4f step-time %.2f perplexity "
                   "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                             step_time, perplexity))
            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
              sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)
            # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join(FLAGS.train_dir, "DMN.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            step_time, loss = 0.0, 0.0
            # Run evals on development set and print their perplexity.
            # for bucket_id in xrange(len(_buckets)):
            #   if len(dev_set[bucket_id]) == 0:
            #     print("  eval: empty bucket %d" % (bucket_id))
            #     continue
            #   encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            #       dev_set, bucket_id)
            #   _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
            #                                target_weights, bucket_id, True)
            #   eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
            #   print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
            # sys.stdout.flush()

        
def validation():
  with tf.Session() as sess:
    print("[*] Preparing bAbI data ... in %s" % FLAGS.data_dir)
    _, babi_validation_raw = data_utils.get_babi_raw(FLAGS.data_type)
    vocab, ivocab, embedding, embedding_size = data_utils.load_glove_w2v(FLAGS.word2vec_dir, FLAGS.embedding_size)
    if embedding_size != FLAGS.embedding_size:
      raise Exception ("Embedding size of model and word2vec must match.")
    v_context, v_questions, v_answers, v_fact_counts, v_input_masks = data_utils.process_input(babi_validation_raw, vocab, ivocab)
    # Create model and load parameters.
    model = create_model(sess, len(vocab), True)
    model.init_embedding(sess, embedding)
    counter = 0
    for i in range(len(v_context)):
      start_time = time.time()
      _, step_loss, answer = model.step(sess, v_context[i], v_input_masks[i], v_questions[i], v_answers[i], True)
      print (step_loss,answer,v_answers[i])
      if answer == v_answers[i]:
        counter += 1
    print ('counter %i' % counter)
    accuracy = counter /len(v_context)
    print ('accuracy = %f' % accuracy)


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
  if FLAGS.validation:
    validation()
  else:
    train()




if __name__ == '__main__':
  tf.app.run()




