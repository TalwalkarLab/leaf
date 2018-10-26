"""Bag-of-words logistic regression."""

import numpy as np
import os
import sys
import tensorflow as tf

from model import Model
from utils.language_utils import bag_of_words, get_word_emb_arr, val_to_vec

VOCAB_DIR = 'sent140/embs.json'


class ClientModel(Model):

    def __init__(self, lr, num_classes, input_dim=None):
        self.num_classes = num_classes
        _, _, self.vocab = get_word_emb_arr(VOCAB_DIR)
        if not input_dim:
            input_dim = len(self.vocab)
        self.input_dim = input_dim
        super(ClientModel, self).__init__(lr)

    def create_model(self):
        features = tf.placeholder(tf.float32, [None, self.input_dim])
        labels = tf.placeholder(tf.float32, [None, self.num_classes])
        
        W = tf.Variable(tf.random_normal(shape=[self.input_dim, self.num_classes]))
        b = tf.Variable(tf.random_normal(shape=[self.num_classes]))

        pred = tf.nn.softmax(tf.matmul(features, W) + b)

        # # Minimize error using cross entropy
        loss = tf.reduce_mean(-tf.reduce_sum(labels*tf.log(pred), reduction_indices=1))
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(labels,1))
        eval_metric_ops = tf.count_nonzero(correct_pred)
        
        return features, labels, train_op, eval_metric_ops

    def process_x(self, raw_x_batch):
        """
        Return:
            len(vocab) by len(raw_x_batch) np array
        """
        x_batch = [e[4] for e in raw_x_batch] # list of lines/phrases
        bags = [bag_of_words(line, self.vocab) for line in x_batch]
        bags = np.array(bags)
        return bags

    def process_y(self, raw_y_batch):
        y_batch = [int(e) for e in raw_y_batch]
        y_batch = [val_to_vec(self.num_classes, e) for e in y_batch]
        y_batch = np.array(y_batch)
        return y_batch


