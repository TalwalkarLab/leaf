import collections
import numpy as np
import os
import pickle
import sys
import tensorflow as tf

from tensorflow.contrib import rnn

from model import Model

VOCABULARY_PATH = '../data/reddit/vocab/reddit_vocab.pck'


# Code adapted from https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
# and https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
class ClientModel(Model):
    def __init__(self, seed, lr, seq_len, n_hidden, num_layers,
        keep_prob=1.0, max_grad_norm=5, init_scale=0.1):

        self.seq_len = seq_len
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.max_grad_norm = max_grad_norm

        # initialize vocabulary
        self.vocab, self.vocab_size, self.unk_symbol, self.pad_symbol = self.load_vocab()

        self.initializer = tf.random_uniform_initializer(-init_scale, init_scale)

        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):

        with tf.variable_scope('language_model', reuse=None, initializer=self.initializer):
            features = tf.placeholder(tf.int32, [None, self.seq_len], name='features')
            labels = tf.placeholder(tf.int32, [None, self.seq_len], name='labels')
            self.sequence_length_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
            self.sequence_mask_ph = tf.placeholder(tf.float32, [None, self.seq_len], name='seq_mask_ph')

            self.batch_size = tf.shape(features)[0]

            # word embedding
            embedding = tf.get_variable(
                'embedding', [self.vocab_size, self.n_hidden], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, features)

            # LSTM
            output, state = self._build_rnn_graph(inputs) # TODO: check!

            # softmax
            with tf.variable_scope('softmax'):
                softmax_w = tf.get_variable(
                    'softmax_w', [self.n_hidden, self.vocab_size], dtype=tf.float32)
                softmax_b = tf.get_variable('softmax_b', [self.vocab_size], dtype=tf.float32)
            
            logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

            # correct predictions
            labels_reshaped = tf.reshape(labels, [-1])
            pred = tf.cast(tf.argmax(logits, 1), tf.int32)
            correct_pred = tf.cast(tf.equal(pred, labels_reshaped), tf.int32)
            
            # predicting unknown is always considered wrong
            unk_tensor = tf.fill(tf.shape(labels_reshaped), self.unk_symbol)
            pred_unk = tf.cast(tf.equal(pred, unk_tensor), tf.int32)
            correct_unk = tf.multiply(pred_unk, correct_pred)

            # predicting padding is always considered wrong
            pad_tensor = tf.fill(tf.shape(labels_reshaped), 0)
            pred_pad = tf.cast(tf.equal(pred, pad_tensor), tf.int32)
            correct_pad = tf.multiply(pred_pad, correct_pred)

            # Reshape logits to be a 3-D tensor for sequence loss
            logits = tf.reshape(logits, [-1, self.seq_len, self.vocab_size])

            # Use the contrib sequence loss and average over the batches
            loss = tf.contrib.seq2seq.sequence_loss(
                logits,
                labels,
                weights=self.sequence_mask_ph,
                average_across_timesteps=False,
                average_across_batch=True)

            # Update the cost
            #self.cost = tf.reduce_sum(loss)
            self.cost = tf.reduce_mean(loss)
            self.final_state = state

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)
            train_op = self.optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.train.get_or_create_global_step())

            eval_metric_ops = tf.count_nonzero(correct_pred) - tf.count_nonzero(correct_unk) - tf.count_nonzero(correct_pad)

        return features, labels, train_op, eval_metric_ops, self.cost

    def _build_rnn_graph(self, inputs):
        def make_cell():
            cell = tf.contrib.rnn.LSTMBlockCell(self.n_hidden, forget_bias=0.0)
            if self.keep_prob < 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            return cell

        cell = tf.nn.rnn_cell.MultiRNNCell(
            [make_cell() for _ in range(self.num_layers)], state_is_tuple=True)

        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        outputs, state = tf.nn.dynamic_rnn(
            cell, inputs, initial_state=self.initial_state, sequence_length=self.sequence_length_ph)
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.n_hidden])
        return output, state

    def process_x(self, raw_x_batch):
        tokens = self._tokens_to_ids([s for s in raw_x_batch])
        lengths = np.sum(tokens != self.pad_symbol, axis=1)
        return tokens, lengths

    def process_y(self, raw_y_batch):
        tokens = self._tokens_to_ids([s for s in raw_y_batch])
        return tokens

    def _tokens_to_ids(self, raw_batch):
        def tokens_to_word_ids(tokens, vocab):
            return [vocab[word] for word in tokens]

        to_ret = [tokens_to_word_ids(seq, self.vocab) for seq in raw_batch]
        return np.array(to_ret)

    def batch_data(self, data, batch_size):
        data_x = data['x']
        data_y = data['y']

        perm = np.random.permutation(len(data['x']))
        data_x = [data_x[i] for i in perm]
        data_y = [data_y[i] for i in perm]

        # flatten lists
        def flatten_lists(data_x_by_comment, data_y_by_comment):
            data_x_by_seq, data_y_by_seq, mask_by_seq = [], [], []
            for c, l in zip(data_x_by_comment, data_y_by_comment):
                data_x_by_seq.extend(c)
                data_y_by_seq.extend(l['target_tokens'])
                mask_by_seq.extend(l['count_tokens'])

            if len(data_x_by_seq) % batch_size != 0:
                dummy_tokens = [self.pad_symbol for _ in range(self.seq_len)]
                dummy_mask = [0 for _ in range(self.seq_len)]
                num_dummy = batch_size - len(data_x_by_seq) % batch_size

                data_x_by_seq.extend([dummy_tokens for _ in range(num_dummy)])
                data_y_by_seq.extend([dummy_tokens for _ in range(num_dummy)])
                mask_by_seq.extend([dummy_mask for _ in range(num_dummy)])

            return data_x_by_seq, data_y_by_seq, mask_by_seq
        
        data_x, data_y, data_mask = flatten_lists(data_x, data_y)

        for i in range(0, len(data_x), batch_size):
            batched_x = data_x[i:i+batch_size]
            batched_y = data_y[i:i+batch_size]
            batched_mask = data_mask[i:i+batch_size]

            input_data, input_lengths = self.process_x(batched_x)
            target_data = self.process_y(batched_y)

            yield (input_data, target_data, input_lengths, batched_mask)

    def run_epoch(self, data, batch_size=5):
        state = None

        fetches = {
            'cost': self.cost,
            'final_state': self.final_state,
        }

        for input_data, target_data, input_lengths, input_mask in self.batch_data(data, batch_size):

            feed_dict = {
                self.features: input_data,
                self.labels: target_data,
                self.sequence_length_ph: input_lengths,
                self.sequence_mask_ph: input_mask,
            }

            # We need to feed the input data so that the batch size can be inferred.
            if state is None:
                state = self.sess.run(self.initial_state, feed_dict=feed_dict)

            for i, (c, h) in enumerate(self.initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

            with self.graph.as_default():
                _, vals = self.sess.run([self.train_op, fetches], feed_dict=feed_dict)
            
            state = vals['final_state']

    def test(self, data, batch_size=5):
        tot_acc, tot_samples = 0, 0
        tot_loss, tot_batches = 0, 0

        for input_data, target_data, input_lengths, input_mask in self.batch_data(data, batch_size):

            with self.graph.as_default():
                acc, loss = self.sess.run(
                    [self.eval_metric_ops, self.loss], 
                    feed_dict={
                        self.features: input_data,
                        self.labels: target_data,
                        self.sequence_length_ph: input_lengths, 
                        self.sequence_mask_ph: input_mask,
                    })
            
            tot_acc += acc
            tot_samples += np.sum(input_lengths)

            tot_loss += loss
            tot_batches += 1

        acc = float(tot_acc) / tot_samples # this top 1 accuracy considers every pred. of unknown and padding as wrong
        loss = tot_loss / tot_batches # the loss is already averaged over samples
        return {'accuracy': acc, 'loss': loss}

    def load_vocab(self):
        vocab_file = pickle.load(open(VOCABULARY_PATH, 'rb'))
        vocab = collections.defaultdict(lambda: vocab_file['unk_symbol'])
        vocab.update(vocab_file['vocab'])

        return vocab, vocab_file['size'], vocab_file['unk_symbol'], vocab_file['pad_symbol']

