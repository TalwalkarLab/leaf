import tensorflow as tf

from model import Model
import numpy as np


IMAGE_SIZE = 28


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        """Model function for CNN."""
        with tf.device('/gpu:0'):
            features = tf.placeholder(
                tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
            labels = tf.placeholder(tf.int64, shape=[None], name='labels')
            input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
            conv1 = tf.layers.conv2d(
              inputs=input_layer,
              filters=32,
              kernel_size=[5, 5],
              padding="same",
              activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
            dense = tf.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.relu)
            logits = tf.layers.dense(inputs=dense, units=self.num_classes)
            predictions = {
              "classes": tf.argmax(input=logits, axis=1),
              "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            # TODO: Confirm that opt initialized once is ok?
            train_op = self.optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, train_op, eval_metric_ops, loss

    # todo fedsp
    def create_fedsp_model(self):
        """Model function for FedSP-CNN."""
        features = tf.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')
        input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        # global encoder
        global_conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu, name='global_conv1')
        global_pool1 = tf.layers.max_pooling2d(inputs=global_conv1, pool_size=[2, 2], strides=2)
        global_conv2 = tf.layers.conv2d(
            inputs=global_pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu, name='global_conv2')
        global_pool2 = tf.layers.max_pooling2d(inputs=global_conv2, pool_size=[2, 2], strides=2)
        global_pool2_flat = tf.reshape(global_pool2, [-1, 7 * 7 * 64])

        # local encoder
        local_conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu, name='local_conv1')
        local_pool1 = tf.layers.max_pooling2d(inputs=local_conv1, pool_size=[2, 2], strides=2)
        local_conv2 = tf.layers.conv2d(
            inputs=local_pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu, name='local_conv2')
        local_pool2 = tf.layers.max_pooling2d(inputs=local_conv2, pool_size=[2, 2], strides=2)
        local_pool2_flat = tf.reshape(local_pool2, [-1, 7 * 7 * 64])

        concat_res = tf.concat([global_pool2_flat, local_pool2_flat], 1)

        dense = tf.layers.dense(inputs=concat_res, units=2048, activation=tf.nn.relu)

        logits = tf.layers.dense(inputs=dense, units=self.num_classes)

        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # TODO: Confirm that opt initialized once is ok?
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))

        return features, labels, train_op, eval_metric_ops, loss

    @staticmethod
    def process_x(raw_x_batch):
        return np.array(raw_x_batch)

    @staticmethod
    def process_y(raw_y_batch):
        return np.array(raw_y_batch)
