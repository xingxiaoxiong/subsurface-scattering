import argparse
import datetime
import json
import math
import os
import time

import tensorflow as tf

from tfmodel.data_loader import Loader

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--output_dir", default=None, help="where to put output files")
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.9, help="momentum term of adam")
parser.add_argument("--max_epochs", type=int, default=1000, help="number of training epochs")

# in epochs
parser.add_argument("--save_freq", type=int, default=5, help="save model every save_freq epochs, 0 to disable")
parser.add_argument("--summary_freq", type=int, default=5, help="update summaries every summary_freq epochs")
parser.add_argument("--progress_freq", type=int, default=1, help="display progress every progress_freq epochs")
parser.add_argument("--validation_freq", type=int, default=5, help="display progress every validation_freq epochs")
parser.add_argument("--display_freq", type=int, default=0, help="write images every display_freq epochs")

a = parser.parse_args()
if not a.output_dir:
    output_prepath = 'output'
    if not os.path.isdir(output_prepath):
        os.makedirs(output_prepath)
    a.output_dir = os.path.join(output_prepath, datetime.datetime.now().strftime("%I_%M%p_on_%B_%d_%Y"))
    if not os.path.isdir(a.output_dir):
        os.makedirs(a.output_dir)


class CNN:

    def __init__(self, batch_size, height, width, depth):
        self.batch_size = batch_size
        self.input = tf.placeholder(tf.float32, [batch_size, height, width, depth], name='X_placeholder')
        self.target = tf.placeholder(tf.float32, [batch_size, 3], name='y_placeholder')

    def build_graph(self, reuse, train_mode):
        with tf.variable_scope('cnn', reuse=reuse):
            #  VGG-16 https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
            filter_num = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

            self.conv1_1 = self.conv_layer(self.input, 'conv1_1', filter_num[0])
            self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2', filter_num[1])
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            self.conv2_1 = self.conv_layer(self.pool1, "conv2_1", filter_num[2])
            self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2", filter_num[3])
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

            self.conv3_1 = self.conv_layer(self.pool2, "conv3_1", filter_num[4])
            self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2", filter_num[5])
            self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3", filter_num[6])
            self.pool3 = self.max_pool(self.conv3_3, 'pool3')

            self.conv4_1 = self.conv_layer(self.pool3, "conv4_1", filter_num[7])
            self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2", filter_num[8])
            self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3", filter_num[9])
            self.pool4 = self.max_pool(self.conv4_3, 'pool4')

            self.conv5_1 = self.conv_layer(self.pool4, "conv5_1", filter_num[10])
            self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2", filter_num[11])
            self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3", filter_num[12])
            self.pool5 = self.max_pool(self.conv5_3, 'pool5')

            self.pool5 = tf.reshape(self.pool5, [self.batch_size, -1])

            with tf.variable_scope('fc6'):
                self.fc6 = tf.layers.dense(self.pool5, units=128, activation=tf.nn.relu, use_bias=True, name="fc6")
            if train_mode:
                self.fc6 = tf.nn.dropout(self.fc6, keep_prob=0.5, name='dropout6')

            with tf.variable_scope('fc7'):
                self.fc7 = tf.layers.dense(self.fc6, units=128, activation=tf.nn.relu, use_bias=True, name="fc7")
            if train_mode:
                self.fc7 = tf.nn.dropout(self.fc7, keep_prob=0.5, name='dropout7')

            with tf.variable_scope('fc8'):
                self.output = tf.layers.dense(self.fc7, units=3, activation=tf.nn.sigmoid, use_bias=True, name="fc8")

            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.target, self.output)))
            self.optimize = tf.train.AdamOptimizer(a.lr, a.beta1).minimize(self.loss)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name, filter_num):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(bottom, filters=filter_num, kernel_size=3, padding='same')
            relu = tf.nn.relu(conv)
            return relu


def main():
    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    # no need to load options from options.json
    loader = Loader(a.batch_size)

    train_cnn = CNN(loader.batch_size, loader.height, loader.width, loader.depth)
    train_cnn.build_graph(False, True)

    val_cnn = CNN(loader.batch_size, loader.height, loader.width, loader.depth)
    val_cnn.build_graph(True, False)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=50)

    logdir = a.output_dir
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        if a.mode == 'test':
            pass
        else:
            # training
            start = time.time()
            for epoch in range(a.max_epochs):
                def should(freq):
                    return freq > 0 and ((epoch + 1) % freq == 0 or epoch == a.max_epochs - 1)

                fetches = {
                    "train": train_cnn.optimize,
                    "loss": train_cnn.loss,
                }

                training_loss = 0
                for _ in range(loader.ntrain):
                    X, y = loader.next_batch(0)
                    results = sess.run(fetches, {train_cnn.input: X, train_cnn.target: y})
                    training_loss += results['loss']
                training_loss /= loader.ntrain

                if should(a.validation_freq):
                    print('validating model')
                    validation_loss = 0
                    for _ in range(loader.nval):
                        X, y = loader.next_batch(1)
                        loss = sess.run(val_cnn.loss, {val_cnn.input: X, val_cnn.target: y})
                        validation_loss += loss
                    validation_loss /= loader.nval

                if should(a.summary_freq):
                    print("recording summary")
                    with open(os.path.join(a.output_dir, 'loss_record.txt'), "a") as loss_file:
                        loss_file.write("%s\t%s\t%s\n" % (epoch, training_loss, validation_loss))

                if should(a.progress_freq):
                    rate = (epoch + 1) / (time.time() - start)
                    remaining = (a.max_epochs - 1 - epoch) / rate
                    print("progress  epoch %d  remaining %dh" % (epoch, remaining / 3600))
                    print("training loss", training_loss)

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=epoch)


if __name__ == '__main__':
    main()


