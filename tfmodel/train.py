import argparse
import datetime
import json
import math
import os
import time
import numpy as np

import tensorflow as tf

from tfmodel.data_loader import Loader

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='train', choices=["train", "test"])
parser.add_argument("--output_dir", default=None, help="where to put output files")
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--batch_size", type=int, default=2, help="number of images in batch")
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

    def __init__(self, height, width, depth):
        self.input = tf.placeholder(tf.float32, [None, height, width, depth], name='X_placeholder')
        self.target = tf.placeholder(tf.float32, [None, 3], name='y_placeholder')

    def build_graph(self, reuse, train_mode):
        self.train_mode = train_mode
        with tf.variable_scope('cnn', reuse=reuse):
            self.output = self.input

            filter_nums = [32, 64, 128, 256, 512, 1024, 1024, 2048, 2048]
            for i, filter_num in enumerate(filter_nums):
                self.output = self.conv_layer(self.output, 'conv_%s' % i, filter_num)

            self.shape = tf.shape(self.output)
            self.output = tf.reshape(self.output, [self.shape[0], 2048])

            layer_sizes = [512, 256, 128, 64]
            for i, layer_size in enumerate(layer_sizes):
                self.output = tf.layers.dense(self.output, units=layer_size, activation=tf.nn.elu, use_bias=True, name='fc_%s' % i)
                if self.train_mode:
                    self.output = tf.nn.dropout(self.output, keep_prob=0.95)

            self.output = tf.layers.dense(self.output, units=3, activation=None, use_bias=True, name="fc_last")

            #  VGG-16 https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
            # filter_num = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            #
            # self.conv1_1 = self.conv_layer(self.input, 'conv1_1', filter_num[0])
            # self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2', filter_num[1])
            # self.pool1 = self.avg_pool(self.conv1_2, 'pool1')
            #
            # self.conv2_1 = self.conv_layer(self.pool1, "conv2_1", filter_num[2])
            # self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2", filter_num[3])
            # self.pool2 = self.avg_pool(self.conv2_2, 'pool2')
            #
            # self.conv3_1 = self.conv_layer(self.pool2, "conv3_1", filter_num[4])
            # self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2", filter_num[5])
            # self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3", filter_num[6])
            # self.pool3 = self.avg_pool(self.conv3_3, 'pool3')
            #
            # self.conv4_1 = self.conv_layer(self.pool3, "conv4_1", filter_num[7])
            # self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2", filter_num[8])
            # self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3", filter_num[9])
            # self.pool4 = self.avg_pool(self.conv4_3, 'pool4')
            #
            # self.conv5_1 = self.conv_layer(self.pool4, "conv5_1", filter_num[10])
            # self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2", filter_num[11])
            # self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3", filter_num[12])
            # self.pool5 = self.avg_pool(self.conv5_3, 'pool5')
            #
            # self.shape = tf.shape(self.pool5)
            # fc_input = tf.reshape(self.pool5, [self.shape[0], 131072])
            #
            # with tf.variable_scope('fc6'):
            #     self.fc6 = tf.layers.dense(fc_input, units=128, activation=tf.nn.elu, use_bias=True, name="fc6")
            # if train_mode:
            #     self.fc6 = tf.nn.dropout(self.fc6, keep_prob=0.95, name='dropout6')
            #
            # with tf.variable_scope('fc7'):
            #     self.fc7 = tf.layers.dense(self.fc6, units=128, activation=tf.nn.elu, use_bias=True, name="fc7")
            # if train_mode:
            #     self.fc7 = tf.nn.dropout(self.fc7, keep_prob=0.95, name='dropout7')
            #
            # with tf.variable_scope('fc8'):
            #     self.output = tf.layers.dense(self.fc7, units=3, activation=None, use_bias=True, name="fc8")
            #     # self.output = tf.layers.dense(self.fc7, units=3, activation=tf.nn.sigmoid, use_bias=True, name="fc8")

            self.color = tf.nn.sigmoid(self.output)

            output = tf.reshape(self.output, [-1])
            target = tf.reshape(self.target, [-1])

            # self.loss = tf.reduce_mean(tf.square(tf.subtract(self.target, self.output)))
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=target))

            vars = [var for var in tf.trainable_variables()]
            self.optimizer = tf.train.AdamOptimizer(a.lr, a.beta1)
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss, var_list=vars)
            self.train = self.optimizer.apply_gradients(self.grads_and_vars)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv(self, batch_input, out_channels, stride):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv

    def conv_layer(self, bottom, name, filter_num):
        with tf.variable_scope(name):
            conv = self.conv(bottom, filter_num, 2)
            elu = tf.nn.elu(conv)
        return elu


def draw(sess, model, save_path, depth):
    data_dir = '../data'
    front_position = np.load(os.path.join(data_dir, 'front_position.npy'))
    back_position = np.load(os.path.join(data_dir, 'back_position.npy'))
    front_lit = np.load(os.path.join(data_dir, 'front_irradiance.npy'))
    back_lit = np.load(os.path.join(data_dir, 'back_irradiance.npy'))
    object_mask = np.load(os.path.join(data_dir, 'object_mask.npy'))

    height, width, _ = object_mask.shape
    object_pos = []
    for h in range(height):
        for w in range(width):
            if object_mask[h, w, 0] == 1:
                object_pos.append([h, w])

    batch_size = 30
    image = np.zeros((height, width, 3)).astype('uint8')
    for start_index in range(0, len(object_pos), batch_size):
        batch = object_pos[start_index: start_index + batch_size]
        X = np.zeros(shape=(len(batch), height, width, depth), dtype=np.float32)
        for i, pos in enumerate(batch):
            position = front_position[pos[0], pos[1]]
            front_relative_position = front_position - position
            back_relative_position = back_position - position
            front_relative_position *= object_mask
            back_relative_position *= object_mask

            front_relative_distance = np.sqrt(
                front_relative_position[:, :, 0] * front_relative_position[:, :, 0] + front_relative_position[:, :,
                                                                                      1] * front_relative_position[:, :,
                                                                                           1] + front_relative_position[
                                                                                                :, :,
                                                                                                2] * front_relative_position[
                                                                                                     :, :, 2])
            back_relative_distance = np.sqrt(
                back_relative_position[:, :, 0] * back_relative_position[:, :, 0] + back_relative_position[:, :,
                                                                                    1] * back_relative_position[:, :,
                                                                                         1] + back_relative_position[:,
                                                                                              :,
                                                                                              2] * back_relative_position[
                                                                                                   :, :, 2])
            front_relative_distance = front_relative_distance[..., None]
            back_relative_distance = back_relative_distance[..., None]

            X[i] = np.concatenate((front_relative_distance, back_relative_distance, front_lit, back_lit), axis=2)
        colors = sess.run(model.color, {model.input: X})
        for i, color in enumerate(colors):
            pos = batch[i]
            image[pos[0], pos[1]] = [int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)]

    from PIL import Image
    img = Image.fromarray(image)
    img.save(os.path.join(save_path))


def main():
    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    # no need to load options from options.json
    loader = Loader(a.batch_size)

    train_cnn = CNN(loader.height, loader.width, loader.depth)
    train_cnn.build_graph(False, True)

    val_cnn = CNN(loader.height, loader.width, loader.depth)
    val_cnn.build_graph(True, False)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=50)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in train_cnn.grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)
        

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("parameter_count =", sess.run(parameter_count))

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(a.output_dir, sess.graph)

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        if a.mode == 'test':
            draw(sess, val_cnn, os.path.join(a.output_dir, 'test.jpg'), loader.depth)
        else:
            # training
            start = time.time()
            for epoch in range(a.max_epochs):
                def should(freq):
                    return freq > 0 and ((epoch + 1) % freq == 0 or epoch == a.max_epochs - 1)

                fetches = {
                    "train": train_cnn.train,
                    "loss": train_cnn.loss
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
                    summary = sess.run(merged)
                    writer.add_summary(summary, global_step=epoch)
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

                if should(a.display_freq):
                    print('drawing...')
                    draw(sess, val_cnn, os.path.join(a.output_dir, '%s.jpg' % epoch), loader.depth)


if __name__ == '__main__':
    main()


