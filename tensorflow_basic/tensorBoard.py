# coding=utf-8
# Author: zhengxiongfeng
# mail: 657019943@qq.com
#!/usr/bin/python

import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, n_layer, activation_func=None):
    layer_name = "layer%s" % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope("Weights"):
            weights = tf.Variable(tf.random_normal([in_size, out_size]), name="W")
            tf.summary.histogram(layer_name+"/weights", weights)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size])+0.1, name="b")
            tf.summary.histogram(layer_name+"/biases", biases)
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)
        if activation_func is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_func(Wx_plus_b)
        return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope("input"):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_input")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_input")

l1 = add_layer(xs, 1, 10, 1, activation_func=tf.nn.relu)
prediction = add_layer(l1, 10, 1, 2, activation_func=None)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar("loss", loss)

with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 1.合并所有summary
    merged = tf.summary.merge_all()
    # 2. 定义FileWriter,并指定本地文件夹
    writer = tf.summary.FileWriter("log_tensorboard/", sess.graph)
    sess.run(init)
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            # 3. 先运行erge操作
            result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            # 4. 运行中间结果保存到本地
            writer.add_summary(result, i)

print "done"

