# coding=utf-8
# Author: zhengxiongfeng
# mail: 657019943@qq.com
#!/usr/bin/python

import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

digits = load_digits()
x = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
print y.shape
trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3)

def add_layer(input, input_size, output_size, layer_name, activation_func=None):
    weights = tf.Variable(tf.random_normal([input_size, output_size]))
    biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)
    wx_plus_bias = tf.add(tf.matmul(input, weights), biases)
    wx_plus_bias = tf.nn.dropout(wx_plus_bias, keep_prob) # 这里的keep_prob是一个placeholder
    if activation_func == None:
        outputs = wx_plus_bias
    else:
        outputs = activation_func(wx_plus_bias)
    tf.summary.histogram(layer_name + "/outputs", outputs)
    return outputs

xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

l1 = add_layer(xs, 64, 50, "layer1", activation_func=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, "layer2", activation_func=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
tf.summary.scalar("loss", cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 1. 合并所有summary
    merged = tf.summary.merge_all()
    # 2. 定义summary的FileWriter
    train_writer = tf.summary.FileWriter("log_dropout/train", sess.graph)
    test_writer = tf.summary.FileWriter("log_dropout/test", sess.graph)
    sess.run(init)
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: trainx, ys: trainy, keep_prob: 0.5})
        if i % 50 == 0:
            # 3. 先运行merge操作
            train_loss = sess.run(merged, feed_dict={xs: trainx, ys: trainy, keep_prob: 0.5})
            test_loss = sess.run(merged, feed_dict={xs: testx, ys: testy, keep_prob: 0.5})
            # 4. 最后写入文件
            train_writer.add_summary(train_loss, i)
            test_writer.add_summary(test_loss, i)

print "done"
