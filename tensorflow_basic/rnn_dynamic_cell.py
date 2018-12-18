# coding=utf-8
# Author: zhengxiongfeng
# mail: 657019943@qq.com
#!/usr/bin/python
"""
练习：tf.nn.dynamic_rnn的用法
"""

import tensorflow as tf
import numpy as np

import matplotlib.pylab as plt

batch_size = 4
num_classes = 2
num_steps = 10
state_size = 4
learning_rate = 0.2

def gen_data(size=1000000):
    """
           生成数据:
           输入数据X：在时间t，Xt的值有50%的概率为1，50%的概率为0；
           输出数据Y：在实践t，Yt的值有50%的概率为1，50%的概率为0，除此之外，如果`Xt-3 == 1`，Yt为1的概率增加50%， 如果`Xt-8 == 1`，则Yt为1的概率减少25%， 如果上述两个条件同时满足，则Yt为1的概率为75%。
    """
    x = np.array(np.random.choice(2, size=(size,))) #[0, 1, ..., 0, 1]共size大小
    y = []
    for i in range(size):
        threshold = 0.5
        if x[i - 3] == 1:
            threshold += 0.5
        if x[i - 8 ] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            y.append(0)
        else:
            y.append(1)
    return x, np.array(y)

def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data # raw_data利用gen_data得到
    data_x = raw_x.reshape(-1, batch_size, num_steps) # n_batch * batch_size * num_steps
    data_y = raw_y.reshape(-1, batch_size, num_steps)
    for i in range(data_x.shape[0]):
        yield (data_x[i], data_y[i])

def gen_epochs(n):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)

x = tf.placeholder(tf.int32, [batch_size, num_steps], name="input_placeholder")
y = tf.placeholder(tf.int32, [batch_size, num_steps], name="output_placeholder")

init_state = tf.zeros([batch_size, state_size])

rnn_inputs = tf.one_hot(x, num_classes)

cell = tf.contrib.rnn.BasicRNNCell(state_size)
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

with tf.variable_scope("softmax"):
    w= tf.get_variable("w", [state_size, num_classes])
    b = tf.get_variable("b", [num_classes], initializer=tf.constant_initializer(0.0))

logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), w) + b, [batch_size, num_steps, num_classes])
predictions = tf.nn.softmax(logits)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=predictions)

total_loss = tf.reduce_mean(losses)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def train_network(num_epochs, num_steps, state_size=4, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs)):
            training_loss = 0
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print "\n Epoch: ", idx
            for step, (x_sample, y_sample) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _, _ = sess.run(\
                    [losses, total_loss, final_state, train_step, rnn_outputs], \
                    feed_dict={x: x_sample, y: y_sample, init_state: training_state})
                #print rnn_outputs.shape
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print "average loss at step %d for last 100 steps: %f" % (step, training_loss/100.0)
                    training_losses.append(training_loss/100.0)
                training_loss = 0
        return training_losses

train_losses = train_network(5, num_steps)
plt.plot(train_losses)
plt.show()

