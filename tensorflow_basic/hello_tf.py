# coding=utf-8
# Author: zhengxiongfeng
# mail: 657019943@qq.com
#!/usr/bin/python

import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = weight * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print step, ": ", sess.run(weight), ", ", sess.run(biases)

sess.close()

print "done"
