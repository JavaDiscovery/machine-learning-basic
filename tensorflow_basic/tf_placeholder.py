# coding=utf-8
# Author: zhengxiongfeng
# mail: 657019943@qq.com
#!/usr/bin/python

import tensorflow as tf

input1 = tf.placeholder(dtype=tf.float32)
input2 = tf.placeholder(dtype=tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print sess.run(output, feed_dict={input1: [3], input2: [5]})

print "done"