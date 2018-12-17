# coding=utf-8
# Author: zhengxiongfeng
# mail: 657019943@qq.com
#!/usr/bin/python

"""
练习：保存模型到本地，加载本地模型
"""

import tensorflow as tf
import numpy as np

## 保存模型到本地
#w = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name="weights")
#b = tf.Variable([1, 2, 3],dtype=tf.float32, name="biases")
#
#init = tf.global_variables_initializer()
#
#saver = tf.train.Saver()
#
#with tf.Session() as sess:
#    sess.run(init)
#    save_path = saver.save(sess, "my_net/save_net.ckpt")
#    print "save model to path: ", save_path

## 加载模型
w = tf.Variable(np.arange(6).reshape(2, 3), name="weights", dtype=tf.float32)
# 注意这里b的shape
b = tf.Variable(np.arange(3).reshape(3), name="biases", dtype=tf.float32)

#注意：这里定义的变量不需要初始化计算
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print "weights: ", sess.run(w)
    print "biases: ", sess.run(b)
