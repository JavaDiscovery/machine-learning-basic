# coding=utf-8
# Author: zhengxiongfeng
# mail: 657019943@qq.com
#!/usr/bin/python

"""
练习： 使用eager api
"""

import tensorflow as tf
import numpy as np

print "setting eager mode."
tf.enable_eager_execution()
tfe = tf.contrib.eager # 用于定义变量

print "define constant tensors"
a = tf.constant(2)
print "a = %i" % a
b = tf.constant(3)
print "b = %i" % b

print "running operations, without tf.Session"
c = a + b
print "a + b = %i" % c
d = a * b
print "a * b = %i" %d

print "mixing operations with tensors and numpy arrays"
a = tf.constant([[2, 1],
                 [1, 0]], dtype=tf.float32)
print "tensor:\n a = %s" % a
b = np.array([[3, 0],
              [5, 1]], dtype=np.float32) #注意这里是np.float32
print "array:\n b = %s" % b

c = a + b
print "a(tensor) + b(array) = %s" % c
d = tf.matmul(a, b)
print "a * b = %s" % d

print "iterate through tensor a "
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print a[i][j], type(a[i][j])

print "iterate through array b "
for i in range(b.shape[0]):
    for j in range(b.shape[1]):
        print b[i][j], type(b[i][j])
