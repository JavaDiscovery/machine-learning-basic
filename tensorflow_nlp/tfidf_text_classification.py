# coding=utf-8
# Author: zhengxiongfeng
# mail: 657019943@qq.com
#!/usr/bin/python

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

VOCAB_SIZE = 20000
N_CLASS = 2
BATCH_SIZE = 32
N_EPOCH = 2
LR = 5e-3

def sparse_tfidf(x):
    t0 = time.time()
    count = np.zeros((len(x), VOCAB_SIZE))
    for i, indices in enumerate(x):
        for idx in indices:
            count[i, idx] += 1
    print ("%.2f secs ==> Document-Term Matrix.\n" % (time.time() - t0))

    t0 = time.time()
    x= TfidfTransformer().fit_transform(count)
    print ("%.2f secs ==> TF-IDF transform.\n" % (time.time() - t0))
    return x

def next_train_batch(x, y):
    for i in range(0, x.shape[0], BATCH_SIZE):
        yield x[i: i+BATCH_SIZE].toarray(), y[i: i+BATCH_SIZE]

def next_test_batch(x):
    for i in range(0, x.shape[0], BATCH_SIZE):
        yield x[i: i+BATCH_SIZE].toarray()

def train_input_fn(x_train, y_train):
    dataset = tf.data.Dataset.from_generator(
        lambda: next_train_batch(x_train, y_train), (tf.float32, tf.int64),
        (tf.TensorShape([None, VOCAB_SIZE]), tf.TensorShape([None]))
    )
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def predict_input_fn(x_test):
    dataset = tf.data.Dataset.from_generator(
        lambda : next_test_batch(x_test), tf.float32,
        tf.TensorShape([None, VOCAB_SIZE])
    )
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def model_fn(features, labels, mode, params):
    logits = tf.layers.dense(features, N_CLASS)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=tf.argmax(logits, -1))
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels
        ))
        train_op = tf.train.AdamOptimizer(LR).minimize(loss_op,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, train_op=train_op)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
x_train = sparse_tfidf(x_train)
x_test = sparse_tfidf(x_test)

estimator = tf.estimator.Estimator(model_fn)

for _ in range(N_EPOCH):
    estimator.train(lambda : train_input_fn(*shuffle(x_train, y_train)))
    y_pred = np.fromiter(estimator.predict(lambda : predict_input_fn(x_test)), np.int32)
    print ("\nValidation Accuracy: %.4f" % (y_pred==y_test).mean())
