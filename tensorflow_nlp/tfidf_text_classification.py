# coding=utf-8
# Author: zhengxiongfeng
# mail: 657019943@qq.com
#!/usr/bin/python

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import time

VOCAB_SIZE = 20000
N_CLASS = 2
BATCH_SIZE = 32
N_EPOCH = 2
LR = 5e-3

def sparse_tfidf(x):
    t0 = time.time()
    count = np.zeros(len(x), VOCAB_SIZE)
    for i, indices in enumerate(x):
        for idx in indices:
            count[i, idx] += 1
    print "%.2f secs ==> Document-Term Matrix.\n" % (time.time() - t0)

    t0 = time.time()
    x= TfidfTransformer.fit_transform(count)
    print "%.2f secs ==> TF-IDF transform.\n" % (time.time() - t0)
    return x

def next_train_batch(x, y):
    for i in range(0, x.shape[0], BATCH_SIZE):
        yield x[i: i+BATCH_SIZE].toarray(), y[i: i+BATCH_SIZE]
