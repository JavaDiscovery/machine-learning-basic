# coding=utf-8
# Author: zhengxiongfeng
# mail: 657019943@qq.com
#!/usr/bin/python

"""
基于skipgram的词向量
"""
from collections import Counter

import tensorflow as tf
import numpy as np
import re

PARAMS = {
    "min_freq": 5,
    "skip_window": 5,
    "n_sampled": 100,
    "embed_dim": 200,
    "sample_words": ["six", "gold", "japan", "college"],
    "batch_size": 1000,
    "n_epochs": 10,
}

def preprocess_text(text):
    text = text.replace("\n", " ")
    text = re.sub("\s+", " ", text).strip().lower()

    words = text.split()
    word2freq = Counter(words)
    words = [word for word in words if word2freq[word] > PARAMS["min_freq"]]
    print "Total words:", len(words)

    _words = set(words)
    PARAMS["word2idx"] = {c: i for i, c in enumerate(_words)}
    PARAMS["idx2word"] = {i: c for i, c in enumerate(_words)}
    PARAMS["vocab_size"] = len(PARAMS["idx2word"])
    print "Vocabulary size:", PARAMS["vocab_size"]

    indexed = [PARAMS["word2idx"][w] for w in words]
    #indexed = filter_high_freq(indexed)
    print "word preprocessing completed."
    return indexed

def filter_high_freq(int_words, t=1e-5, threshold=0.8):
    """感觉这个函数没啥意义啊"""
    int_word_counts = Counter(int_words)
    total_count = len(int_words)
    word_freqs = {w: float(c) / total_count for w, c in int_word_counts.items()}
    prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in int_word_counts}
    train_words = [w for w in int_words if prob_drop[w] < threshold]
    return  train_words

def make_data(int_words):
    x, y = [], []
    for i in range(0, len(int_words)):
        input_w = int_words[i]
        labels = get_y(int_words, i)
        x.extend([input_w] * len(labels))
        y.extend(labels)
    return x, y

def get_y(words, idx):
    skip_window = np.random.randint(1, PARAMS["skip_window"] + 1)
    left = idx - skip_window if (idx - skip_window) > 0 else 0
    right = idx + skip_window
    y = words[left: idx] + words[idx+1: right+1]
    return list(set(y))


def model_fn(features, labels, mode, params):
    W = tf.get_variable("softmax_W", [PARAMS["vocab_size"], PARAMS["embed_dim"]])
    b = tf.get_variable("softmax_b", [PARAMS["vocab_size"]])
    E = tf.get_variable("embedding", [PARAMS["vocab_size"], PARAMS["embed_dim"]])

    embedded = tf.nn.embedding_lookup(E, features)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss_op = tf.reduce_mean(tf.nn.sampled_softmax_loss(
            weights=W,
            biases=b,
            labels=labels,
            inputs=embedded,
            num_sampled=PARAMS["n_sampled"],
            num_classes=PARAMS["vocab_size"]
        ))
        train_op = tf.train.AdamOptimizer().minimize(
            loss_op, global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, train_op=train_op)
    if mode == tf.estimator.ModeKeys.PREDICT:
        normalized_E = tf.nn.l2_normalize(E, -1)
        sample_E = tf.nn.embedding_lookup(normalized_E, features)
        similarity = tf.matmul(sample_E, normalized_E, transpose_b=True)
        return tf.estimator.EstimatorSpec(mode, predictions=similarity)

def print_neighbours(similarity, top_k=5):
    for i in range(len(PARAMS["sample_words"])):
        neighbours = (-similarity[i]).argsort()[1: top_k+1]
        log = "nearest to [%s]:" % PARAMS["sample_words"][i]
        for k in range(top_k):
            neighbour = PARAMS["idx2word"][neighbours[k]]
            log = "%s %s," % (log, neighbour)
        print log

with open("data/ptb.train.txt") as f:
    x_train, y_train = make_data(preprocess_text(f.read()))

estimator = tf.estimator.Estimator(model_fn)
estimator.train(tf.estimator.inputs.numpy_input_fn(
    np.array(x_train), np.expand_dims(y_train, -1),
    batch_size=PARAMS["batch_size"],
    num_epochs=PARAMS["n_epochs"],
    shuffle=True
))
sim = np.array(list(estimator.predict(tf.estimator.inputs.numpy_input_fn(
    x = np.array([PARAMS["word2idx"][w] for w in PARAMS["sample_words"]]),
    shuffle=False
))))
print_neighbours(sim)
