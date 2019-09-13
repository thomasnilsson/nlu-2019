import numpy as np
import tensorflow as tf

import constants as C
from models.model import Model


class SentimentFeedForward(Model):
    def __init__(self, learning_rate=1e-3, **kwargs):
        super(SentimentFeedForward, self).__init__(**kwargs)

        self.train_single = True  # Train on single ending data

        # Initializer
        self.initializer = tf.contrib.layers.xavier_initializer(seed=C.rand_seed)

        # Global step
        self.global_step = tf.train.get_or_create_global_step()

        # Construct tf graph
        self.context_sentiment_input = tf.placeholder(shape=(None, 8),
                                            dtype=tf.float32, name="context_sentiment")
        self.end_sentiment_input = tf.placeholder(shape=(None, 2),
                                            dtype=tf.float32, name="end_sentiment")

        # Make end sentiment full distribution (3 entries)
        self.end_sentiment = tf.stack([
                self.end_sentiment_input[:,0],
                self.end_sentiment_input[:,1],
                1 - (self.end_sentiment_input[:,0] + self.end_sentiment_input[:,1])
            ], axis=1)

        hidden = tf.layers.dense(inputs=self.context_sentiment_input,
                                 units=10,
                                 activation=tf.nn.sigmoid,
                                 kernel_initializer=self.initializer,
                                 bias_initializer=self.initializer, name="hidden_first")
        hidden = tf.layers.dense(inputs=self.context_sentiment_input,
                                 units=3,
                                 kernel_initializer=self.initializer,
                                 bias_initializer=self.initializer, name="hidden_second")
        self.context_sentiment = tf.nn.softmax(hidden, axis=1)

        # Weird hack, only works with 2 entries in sentiment vector
        self.distances = tf.losses.cosine_distance(
                tf.math.l2_normalize(self.end_sentiment[:,:2], axis=1),
                tf.math.l2_normalize(self.context_sentiment[:,:2], axis=1),
                axis=1,reduction=tf.losses.Reduction.NONE)
        self.loss = tf.reduce_mean(self.distances)

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, name="train_op",
                                                global_step=self.global_step)

    def concat_sentences(self, sentence_batch):
        out = []
        for story in sentence_batch:
            out.append(np.array(story).flatten())

        return out

    def train_step(self, context_batch, end_batch, sess, single, labels=None, **kwargs):
        fetches = [
            self.loss,
            self.global_step,
            self.train_op,
        ]

        feed_dict = self.get_feed_dict(context_batch, end_batch)

        train_output = sess.run(fetches, feed_dict)
        train_loss = train_output[0]
        step = train_output[1]

        return train_loss, step

    def predict(self, sentences, endings, sess, **kwargs):
        fd0 = self.get_feed_dict(sentences, [s[0] for s in endings])
        fd1 = self.get_feed_dict(sentences, [s[1] for s in endings])

        d0 = sess.run(self.distances, fd0)
        d1 = sess.run(self.distances, fd1)

        dists = np.stack((d0, d1), axis=1)

        return np.argmin(dists, axis=1).flatten()

    def get_features(self):
        return [self.end_sentiment, self.context_sentiment]

    def get_feed_dict(self, contexts, endings, **kwargs):
        context_sentiment = self.concat_sentences(contexts)
        end_sentiment = self.concat_sentences(endings)

        feed_dict = {
            self.context_sentiment_input: context_sentiment,
            self.end_sentiment_input: end_sentiment
        }

        return feed_dict
