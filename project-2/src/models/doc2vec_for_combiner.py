import numpy as np
import tensorflow as tf
import pickle

from models.model import Model


class doc2vecForCombiner(Model):
    """
    Model only used to load pre-trained doc2vec model.
    (It is NOT the doc2vec model itself!)
    """

    def __init__(self, path_to_d2v, **kwargs):
        super(doc2vecForCombiner, self).__init__(**kwargs)

        self.doc2vec_cosine = tf.placeholder(dtype=tf.float32,
                                             shape=(None, 1), name="doc2vec_cosine")

        self.trained_d2v = pickle.load(open(path_to_d2v, "rb"))

    def get_features(self):
        return [self.doc2vec_cosine]

    def get_feed_dict(self, contexts, endings, **kwargs):
        """
        Using template from feature model...
        """

        # Extract cosines from doc2vec
        dummy = self.trained_d2v.predict(contexts, endings)
        feed_dict = {self.doc2vec_cosine: self.trained_d2v.get_features()}

        return feed_dict

    def predict(self, contexts, endings, **kwargs):
        raise NotImplementedError("Should not be used for prediction")

    def train_step(self, context_batch, end_batch, single=True, labels=None, **kwargs):
        raise NotImplementedError("Model can not be trained")
