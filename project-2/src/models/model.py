import numpy as np

class Model(object):
    """
    Base model class
    """

    def __init__(self, embeddings, **kwargs):
        self.embeddings = embeddings

        # NOTE: These have to be set for all classes extending Model
        # If model should be trained on training data with single ending
        self.train_single = False
        # If model should be trained on training data with two endings
        self.train_multi = False

        # Tells the training loop if model contains variables that should be initialized
        self.initialize = True

    def predict(self, contexts, endings, **kwargs):
        return np.zeros(len(endings))

    def train_step(self, context_batch, end_batch, single=True, labels=None, **kwargs):
        raise NotImplementedError("Model can not be trained")

    def get_features(self):
        """
        Should return a list of useful features in combinations. Each feature
        of shape (batch_size, feature_size), where feature size is dynamic
        """
        return []

    def get_feed_dict(self, contexts, endings, **kwargs):
        """
        Should return a feed dict with neccesary data to feed to get features out

        """
        return {}
