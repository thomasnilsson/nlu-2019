import numpy as np
import tensorflow as tf

from models.model import Model

class FeatureModel(Model):
    """
    Model to hold hand-crafted features
    """
    pronouns_m = ["he", "him", "his"]
    pronouns_f =  ["she", "her", "hers"]

    inverse_words = ["not", "n't"]

    def __init__(self, word_to_id, **kwargs):
        super(FeatureModel, self).__init__(**kwargs)

        # Each feature vector has to have an entry here
        # Format (feature_name, feature_vector_size)
        self.features = [
            ("end_len", 1),
            ("pronouns", 6),
            ("inverse", 1),
        ]

        self.placeholders = {
                f[0]: tf.placeholder(dtype=tf.float32,
                    shape=(None, f[1]), name="feature_{}".format(f[0]))
                for f in self.features
            }

        # Get ids for all special words
        self.male_ids = [word_to_id[w] for w in self.pronouns_m]
        self.female_ids = [word_to_id[w] for w in self.pronouns_f]
        self.inverse_ids = [word_to_id[w] for w in self.inverse_words]

    def get_features(self):
        return list(self.placeholders.values())

    def get_feed_dict(self, contexts, endings, **kwargs):
        """
        Compute feature vectors of shape (batch_size, k), where k can be any size.
        Assign each feature vector to corresponding entry in the features dict.
        Feature name should correspond to an entry in self.feature_names
        """
        features = {}

        # Ending Length
        end_len = np.array([len(e[0]) for e in endings])
        features["end_len"] = np.expand_dims(end_len, axis=1)

        # Male and female words
        def words_in_sentences(sentences, word_ids):
            for sent in sentences:
                if len(set(sent).intersection(word_ids)) > 0:
                    return 1

            return 0

        pronoun_features = []
        for cont, end in zip(contexts, endings):
            male_cont = words_in_sentences(cont, self.male_ids)
            female_cont = words_in_sentences(cont, self.female_ids)
            male_end = words_in_sentences(end, self.male_ids)
            female_end = words_in_sentences(end, self.female_ids)
            pronoun_features.append([
                male_cont,
                female_cont,
                male_end,
                female_end,
                int(male_cont != male_end), # Detect missmatches
                int(female_cont != female_end),
            ])

        features["pronouns"] = np.array(pronoun_features, dtype=float)

        # Inverse words
        inverse_found = np.array([words_in_sentences(e, self.inverse_ids)
            for e in endings], dtype=float)
        features["inverse"] = np.expand_dims(inverse_found, axis=1)

        # Parse and return feed_dict
        feed_dict = {self.placeholders[k]: features[k] for k in features}
        return feed_dict

    def predict(self, contexts, endings, **kwargs):
        raise NotImplementedError("Should not be used for prediction")

    def train_step(self, context_batch, end_batch, single=True, labels=None, **kwargs):
        raise NotImplementedError("Model can not be trained")
