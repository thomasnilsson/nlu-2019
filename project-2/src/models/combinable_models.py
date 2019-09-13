# All available combinable models
import data_loader as dl
import constants as C

from models.model import Model
from models.rnn_classifier import RNNClassifier
from models.sentiment_model import SentimentFeedForward
from models.doc2vec import Doc2VecModel
from models.feature_model import FeatureModel
from models.rnn_language_model import RNNLanguageModel
from models.doc2vec_for_combiner import doc2vecForCombiner

standard_info = {"loader": dl.load_preprocess,
                 "file_name": C.preprocessed_file}

MODELS = {
    "dummy": [Model, standard_info],
    "rnn": [RNNClassifier, standard_info],
    "doc2vec": [Doc2VecModel, {"loader": dl.load_doc2vec_data,
                               "file_name": C.doc2vec_file}],
    "lm": [RNNLanguageModel, standard_info],
    "sentiment": [SentimentFeedForward, {"loader": dl.load_sentiment,
                                         "file_name": C.sentiment_file}],
    "features": [FeatureModel, standard_info],
    "doc2vec_combiner": [doc2vecForCombiner, {"loader": dl.load_doc2vec_data,
                                              "file_name": C.doc2vec_file}],
}


def get_model_class(name):
    if not name in MODELS:
        raise Exception('Model "{}" not in model list'.format(name))

    return MODELS[name][0]


def get_model_info(name):
    if not name in MODELS:
        raise Exception('Model "{}" not in model list'.format(name))

    return MODELS[name][1]
