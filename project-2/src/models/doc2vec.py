from models.model import Model
import numpy as np
from scipy import spatial
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


class Doc2VecModel(Model):
    def __init__(self, train_dataset, vector_size=50, epochs=50, **kwargs):
        super(Doc2VecModel, self).__init__(train_dataset, **kwargs)

        self.epochs = epochs
        self.vector_size = vector_size
        self.train_dataset = train_dataset
        self.model = Doc2Vec(
            dm=0, vector_size=vector_size,
            window=5, negative=5, hs=0,
            min_count=2, workers=4)

    def train(self):
        print("Training Doc2Vec Model")
        self.model.build_vocab(self.train_dataset)

        self.model.train(
            self.train_dataset,
            total_examples=self.model.corpus_count,
            epochs=self.epochs)

    def calc_cosines(self, i):
        self.v_context = self.model.infer_vector(self.contexts[i])
        self.v_endings = [self.model.infer_vector(e) for e in self.endings[i]]
        v_cosines = [spatial.distance.cosine(self.v_context, v_end) for v_end in self.v_endings]
        return np.array(v_cosines)

    def predict(self, contexts, endings, **kwargs):
        self.contexts = contexts
        self.endings = endings
        n = len(contexts)
        self.cosines = np.array([self.calc_cosines(i) for i in range(n)])
        self.pred = np.array([cos.argmin() for cos in self.cosines])
        return self.pred

    def get_features(self):
        return self.cosines
