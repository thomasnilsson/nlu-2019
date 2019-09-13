"""
NOTE:
Code adopted from NLU project 1
"""
from gensim import models
import numpy as np
import pandas as pd
from text_to_uri import standardized_uri

def numberbatch_embedding(vocab, path, dim_embedding, vocab_size, verbose=False):
    print("Loading numberbatch embeddings from %s" % path)
    embeddings = pd.read_hdf(path, 'mat', encoding='utf-8')
    matches = 0

    external_embedding = np.zeros(shape=(vocab_size, dim_embedding), dtype=np.float32)

    for tok, idx in vocab.items():
      uri = standardized_uri('en', tok)
      try:
        vec = embeddings.loc[uri]
        matches += 1
      except KeyError:
        vec = pd.Series(index=embeddings.columns).fillna(0)
      external_embedding[idx] = vec

    if verbose:
      print("%d words out of %d could be loaded" % (matches, vocab_size))
    return external_embedding


def word2vec_embedding(vocab, path, dim_embedding, vocab_size, verbose=False):
    '''
    input:
        vocab                      A dictionary mapping token strings to vocabulary IDs
        path                       Path to embedding file
        dim_embedding, vocab_size  Dimensionality of the external embedding.
    output:
        embedding np.matrix, shape=[vocab_size, dim_embedding]
    '''

    print("Loading external embeddings from %s" % path)
    model = models.KeyedVectors.load_word2vec_format(path, binary=False)
    external_embedding = np.zeros(shape=(vocab_size, dim_embedding), dtype=np.float32)
    matches = 0

    for tok, idx in vocab.items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            if verbose:
              print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)

    if verbose:
      print("%d words out of %d could be loaded" % (matches, vocab_size))

    return external_embedding
