import pickle
import codecs
import numpy as np
from scipy.spatial.distance import cosine
from data import load_vocab


def embed_lookup(w):
    return embeddings[vocab_idx[w]]


def sim(a, b):
    return 1 - cosine(a, b)


def best_match(v, n=10, vocab_size=30000):
    sims = {}
    for i, e in enumerate(embeddings):
        sims[i] = sim(v, e)
    
    sims = sorted(sims, key=sims.get, reverse=True)
    for s in sims[:n]:
        print(vocab[s])
    
    
if __name__ == '__main__':
    embed_file = '../models/300-0.9/embeddings.pkl'
    vocab_file = '../data/snli_1.0/vocab.pkl'
    
    embeddings = pickle.load(codecs.open(embed_file, 'rb'))
    vocab, vocab_idx = load_vocab(vocab_file)
    
    a = embed_lookup('king') - embed_lookup('man') + embed_lookup('woman')
    best_match(a)
