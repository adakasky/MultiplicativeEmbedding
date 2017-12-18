import pickle
import codecs
import numpy as np
import matplotlib.pyplot as plt
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


def plot_graph(arr, labels, xlabel='test-x', ylabel='test-y', title='title', x_min=0, x_max=10000, y_min=0, y_max=1,
               f='../doc/test.pdf'):
    x = range(x_max)
    
    plt.figure(figsize=(6, 4))
    for (a, label) in zip(arr, labels):
        plt.plot(x, a, '-', label=label, alpha=0.6)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(x_min, x_max)  # set x axis range
    plt.ylim(y_min, y_max)  # Set y axis range
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(f)
    plt.close()


if __name__ == '__main__':
    embed_file = '../models/300-0.9/embeddings.pkl'
    vocab_file = '../data/snli_1.0/vocab.pkl'
    
    embeddings = pickle.load(codecs.open(embed_file, 'rb'))
    vocab, vocab_idx = load_vocab(vocab_file)
    
    dpout_rates = np.arange(0.5, 1.0, 0.1)
    num_iter = 4000
    train_acc = [pickle.load(codecs.open('../models/300-%.1f/accuracies.pkl' % r, 'rb'))['train_accuracies'][:num_iter]
                 for r in dpout_rates]
    dev_acc = [pickle.load(codecs.open('../models/300-%.1f/accuracies.pkl' % r, 'rb'))['dev_accuracies'][:num_iter]
               for r in dpout_rates]
    
    train_loss = [pickle.load(codecs.open('../models/300-%.1f/losses.pkl' % r, 'rb'))['train_losses'][:num_iter]
                  for r in dpout_rates]
    dev_loss = [pickle.load(codecs.open('../models/300-%.1f/losses.pkl' % r, 'rb'))['dev_losses'][:num_iter]
                for r in dpout_rates]
    
    plot_graph(train_acc, dpout_rates, 'Iterations', 'Accuracy', 'Training Accuracy vs. Iteration', 0, num_iter, 0, 1,
               '../doc/train_acc.pdf')
    plot_graph(dev_acc, dpout_rates, 'Iterations', 'Accuracy', 'Validation Accuracy vs. Iteration', 0, num_iter, 0, 1,
               '../doc/dev_acc.pdf')
    plot_graph(train_loss, dpout_rates, 'Iterations', 'Loss', 'Training Loss vs. Iteration', 0, num_iter, 0, 5,
               '../doc/train_loss.pdf')
    plot_graph(dev_loss, dpout_rates, 'Iterations', 'Loss', 'Validation Loss vs. Iteration', 0, num_iter, 0, 5,
               '../doc/dev_loss.pdf')
    
    print(np.max(dev_acc[4]))
