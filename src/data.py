from __future__ import division, print_function
import os
import json
import codecs
import pickle
import string
import numpy as np
from operator import itemgetter
from nltk.tokenize import word_tokenize as wt
from nltk.corpus import stopwords
from collections import defaultdict

np.random.seed(0)
stop_words = stopwords.words('english')


def preprocess_snli_jsonl(file_path, vocab_idx, out_file, vocab_size=30000):
    X1 = []
    X2 = []
    l1 = []
    l2 = []
    Y = []
    labels = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
    with codecs.open(file_path, 'r', 'utf-8') as f:
        for line in f:
            line = json.loads(line)
            if line['gold_label'] not in labels:
                continue
            sentence1 = [w.lower() for w in wt(line['sentence1'])]
            s1 = []
            for w in sentence1:
                s1.append(vocab_idx[w] if w in vocab_idx else vocab_size - 1)
            sentence2 = [w.lower() for w in wt(line['sentence2'])]
            s2 = []
            for w in sentence2:
                s2.append(vocab_idx[w] if w in vocab_idx else vocab_size - 1)
            
            X1.append(np.array(s1))
            X2.append(np.array(s2))
            l1.append(len(s1))
            l2.append(len(s2))
            Y.append(labels[line['gold_label']])
    
    writer = codecs.open(out_file, 'wb')
    data = {'X1': np.array(X1), 'X2': np.array(X2), 'l1': np.array(l1), 'l2': np.array(l2), 'Y': np.array(Y)}
    pickle.dump(data, writer)
    writer.close()


def build_vocab(train_file, vocab_file):
    vocab = defaultdict(int)
    with codecs.open(train_file, 'r', 'utf-8') as f:
        for line in f:
            d = json.loads(line)
            sentence1 = [w.lower() for w in wt(d['sentence1'])]
            sentence2 = [w.lower() for w in wt(d['sentence2'])]
            for word in sentence1:
                vocab[word] += 1
            for word in sentence2:
                vocab[word] += 1
    
    writer = codecs.open(vocab_file, 'wb')
    vocab = sorted(vocab.items(), key=itemgetter(1), reverse=True)
    pickle.dump(vocab, writer)
    writer.close()


def load_vocab(vocab_file, vocab_size=30000):
    f = open(vocab_file, 'rb')
    vocab = pickle.load(f)
    vocab = [v[0] for v in vocab if v[0] not in string.punctuation]
    # 'PAD' for padding and 'UNK' for words outside vocabulary, i.e. ones with either high or low frequencies.
    vocab = ['PAD'] + vocab[:vocab_size - 2] + ['UNK']
    
    vocab_idx = {}
    for i, v in enumerate(vocab):
        vocab_idx[v] = i
    
    f.close()
    return vocab, vocab_idx


def load_snli(file_path):
    f = open(file_path, 'rb')
    data = pickle.load(f)
    f.close()
    
    return data


def get_padded_batch(data, batch_size=256, embedding_size=100, random=True, start=0):
    n = len(data['Y'])
    idx = None
    if random:
        idx = np.random.choice(n, batch_size, replace=True)
    else:
        idx = np.arange(start, batch_size + start)
    X1_temp = data['X1'][idx]
    X2_temp = data['X2'][idx]
    l1_batch = data['l1'][idx]
    l2_batch = data['l2'][idx]
    Y_batch = data['Y'][idx]
    max_length = max(np.max(l1_batch), np.max(l2_batch))
    X1_batch = np.zeros((batch_size, max_length), dtype=np.int32)
    X2_batch = np.zeros((batch_size, max_length), dtype=np.int32)
    
    for i, (x1, x2) in enumerate(zip(X1_batch, X2_batch)):
        x1[:l1_batch[i]] = X1_temp[i]
        x2[:l2_batch[i]] = X2_temp[i]

    idx = np.where(Y_batch == 2)
    X1_embed_cont = X1_batch[idx]
    X2_embed_cont = X2_batch[idx]
    Y_embed_temp = Y_batch[idx]
    num_cont_sent = len(Y_embed_temp)
    Y_embed_cont = np.zeros((num_cont_sent, embedding_size), dtype=np.float32)

    idx = np.where(Y_batch == 1)
    X1_embed_ent = X1_batch[idx]
    X2_embed_ent = X2_batch[idx]
    Y_embed_temp = Y_batch[idx]
    num_ent_sent = len(Y_embed_temp)
    Y_embed_ent = np.zeros((num_ent_sent, embedding_size), dtype=np.float32)
    
    return {'X1': X1_batch,
            'X2': X2_batch,
            'l1': l1_batch,
            'l2': l2_batch,
            'Y': Y_batch,
            'X1_embed_cont': X1_embed_cont,
            'X2_embed_cont': X2_embed_cont,
            'Y_embed_cont': Y_embed_cont,
            'X1_embed_ent': X1_embed_ent,
            'X2_embed_ent': X2_embed_ent,
            'Y_embed_ent': Y_embed_ent}


if __name__ == '__main__':
    data_dir = '../data/snli_1.0/'
    files = [data_dir + s for s in ['snli_1.0_train.jsonl', 'snli_1.0_dev.jsonl', 'snli_1.0_test.jsonl']]
    
    vocab_file = data_dir + 'vocab.pkl'
    # including place for padding and UNK
    vocab_size = 30000
    vocab = []
    if not os.path.exists(vocab_file):
        build_vocab(files[0], vocab_file)
    vocab, vocab_idx = load_vocab(vocab_file)
    
    snli_train_file = data_dir + 'snli_train.pkl'
    if not os.path.exists(snli_train_file):
        preprocess_snli_jsonl(files[0], vocab_idx, snli_train_file, vocab_size)
    
    data = load_snli(snli_train_file)
    batch = get_padded_batch(data, batch_size=256, embedding_size=100)
