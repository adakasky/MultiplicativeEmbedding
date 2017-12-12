import os
import codecs
import pickle
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
from data import build_vocab, load_vocab, preprocess_snli_jsonl, load_snli, get_padded_batch

tf.set_random_seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def build_graph(vocab_size=30000, embedding_size=100, state_size=100, batch_size=256, inverse_drop_rate=0.9,
                learning_rate=1e-4, num_classes=3):
    initializer = tf.contrib.layers.xavier_initializer()
    
    # Placeholders
    X1 = tf.placeholder(tf.int32, [batch_size, None])
    X2 = tf.placeholder(tf.int32, [batch_size, None])
    X1_embed_cont = tf.placeholder(tf.int32, [None, None])
    X2_embed_cont = tf.placeholder(tf.int32, [None, None])
    X1_embed_ent = tf.placeholder(tf.int32, [None, None])
    X2_embed_ent = tf.placeholder(tf.int32, [None, None])
    l1 = tf.placeholder(tf.int32, [batch_size])
    l2 = tf.placeholder(tf.int32, [batch_size])
    Y = tf.placeholder(tf.int32, [batch_size])
    Y_embed_cont = tf.placeholder(tf.float32, [None, embedding_size])
    Y_embed_ent = tf.placeholder(tf.float32, [None, embedding_size])
    
    # Embeddings
    embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], 1.0, -1.0), name="embeddings")
    sent1 = tf.nn.embedding_lookup(embeddings, X1)
    sent2 = tf.nn.embedding_lookup(embeddings, X2)
    sent1_embed_cont = tf.nn.embedding_lookup(embeddings, X1_embed_cont)
    sent2_embed_cont = tf.nn.embedding_lookup(embeddings, X2_embed_cont)
    sent1_embed_ent = tf.nn.embedding_lookup(embeddings, X1_embed_ent)
    sent2_embed_ent = tf.nn.embedding_lookup(embeddings, X2_embed_ent)
    
    # LSTM
    lstm_cell = {}
    initial_state = {}
    for direction in ["forward", "backward"]:
        with tf.variable_scope(direction):
            # LSTM cell
            lstm_cell[direction] = tf.contrib.rnn.LSTMCell(state_size, forget_bias=1.0, initializer=initializer,
                                                           state_is_tuple=True)
            # initial state: http://stackoverflow.com/questions/38441589/tensorflow-rnn-initial-state
            initial_cell_state = tf.get_variable("initial_cell_state", shape=[1, state_size],
                                                 dtype=tf.float32, initializer=initializer)
            initial_output_state = tf.get_variable("initial_output_state", shape=[1, state_size],
                                                   dtype=tf.float32, initializer=initializer)
            c_states = tf.tile(initial_cell_state, tf.stack([batch_size, 1]))
            h_states = tf.tile(initial_output_state, tf.stack([batch_size, 1]))
            initial_state[direction] = tf.contrib.rnn.LSTMStateTuple(c_states, h_states)
    
    with tf.variable_scope('lstm') as scope:
        lstm_outputs1, lstm_states1 = tf.nn.bidirectional_dynamic_rnn(lstm_cell["forward"],
                                                                      lstm_cell["backward"],
                                                                      sent1,
                                                                      dtype=tf.float32,
                                                                      sequence_length=l1,
                                                                      initial_state_fw=initial_state["forward"],
                                                                      initial_state_bw=initial_state["backward"])
        scope.reuse_variables()
        lstm_outputs2, lstm_states2 = tf.nn.bidirectional_dynamic_rnn(lstm_cell["forward"],
                                                                      lstm_cell["backward"],
                                                                      sent2,
                                                                      dtype=tf.float32,
                                                                      sequence_length=l2,
                                                                      initial_state_fw=initial_state["forward"],
                                                                      initial_state_bw=initial_state["backward"])
    
    fw_output1, bw_output1 = lstm_outputs1
    output1 = tf.concat([fw_output1, fw_output1], axis=2, name='output1')
    fw_output2, bw_output2 = lstm_outputs2
    output2 = tf.concat([fw_output2, fw_output2], axis=2, name='output2')
    
    output1 = tf.nn.dropout(output1, inverse_drop_rate)
    output2 = tf.nn.dropout(output2, inverse_drop_rate)
    
    idx1 = tf.range(batch_size) * tf.shape(output1)[1] + (l1 - 1)
    output1 = tf.gather(tf.reshape(output1, [-1, state_size * 2]), idx1)
    idx2 = tf.range(batch_size) * tf.shape(output2)[1] + (l2 - 1)
    output2 = tf.gather(tf.reshape(output2, [-1, state_size * 2]), idx2)
    
    addition = output1 + output2
    multiplication = output1 * output2
    abs_diff = tf.abs(output1 - output2)
    
    interact = tf.concat([addition, multiplication, abs_diff], axis=1)
    
    # Softmax
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size * 6, num_classes], initializer=initializer)
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    
    logits = tf.matmul(interact, W) + b
    preds = tf.argmax(tf.nn.softmax(logits), axis=1)
    correct = tf.equal(tf.cast(preds, tf.int32), Y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    classification_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    
    sent_cont_score = tf.add(tf.reduce_prod(sent1_embed_cont, axis=1), tf.reduce_prod(sent2_embed_cont, axis=1))
    embed_cont_loss = tf.reduce_sum((sent_cont_score - Y_embed_cont) ** 2)
    sent_ent_score = tf.subtract(tf.reduce_prod(sent1_embed_ent, axis=1), tf.reduce_prod(sent2_embed_ent, axis=1))
    embed_ent_loss = tf.reduce_sum((sent_ent_score - Y_embed_ent) ** 2)
    embed_loss = embed_cont_loss + embed_ent_loss
    
    loss = classification_loss + embed_loss
    trainer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    return {'X1': X1, 'X2': X2, 'Y': Y, 'l1': l1, 'l2': l2,
            'X1_embed_cont': X1_embed_cont, 'X2_embed_cont': X2_embed_cont, 'Y_embed_cont': Y_embed_cont,
            'X1_embed_ent': X1_embed_ent, 'X2_embed_ent': X2_embed_ent, 'Y_embed_ent': Y_embed_ent,
            'embeddings': embeddings, 'embed_loss': embed_loss, 'loss': loss,
            'trainer': trainer, 'preds': preds, 'accuracy': accuracy}


def fit(data, graph, batch_size=256, embedding_size=100, num_epochs=10, model_dir='../models/'):
    saver = tf.train.Saver()
    embedding_file = model_dir + 'embeddings-%d.pkl' % embedding_size
    model_file = model_dir + 'model-%d.ckpt' % embedding_size
    loss_file = model_dir + 'losses-%d.pkl' % embedding_size
    acc_file = model_dir + 'accuracies-%d.pkl' % embedding_size
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        num_train = len(data['train']['Y'])
        iter_per_epoch = np.ceil(num_train / batch_size)
        
        current_epoch = 0
        loss_train = np.inf
        loss_dev = np.inf
        train_losses = []
        dev_losses = []
        train_embed_losses = []
        dev_embed_losses = []
        train_accuracies = []
        dev_accuracies = []
        
        while current_epoch < num_epochs:
            current_iter = 0
            while current_iter < iter_per_epoch:
                batch = get_padded_batch(data['train'], batch_size=batch_size, embedding_size=embedding_size)
                feed_dict_train = {graph['X1']: batch['X1'], graph['X2']: batch['X2'], graph['Y']: batch['Y'],
                                   graph['l1']: batch['l1'], graph['l2']: batch['l2'],
                                   graph['X1_embed_cont']: batch['X1_embed_cont'],
                                   graph['X2_embed_cont']: batch['X2_embed_cont'],
                                   graph['Y_embed_cont']: batch['Y_embed_cont'],
                                   graph['X1_embed_ent']: batch['X1_embed_ent'],
                                   graph['X2_embed_ent']: batch['X2_embed_ent'],
                                   graph['Y_embed_ent']: batch['Y_embed_ent']}
                dev_batch = get_padded_batch(data['dev'], batch_size=batch_size, embedding_size=embedding_size)
                feed_dict_dev = {graph['X1']: dev_batch['X1'], graph['X2']: dev_batch['X2'], graph['Y']: dev_batch['Y'],
                                 graph['l1']: dev_batch['l1'], graph['l2']: dev_batch['l2'],
                                 graph['X1_embed_cont']: dev_batch['X1_embed_cont'],
                                 graph['X2_embed_cont']: dev_batch['X2_embed_cont'],
                                 graph['Y_embed_cont']: dev_batch['Y_embed_cont'],
                                 graph['X1_embed_ent']: dev_batch['X1_embed_ent'],
                                 graph['X2_embed_ent']: dev_batch['X2_embed_ent'],
                                 graph['Y_embed_ent']: dev_batch['Y_embed_ent']}
                loss_train, embed_loss_train, acc_train, _ = \
                    sess.run([graph['loss'], graph['embed_loss'], graph['accuracy'], graph['trainer']],
                             feed_dict=feed_dict_train)
                loss_dev, embed_loss_dev, acc_dev = \
                    sess.run([graph['loss'], graph['embed_loss'], graph['accuracy']], feed_dict=feed_dict_dev)
                
                train_losses.append(loss_train)
                dev_losses.append(loss_dev)
                train_embed_losses.append(embed_loss_train)
                dev_embed_losses.append(embed_loss_dev)
                train_accuracies.append(acc_train)
                dev_accuracies.append(acc_dev)
                
                current_iter += 1
                total_iter = current_epoch * iter_per_epoch + current_iter
                if total_iter % 100 == 0:
                    print("Iteration %d train loss %f, train embed loss %f, train accuracy %f,"
                          " dev loss %f, dev embed loss %f, dev accuracy %f"
                          % (total_iter, loss_train, embed_loss_train, acc_train, loss_dev, embed_loss_dev, acc_dev))
            
            current_epoch += 1
            if current_epoch % 1 == 0:
                print("Epoch %d train loss %f, dev loss %f" % (current_epoch, loss_train, loss_dev))
        
        writer = codecs.open(embedding_file, 'wb')
        pickle.dump(sess.run(graph['embeddings']), writer)
        writer.close()
        
        losses = {'train_losses': np.array(train_losses), 'train_embed_losses': np.array(train_embed_losses),
                  'dev_losses': np.array(dev_losses), 'dev_embed_losses': np.array(dev_embed_losses)}
        accuracies = {'train_accuracies': np.array(train_accuracies), 'dev_accuracies': np.array(dev_accuracies)}
        
        writer = codecs.open(embedding_file, 'wb')
        pickle.dump(sess.run(graph['embeddings']), writer)
        writer.close()

        writer = codecs.open(loss_file, 'wb')
        pickle.dump(losses, writer)
        writer.close()

        writer = codecs.open(acc_file, 'wb')
        pickle.dump(accuracies, writer)
        writer.close()

        model_path = saver.save(sess, model_file)
    
    return train_losses, dev_losses, model_path


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
    
    data = {}
    for f in ['train', 'dev', 'test']:
        data[f] = load_snli(data_dir + 'snli_%s.pkl' % f)
    
    batch_size = 256
    embedding_size = 300
    state_size = 100
    inverse_drop_rate = 0.9
    learning_rate = 3e-3
    
    graph = build_graph(vocab_size=vocab_size, embedding_size=embedding_size, state_size=100, batch_size=batch_size,
                        inverse_drop_rate=inverse_drop_rate, learning_rate=learning_rate, num_classes=3)
    model_dir = '../models/'
    model_path = fit(data, graph, batch_size=batch_size, embedding_size=embedding_size, num_epochs=10,
                     model_dir=model_dir)
    
    # embeddings = pickle.load(open(embedding_file, 'rb'))
    # a = embeddings[vocab_idx['not']] * embeddings[vocab_idx['like']]
    # b = embeddings[vocab_idx['dislike']]
    # print(1 - cosine(a, b))
    # print(a * b)
