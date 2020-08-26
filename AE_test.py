import os
import random
import argparse
import tensorflow as tf
import numpy as np
from AE import AE
from nltk.translate.bleu_score import corpus_bleu


def load_dic(datapath):
    word2index = {'<pad>': 0, '<eos>': 1}
    index2word = {0: '<pad>', 1: '<eos>'}
    index = 2
    with open(os.path.join(datapath, 'vocab.txt'), 'r') as f:
        for vocab in f.readlines():
            vocab = vocab.rstrip()
            word2index[vocab] = index
            index2word[index] = vocab
            index = index + 1
    return word2index, index2word


def load_model(emb_dim, rnn_dim, z_dim, vocab_size, lr, model_path):
    ae = AE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=vocab_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=ae)
    ckpt.restore(tf.train.latest_checkpoint(model_path)).expect_partial()
    return ae


def reconstruction(model_path):
    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[1]
        s = s.split(',')

    emb_dim = int(s[0].split()[-1])
    rnn_dim = int(s[1].split()[-1])
    z_dim = int(s[2].split()[-1])
    lr = float(s[5].split()[-1])
    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
    vocab_size = int(s[-1].split()[-1])

    word2index, index2word = load_dic(datapath)
    ae = load_model(emb_dim, rnn_dim, z_dim, vocab_size, lr, model_path)

    maxlen = 0
    sentences = []
    with open(os.path.join(datapath, 'test.unk.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip() + ' <eos>'
            sentence = sentence.split()
            for i in range(len(sentence)):
                sentence[i] = word2index[sentence[i]]
            if len(sentence) > maxlen:
                maxlen = len(sentence)
            sentences.append(sentence)

    x_test = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=maxlen, padding='post', truncating='post')
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(512)

    reconstruction_file = os.path.join(model_path, 'mean.txt')

    print("mean vector reconstruction")
    f = open(reconstruction_file, 'w')
    for x_batch_test in test_dataset:
        enc_embeddings = ae.embeddings(x_batch_test)
        z = ae.encoder(x_batch_test, enc_embeddings)
        input = tf.constant(word2index['<eos>'], shape=(x_batch_test.shape[0], 1), dtype=tf.int64)
        state = None
        output = input
        for _ in range(maxlen):
            dec_embeddings = ae.embeddings(input)
            new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
            dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
            out, h, c = ae.decoder.rnn(dec_input, initial_state=state)
            pred = ae.decoder.vocab_prob(out)
            pred = tf.keras.backend.argmax(pred, axis=-1)
            input = pred
            state = [h, c]
            output = tf.keras.backend.concatenate([output, pred], axis=1)

        output = output[:, 1:]
        output = output.numpy().tolist()
        for element in output:
            if 1 in element:
                element = element[:element.index(1)]
            element = [index2word[i] for i in element]
            f.write(' '.join(element) + '\n')
    f.close()
    print("reconstruction file at :{}.".format(reconstruction_file))


def bleu(candidate_path, reference_path):
    references = []
    with open(reference_path, 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip().split()
            references.append([sentence])

    candidates = []
    with open(candidate_path, 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip().split()
            candidates.append(sentence)

    bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0))
    bleu4 = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
    print('BLEU-1:{:f}, BLEU-2:{:f}, BLEU-4:{:f}'.format(bleu1*100, bleu2*100, bleu4*100))


def homotopy_evaluation(model_path, seed=0):
    tf.random.set_seed(seed)
    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[1]
        s = s.split(',')

    emb_dim = int(s[0].split()[-1])
    rnn_dim = int(s[1].split()[-1])
    z_dim = int(s[2].split()[-1])
    lr = float(s[5].split()[-1])
    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
    vocab_size = int(s[-1].split()[-1])

    word2index, index2word = load_dic(datapath)
    ae = load_model(emb_dim, rnn_dim, z_dim, vocab_size, lr, model_path)

    maxlen = 0
    with open(os.path.join(datapath, 'test.unk.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip() + ' <eos>'
            sentence = sentence.split()
            if len(sentence) > maxlen:
                maxlen = len(sentence)

    z1 = tf.keras.backend.random_normal(shape=(1, z_dim))
    z2 = tf.keras.backend.random_normal(shape=(1, z_dim))
    print('normal homotopy:')
    for i in range(0, 6):
        print(i + 1, end='. ')
        z = (1 - 0.2 * i) * z1 + 0.2 * i * z2
        input = tf.constant(word2index['<eos>'], shape=(1, 1))
        state = None
        output = []
        for _ in range(0, maxlen):
            dec_embeddings = ae.embeddings(input)
            new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
            dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
            out, h, c = ae.decoder.rnn(dec_input, initial_state=state)
            pred = ae.decoder.vocab_prob(out)
            pred = tf.keras.backend.argmax(pred, axis=-1)
            input = pred
            state = [h, c]
            output.append(index2word[int(pred)])
            if output[-1] == '<eos>':
                break
        for j in range(0, len(output) - 2):
            print(output[j], end=' ')
        print(output[-2], end='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model evaluation')
    parser.add_argument('-tm', '--test_mode', default=0, type=int, help='test mode: 0 will reconstruct test set, 1 will '
                                                                        'do homotopy evaluation.')
    parser.add_argument('-m', '--mpath', default='CBT/CBT_Z_64_AE', help='path of model')

    args = parser.parse_args()

    mode = args.test_mode
    model_path = os.path.join(os.path.join(os.getcwd(), 'model'), args.mpath)

    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[1]
        s = s.split(',')

    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))

    if mode == 0:
        reconstruction(model_path)
        reference_path = os.path.join(datapath, 'test.unk.txt')
        print('mean reconstruction file bleu scores (reference original file)')
        candidate_path = os.path.join(model_path, 'mean.txt')
        bleu(candidate_path, reference_path)
    elif mode == 1:
        homotopy_evaluation(model_path)
    else:
        print("wrong mode, please type basic_evaluation.py -h for help")