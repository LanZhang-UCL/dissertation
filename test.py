import os
import random
import argparse
import tensorflow as tf
import numpy as np
from VAE import VAE
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
    vae = VAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=vocab_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=vae)
    ckpt.restore(tf.train.latest_checkpoint(model_path)).expect_partial()
    return vae


def noise(vae, datapath, z_dim, word2index, batch_size, C):
    sentences = []
    maxlen = 0
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

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.shuffle(len(x_test)).batch(batch_size)

    val_maximum = tf.constant(0.0, shape=(z_dim,))
    for x_batch_test in test_dataset:
        enc_embeddings = vae.embeddings(x_batch_test)
        output = vae.encoder.rnn(enc_embeddings)
        mask = tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x_batch_test, 1))
        mask = tf.keras.backend.expand_dims(mask)
        mask = tf.keras.backend.repeat_elements(mask, output.shape[2], axis=2)
        output = tf.keras.backend.sum(output * mask, axis=1)
        logvar = vae.encoder.logvar_layer(output)
        value = tf.keras.backend.exp(logvar) - logvar - 1
        val_maximum = tf.keras.backend.maximum(val_maximum, tf.keras.backend.max(value, axis=0))

    threshold = 0.05*tf.keras.backend.log(C)
    return tf.keras.backend.cast_to_floatx(tf.keras.backend.less_equal(val_maximum, threshold))


def active_units(vae, datapath, z_dim, word2index, batch_size):
    sentences = []
    maxlen = 0
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

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.shuffle(len(x_test)).batch(batch_size)

    total_mean = tf.constant(0.0, shape=(0, z_dim))
    for step, x_batch_test in enumerate(test_dataset):
        enc_embeddings = vae.embeddings(x_batch_test)
        output = vae.encoder.rnn(enc_embeddings)
        mask = tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x_batch_test, 1))
        mask = tf.keras.backend.expand_dims(mask)
        mask = tf.keras.backend.repeat_elements(mask, output.shape[2], axis=2)
        output = tf.keras.backend.sum(output * mask, axis=1)
        mean = vae.encoder.mean_layer(output)
        total_mean = tf.keras.backend.concatenate([mean, total_mean], axis=0)

    total_mean = total_mean.numpy()
    cov = np.cov(total_mean, rowvar=False)
    au = []
    for i in range(0, z_dim):
        if cov[i][i] > 0.01:
            au.append(1.0)
        else:
            au.append(0.0)
    return tf.constant(au)


def noise_quantitative(model_path):
    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[1]
        s = s.split(',')

    emb_dim = int(s[0].split()[-1])
    rnn_dim = int(s[1].split()[-1])
    z_dim = int(s[2].split()[-1])
    batch_size = int(s[3].split()[-1])
    lr = float(s[5].split()[-1])
    C = float(s[-3].split()[-1])
    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
    vocab_size = int(s[-1].split()[-1])

    word2index, index2word = load_dic(datapath)
    vae = load_model(emb_dim, rnn_dim, z_dim, vocab_size, lr, model_path)

    noise_dimension = noise(vae, datapath, z_dim, word2index, batch_size, C)

    noise_dimension = noise_dimension.numpy().tolist()
    s = []
    n = []
    for i in range(0, len(noise_dimension)):
        if noise_dimension[i] == 1:
            n.append(i+1)
        else:
            s.append(i+1)
    print('{} signal dimensions:{}'.format(len(s), s))
    print('{} noise dimensions:{}'.format(len(n), n))

    au = active_units(vae, datapath, z_dim, word2index, batch_size)

    au = au.numpy().tolist()
    s = []
    n = []
    for i in range(0, len(au)):
        if au[i] == 1:
            s.append(i + 1)
        else:
            n.append(i + 1)
    print('{} active units:{}'.format(len(s), s))
    print('{} inactive units:{}'.format(len(n), n))


def mean_vector_reconstruct(vae, test_dataset, maxlen, reconstruction_file, word2index, index2word):
    print("mean vector reconstruction")
    f = open(reconstruction_file, 'w')
    for x_batch_test in test_dataset:
        enc_embeddings = vae.embeddings(x_batch_test)
        output = vae.encoder.rnn(enc_embeddings)
        mask = tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x_batch_test, 1))
        mask = tf.keras.backend.expand_dims(mask)
        mask = tf.keras.backend.repeat_elements(mask, output.shape[2], axis=2)
        output = tf.keras.backend.sum(output * mask, axis=1)
        mean = vae.encoder.mean_layer(output)
        z = mean
        input = tf.constant(word2index['<eos>'], shape=(x_batch_test.shape[0], 1), dtype=tf.int64)
        state = None
        output = input
        for _ in range(maxlen):
            dec_embeddings = vae.embeddings(input)
            new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
            dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
            out, h, c = vae.decoder.rnn(dec_input, initial_state=state)
            pred = vae.decoder.vocab_prob(out)
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


def signal_vector_reconstruct(vae, noise_dimension, test_dataset, maxlen, reconstruction_file, word2index, index2word):
    print("signal vector reconstruction")
    f = open(reconstruction_file, 'w')
    for x_batch_test in test_dataset:
        enc_embeddings = vae.embeddings(x_batch_test)
        output = vae.encoder.rnn(enc_embeddings)
        mask = tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x_batch_test, 1))
        mask = tf.keras.backend.expand_dims(mask)
        mask = tf.keras.backend.repeat_elements(mask, output.shape[2], axis=2)
        output = tf.keras.backend.sum(output * mask, axis=1)
        mean = vae.encoder.mean_layer(output)
        signal_vector = (1-noise_dimension)*mean
        z = signal_vector
        input = tf.constant(word2index['<eos>'], shape=(x_batch_test.shape[0], 1), dtype=tf.int64)
        state = None
        output = input
        for _ in range(maxlen):
            dec_embeddings = vae.embeddings(input)
            new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
            dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
            out, h, c = vae.decoder.rnn(dec_input, initial_state=state)
            pred = vae.decoder.vocab_prob(out)
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


def random_vector_reconstruction(vae, test_dataset, maxlen, reconstruction_file, word2index, index2word):
    print("random vector reconstruction")
    f = open(reconstruction_file, 'w')
    for x_batch_test in test_dataset:
        enc_embeddings = vae.embeddings(x_batch_test)
        output = vae.encoder.rnn(enc_embeddings)
        mask = tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x_batch_test, 1))
        mask = tf.keras.backend.expand_dims(mask)
        mask = tf.keras.backend.repeat_elements(mask, output.shape[2], axis=2)
        output = tf.keras.backend.sum(output * mask, axis=1)
        mean = vae.encoder.mean_layer(output)
        z = tf.keras.backend.random_normal(shape=mean.shape)
        input = tf.constant(word2index['<eos>'], shape=(x_batch_test.shape[0], 1), dtype=tf.int64)
        state = None
        output = input
        for _ in range(maxlen):
            dec_embeddings = vae.embeddings(input)
            new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
            dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
            out, h, c = vae.decoder.rnn(dec_input, initial_state=state)
            pred = vae.decoder.vocab_prob(out)
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


def reconstruction(model_path):
    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[1]
        s = s.split(',')

    emb_dim = int(s[0].split()[-1])
    rnn_dim = int(s[1].split()[-1])
    z_dim = int(s[2].split()[-1])
    batch_size = int(s[3].split()[-1])
    lr = float(s[5].split()[-1])
    C = float(s[-3].split()[-1])
    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
    vocab_size = int(s[-1].split()[-1])

    word2index, index2word = load_dic(datapath)
    vae = load_model(emb_dim, rnn_dim, z_dim, vocab_size, lr, model_path)

    noise_dimension = noise(vae, datapath, z_dim, word2index, batch_size, C)

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

    mean_file = os.path.join(model_path, 'mean.txt')
    signal_file = os.path.join(model_path, 'signal.txt')
    random_file = os.path.join(model_path, 'random.txt')

    mean_vector_reconstruct(vae, test_dataset, maxlen, mean_file, word2index, index2word)
    signal_vector_reconstruct(vae, noise_dimension, test_dataset, maxlen, signal_file, word2index, index2word)
    random_vector_reconstruction(vae, test_dataset, maxlen, random_file, word2index, index2word)


def loss_evaluation(model_path):
    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[1]
        s = s.split(',')

    emb_dim = int(s[0].split()[-1])
    rnn_dim = int(s[1].split()[-1])
    z_dim = int(s[2].split()[-1])
    batch_size = int(s[3].split()[-1])
    lr = float(s[5].split()[-1])
    C = float(s[-3].split()[-1])
    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
    vocab_size = int(s[-1].split()[-1])

    word2index, index2word = load_dic(datapath)
    vae = load_model(emb_dim, rnn_dim, z_dim, vocab_size, lr, model_path)

    noise_dimension = noise(vae, datapath, z_dim, word2index, batch_size, C)

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
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size)

    total_rec_loss = 0
    for step, x_batch_test in enumerate(test_dataset):
        enc_embeddings = vae.embeddings(x_batch_test)
        z, kl_loss = vae.encoder(x_batch_test, enc_embeddings)
        z = z * (1 - noise_dimension)
        y = tf.keras.backend.concatenate([tf.constant(1, shape=(x_batch_test.shape[0], 1)), x_batch_test[:, :-1]], axis=-1)
        dec_embeddings = vae.embeddings(y)
        _, rec_loss = vae.decoder(x_batch_test, dec_embeddings, z)
        total_rec_loss = total_rec_loss + tf.keras.backend.mean(rec_loss)

    test_rec_loss = total_rec_loss / (step + 1)
    print("rec_loss:{:.4f}".format(test_rec_loss))


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


def drop1_reconstruction(vae, noise_dimension, mean_vector, word2index, index2word, maxlen):
    print("drop1 vector reconstruction")
    sd = []
    signal_dimension = (1 - noise_dimension).numpy().tolist()
    for i in range(0, len(signal_dimension)):
        if signal_dimension[i] == 1:
            sd.append(i)
    for dim in sd:
        print("drop {:d} dimension, {:.3f}".format(dim + 1, mean_vector[0, dim]))
        drop1 = [1 for _ in range(0, noise_dimension.shape[0])]
        drop1[dim] = 0
        drop1 = tf.constant(drop1, shape=(1, noise_dimension.shape[0]), dtype=mean_vector.dtype)
        drop1_vector = mean_vector * drop1
        z = drop1_vector
        input = tf.constant(word2index['<eos>'], shape=(1, 1))
        state = None
        output = []
        for _ in range(0, maxlen):
            dec_embeddings = vae.embeddings(input)
            new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
            dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
            out, h, c = vae.decoder.rnn(dec_input, initial_state=state)
            pred = vae.decoder.vocab_prob(out)
            pred = tf.keras.backend.argmax(pred, axis=-1)
            input = pred
            state = [h, c]
            output.append(index2word[int(pred)])
            if output[-1] == '<eos>':
                break
        for j in range(0, len(output) - 2):
            print(output[j], end=' ')
        print(output[-2], end='\n')


def sample_sentence(model_path, seed=0):
    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[1]
        s = s.split(',')

    emb_dim = int(s[0].split()[-1])
    rnn_dim = int(s[1].split()[-1])
    z_dim = int(s[2].split()[-1])
    batch_size = int(s[3].split()[-1])
    lr = float(s[5].split()[-1])
    C = float(s[-3].split()[-1])
    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
    vocab_size = int(s[-1].split()[-1])

    word2index, index2word = load_dic(datapath)
    vae = load_model(emb_dim, rnn_dim, z_dim, vocab_size, lr, model_path)

    noise_dimension = noise(vae, datapath, z_dim, word2index, batch_size, C)
    au = active_units(vae, datapath, z_dim, word2index, batch_size)

    sentences = []
    maxlen = 0
    with open(os.path.join(datapath, 'test.unk.txt'), 'r') as f:
        for sentence in f.readlines():
            sentences.append(sentence)
            sentence = sentence.rstrip() + ' <eos>'
            sentence = sentence.split()
            if len(sentence) > maxlen:
                maxlen = len(sentence)

    random.seed(seed)
    sample_index = random.sample(range(len(sentences)), 4)
    for index in sample_index:
        print("original sentence")
        print(sentences[index])
        s = sentences[index] + ' <eos>'
        s = s.split()
        for i in range(0, len(s)):
            s[i] = float(word2index[s[i]])
        x = tf.constant(s, shape=(1, len(s)))
        enc_embeddings = vae.embeddings(x)
        output = vae.encoder.rnn(enc_embeddings)
        mask = tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x, 1))
        mask = tf.keras.backend.expand_dims(mask)
        mask = tf.keras.backend.repeat_elements(mask, output.shape[2], axis=2)
        output = tf.keras.backend.sum(output * mask, axis=1)
        mean = vae.encoder.mean_layer(output)
        signal_vector = (1 - noise_dimension) * mean
        au_vector = au * mean

        print("mean vector reconstruction")
        z = mean
        input = tf.constant(word2index['<eos>'], shape=(1, 1))
        state = None
        output = []
        for _ in range(0, maxlen):
            dec_embeddings = vae.embeddings(input)
            new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
            dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
            out, h, c = vae.decoder.rnn(dec_input, initial_state=state)
            pred = vae.decoder.vocab_prob(out)
            pred = tf.keras.backend.argmax(pred, axis=-1)
            input = pred
            state = [h, c]
            output.append(index2word[int(pred)])
            if output[-1] == '<eos>':
                break
        for j in range(0, len(output) - 2):
            print(output[j], end=' ')
        print(output[-2], end='\n')
        print()

        print("active units reconstruction")
        z = au_vector
        input = tf.constant(word2index['<eos>'], shape=(1, 1))
        state = None
        output = []
        for _ in range(0, maxlen):
            dec_embeddings = vae.embeddings(input)
            new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
            dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
            out, h, c = vae.decoder.rnn(dec_input, initial_state=state)
            pred = vae.decoder.vocab_prob(out)
            pred = tf.keras.backend.argmax(pred, axis=-1)
            input = pred
            state = [h, c]
            output.append(index2word[int(pred)])
            if output[-1] == '<eos>':
                break
        for j in range(0, len(output) - 2):
            print(output[j], end=' ')
        print(output[-2], end='\n')
        print()

        print("signal vector reconstruction")
        z = signal_vector
        input = tf.constant(word2index['<eos>'], shape=(1, 1))
        state = None
        output = []
        for _ in range(0, maxlen):
            dec_embeddings = vae.embeddings(input)
            new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
            dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
            out, h, c = vae.decoder.rnn(dec_input, initial_state=state)
            pred = vae.decoder.vocab_prob(out)
            pred = tf.keras.backend.argmax(pred, axis=-1)
            input = pred
            state = [h, c]
            output.append(index2word[int(pred)])
            if output[-1] == '<eos>':
                break
        for j in range(0, len(output) - 2):
            print(output[j], end=' ')
        print(output[-2], end='\n')
        print()

        print("drop 1 dimension reconstructions")
        drop1_reconstruction(vae, 1 - au, mean, word2index, index2word, maxlen)
        print()


def dimension_homotopy(model_path, seed=0):
    tf.random.set_seed(seed)
    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[1]
        s = s.split(',')

    emb_dim = int(s[0].split()[-1])
    rnn_dim = int(s[1].split()[-1])
    z_dim = int(s[2].split()[-1])
    batch_size = int(s[3].split()[-1])
    lr = float(s[5].split()[-1])
    C = float(s[-3].split()[-1])
    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
    vocab_size = int(s[-1].split()[-1])

    word2index, index2word = load_dic(datapath)
    vae = load_model(emb_dim, rnn_dim, z_dim, vocab_size, lr, model_path)

    noise_dimension = noise(vae, datapath, z_dim, word2index, batch_size, C)

    maxlen = 0
    with open(os.path.join(datapath, 'test.unk.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip() + ' <eos>'
            sentence = sentence.split()
            if len(sentence) > maxlen:
                maxlen = len(sentence)

    z1 = tf.random.normal(shape=(1, z_dim))
    z2 = tf.random.normal(shape=(1, z_dim))
    signal1 = z1 * (1 - noise_dimension)
    signal2 = z2 * (1 - noise_dimension)
    print('normal homotopy:')
    for i in range(0, 6):
        print(i + 1, end='. ')
        z = (1 - 0.2 * i) * z1 + 0.2 * i * z2
        input = tf.constant(word2index['<eos>'], shape=(1, 1))
        state = None
        output = []
        for _ in range(0, maxlen):
            dec_embeddings = vae.embeddings(input)
            new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
            dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
            out, h, c = vae.decoder.rnn(dec_input, initial_state=state)
            pred = vae.decoder.vocab_prob(out)
            pred = tf.keras.backend.argmax(pred, axis=-1)
            input = pred
            state = [h, c]
            output.append(index2word[int(pred)])
            if output[-1] == '<eos>':
                break
        for j in range(0, len(output) - 2):
            print(output[j], end=' ')
        print(output[-2], end='\n')

    print('from sentence:')
    z = signal1
    input = tf.constant(word2index['<eos>'], shape=(1, 1))
    state = None
    output = []
    for _ in range(0, maxlen):
        dec_embeddings = vae.embeddings(input)
        new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
        dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
        out, h, c = vae.decoder.rnn(dec_input, initial_state=state)
        pred = vae.decoder.vocab_prob(out)
        pred = tf.keras.backend.argmax(pred, axis=-1)
        input = pred
        state = [h, c]
        output.append(index2word[int(pred)])
        if output[-1] == '<eos>':
            break
    for j in range(0, len(output) - 2):
        print(output[j], end=' ')
    print(output[-2], end='\n')

    print('to sentence:')
    z = signal2
    input = tf.constant(word2index['<eos>'], shape=(1, 1))
    state = None
    output = []
    for _ in range(0, maxlen):
        dec_embeddings = vae.embeddings(input)
        new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
        dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
        out, h, c = vae.decoder.rnn(dec_input, initial_state=state)
        pred = vae.decoder.vocab_prob(out)
        pred = tf.keras.backend.argmax(pred, axis=-1)
        input = pred
        state = [h, c]
        output.append(index2word[int(pred)])
        if output[-1] == '<eos>':
            break
    for j in range(0, len(output) - 2):
        print(output[j], end=' ')
    print(output[-2], end='\n')

    sd = []
    signal_dimension = (1 - noise_dimension).numpy().tolist()
    for i in range(0, len(signal_dimension)):
        if signal_dimension[i] == 1:
            sd.append(i)
    for dim in sd:
        print('dim {:d} homotopy'.format(dim+1))
        print('from {:.3f} to {:.3f}'.format(signal1.numpy()[0, dim], signal2.numpy()[0, dim]))
        for i in range(0, 5):
            print(i + 1, end='. ')
            z = signal1.numpy()
            z[0, dim] = (1 - 0.25 * i) * signal1.numpy()[0, dim] + 0.25 * i * signal2.numpy()[0, dim]
            z = tf.constant(z)
            input = tf.constant(word2index['<eos>'], shape=(1, 1))
            state = None
            output = []
            for _ in range(0, maxlen):
                dec_embeddings = vae.embeddings(input)
                new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
                dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
                out, h, c = vae.decoder.rnn(dec_input, initial_state=state)
                pred = vae.decoder.vocab_prob(out)
                pred = tf.keras.backend.argmax(pred, axis=-1)
                input = pred
                state = [h, c]
                output.append(index2word[int(pred)])
                if output[-1] == '<eos>':
                    break
            for j in range(0, len(output) - 2):
                print(output[j], end=' ')
            print(output[-2], end='\n')
        signal1 = z


def sentence_chain(vae, sentence, word2index, index2word, maxlen):
    s = sentence + ' <eos>'
    s = s.split()
    chain = [' '.join(s[:-1])]
    for i in range(0, len(s)):
        s[i] = float(word2index[s[i]])
    x = tf.constant(s, shape=(1, len(s)))
    while True:
        enc_embeddings = vae.embeddings(x)
        output = vae.encoder.rnn(enc_embeddings)
        mask = tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x, 1))
        mask = tf.keras.backend.expand_dims(mask)
        mask = tf.keras.backend.repeat_elements(mask, output.shape[2], axis=2)
        output = tf.keras.backend.sum(output * mask, axis=1)
        mean = vae.encoder.mean_layer(output)
        z = mean
        input = tf.constant(word2index['<eos>'], shape=(1, 1))
        state = None
        x_temp = []
        output = []
        for _ in range(0, maxlen):
            dec_embeddings = vae.embeddings(input)
            new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
            dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
            out, h, c = vae.decoder.rnn(dec_input, initial_state=state)
            pred = vae.decoder.vocab_prob(out)
            pred = tf.keras.backend.argmax(pred, axis=-1)
            input = pred
            state = [h, c]
            x_temp.append(int(pred))
            output.append(index2word[int(pred)])
            if output[-1] == '<eos>':
                break
        if ' '.join(output[:-1]) in chain or output[-1] != '<eos>':
            break
        chain.append(' '.join(output[:-1]))
        x = tf.constant(x_temp, shape=(1, len(x_temp)))
    return chain


def sample_sentence_chain(model_path, seed=0):
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
    vae = load_model(emb_dim, rnn_dim, z_dim, vocab_size, lr, model_path)

    sentences = []
    maxlen = 0
    with open(os.path.join(datapath, 'test.unk.txt'), 'r') as f:
        for sentence in f.readlines():
            sentences.append(sentence)
            sentence = sentence.rstrip() + ' <eos>'
            sentence = sentence.split()
            if len(sentence) > maxlen:
                maxlen = len(sentence)

    random.seed(seed)
    sample_index = random.sample(range(len(sentences)), 4)
    for index in sample_index:
        print('sentence chain')
        chain = sentence_chain(vae, sentences[index], word2index, index2word, maxlen)
        for sentence in chain:
            print(sentence)
        print()


def unique(model_path):
    with open(os.path.join(model_path, 'mean.txt'), 'r') as f:
        candidates = list(set(f.readlines()))

    with open(os.path.join(model_path, 'mean_unique.txt'), 'w') as f:
        f.writelines(candidates)

    with open(os.path.join(model_path, 'signal.txt'), 'r') as f:
        candidates = list(set(f.readlines()))

    with open(os.path.join(model_path, 'signal_unique.txt'), 'w') as f:
        f.writelines(candidates)

    with open(os.path.join(model_path, 'random.txt'), 'r') as f:
        candidates = list(set(f.readlines()))

    with open(os.path.join(model_path, 'random_unique.txt'), 'w') as f:
        f.writelines(candidates)

    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[1]
        s = s.split(',')

    dpath = os.path.basename(s[-2].split()[-1])
    if dpath == 'toy2':
        datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), dpath)
        dic = {}
        with open(os.path.join(datapath, 'root.txt'), 'r') as f:
            for root in f.readlines():
                pos = root[:root.find(':')]
                temp = root[root.find(':') + 1:].split()
                for word in temp:
                    dic[word] = pos

        structures = []
        with open(os.path.join(model_path, 'mean_unique.txt'), 'r') as f:
            for sentence in f.readlines():
                sentence = sentence.rstrip()
                sentence = sentence.split()
                for j in range(0, len(sentence)):
                    sentence[j] = dic[sentence[j]]
                structure = '+'.join(sentence)+'\n'
                if structure not in structures:
                    structures.append(structure)

        with open(os.path.join(model_path, 'mean_structures.txt'), 'w') as f:
            f.writelines(structures)

        structures = []
        with open(os.path.join(model_path, 'signal_unique.txt'), 'r') as f:
            for sentence in f.readlines():
                sentence = sentence.rstrip()
                sentence = sentence.split()
                for j in range(0, len(sentence)):
                    sentence[j] = dic[sentence[j]]
                structure = '+'.join(sentence) + '\n'
                if structure not in structures:
                    structures.append(structure)

        with open(os.path.join(model_path, 'signal_structures.txt'), 'w') as f:
            f.writelines(structures)

        structures = []
        with open(os.path.join(model_path, 'random_unique.txt'), 'r') as f:
            for sentence in f.readlines():
                sentence = sentence.rstrip()
                sentence = sentence.split()
                for j in range(0, len(sentence)):
                    sentence[j] = dic[sentence[j]]
                structure = '+'.join(sentence) + '\n'
                if structure not in structures:
                    structures.append(structure)

        with open(os.path.join(model_path, 'random_structures.txt'), 'w') as f:
            f.writelines(structures)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model evaluation')
    tm_help = 'test mode: ' \
              '0 will report signal dimensions and active units, ' \
              '1 will reconstruct test set, ' \
              '2 will report the loss of using signal dimensions on test set and BLEU scores for reconstruction files, ' \
              '3 will sample individual sentences from test set and reconstruct sentences, ' \
              '4 will do homotopy evaluation, ' \
              '5 will sample sentences from test set and construct sentence chains, ' \
              '6 will filter unique sentences from reconstruct files.'
    parser.add_argument('-tm', '--test_mode', default=0, type=int, help=tm_help)
    parser.add_argument('-s', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-m', '--mpath', default='CBT/CBT_Z_64_C_15_0', help='path of model')

    args = parser.parse_args()

    mode = args.test_mode
    seed = args.seed
    model_path = os.path.join(os.path.join(os.getcwd(), 'model'), args.mpath)

    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[1]
        s = s.split(',')

    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))

    if mode == 0:
        noise_quantitative(model_path)
    elif mode == 1:
        reconstruction(model_path)
    elif mode == 2:
        print('signal vector loss test')
        loss_evaluation(model_path)
        reference_path = os.path.join(datapath, 'test.unk.txt')
        print('mean reconstruction file bleu scores (reference original file)')
        candidate_path = os.path.join(model_path, 'mean.txt')
        bleu(candidate_path, reference_path)
        print('signal reconstruction file bleu scores (reference original file)')
        candidate_path = os.path.join(model_path, 'signal.txt')
        bleu(candidate_path, reference_path)
    elif mode == 3:
        sample_sentence(model_path, seed=seed)
    elif mode == 4:
        dimension_homotopy(model_path, seed=seed)
    elif mode == 5:
        sample_sentence_chain(model_path, seed=seed)
    elif mode == 6:
        unique(model_path)
    else:
        print("wrong mode, please type test.py -h for help")