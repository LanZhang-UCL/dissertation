import os
import argparse
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from Dataset.toy2.toy2 import generate_sentences
from test import load_dic, load_model


def word_position_match_score(candidate_path, reference_path):
    references = []
    with open(reference_path, 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip().split()
            references.append(sentence)

    candidates = []
    with open(candidate_path, 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip().split()
            candidates.append(sentence)

    score = 0
    for reference, candidate in zip(references, candidates):
        s_score = sum([int(reference[i] == candidate[i]) for i in range(min(len(reference), len(candidate)))])
        score = score + s_score / len(reference)

    print("word position match score {}".format(score/len(references)*100))


def toy1_unique_and_rule(candidate_path, n, voc):
        candidates = []
        with open(candidate_path, 'r') as f:
            for sentence in f.readlines():
                sentence = sentence.rstrip()
                candidates.append(sentence)
        print('Unique rate: {:.2f}%.'.format(len(candidates) / n * 100))
        correct_count = 0
        for sentence in candidates:
            sentence = sentence.split()
            state = False
            for word in voc:
                if sentence.count(word) != 1:
                    state = True
                    break
            if state:
                continue
            correct_count += 1
        print('There are {:.2f}% sentences satisfying rule in file.'.format(correct_count / len(candidates) * 100))


def weighted_unique(candidate_path, reference_path):
    index = []
    candidates = []
    with open(candidate_path, 'r') as f:
        sentences = f.readlines()
        for i in range(0, len(sentences)):
            if sentences[i] not in candidates:
                index.append(i)
                candidates.append(sentences[i])

    references = []
    with open(reference_path, 'r') as f:
        sentences = f.readlines()
        n = len(sentences)
        for i in index:
            references.append(sentences[i])

    for i in range(0, len(index)):
        references[i] = references[i].rstrip().split()
        candidates[i] = candidates[i].rstrip().split()

    score = 0
    for reference, candidate in zip(references, candidates):
        s_score = sum([int(reference[i] == candidate[i]) for i in range(min(len(reference), len(candidate)))])
        score = score + s_score / len(reference)

    print("Weighted unique rate {:.2f}%".format(score / n * 100))


def toy1_analysis(model_path):
    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[1]
        s = s.split(',')

    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))

    reference_path = os.path.join(datapath, 'test.unk.txt')
    mean_path = os.path.join(model_path, 'mean.txt')
    random_path = os.path.join(model_path, 'random.txt')
    mean_unique_path = os.path.join(model_path, 'mean_unique.txt')
    random_unique_path = os.path.join(model_path, 'random_unique.txt')

    word_position_match_score(mean_path, reference_path)
    word_position_match_score(random_path, reference_path)

    voc = []
    with open(os.path.join(datapath, 'vocab.txt'), 'r') as f:
        for vocab in f.readlines():
            vocab = vocab.rstrip()
            voc.append(vocab)

    with open(reference_path, 'r') as f:
        n = len(f.readlines())

    toy1_unique_and_rule(mean_unique_path, n, voc)
    toy1_unique_and_rule(random_unique_path, n, voc)

    weighted_unique(mean_path, reference_path)
    weighted_unique(random_path, reference_path)


def toy2_sentence_analysis(candidate_path, reference_path, root_path):
    dic = {}
    pos = []
    count = {}
    with open(root_path, 'r') as f:
        for root in f.readlines():
            temp_pos = root[:root.find(':')]
            pos.append(temp_pos)
            count[temp_pos]= [0, 0]
            temp_word = root[root.find(':') + 1:].split()
            for word in temp_word:
                dic[word] = temp_pos

    references = []
    with open(reference_path, 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip()
            references.append(sentence)

    candidates = []
    with open(candidate_path, 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip()
            candidates.append(sentence)

    match_count = 0

    for i in range(0, len(references)):
        reference = references[i].split()
        candidate = candidates[i].split()

        reference_structure = '+'.join([dic[reference[j]] for j in range(0, len(reference))])
        candidate_structure = '+'.join([dic[candidate[j]] for j in range(0, len(candidate))])
        if reference_structure == candidate_structure:
            match_count = match_count + 1
            for j in range(0, len(reference)):
                count[dic[reference[j]]][0] += 1
                if reference[j] == candidate[j]:
                    count[dic[reference[j]]][1] += 1
    print('There are {:.2f}% match.'.format(match_count / len(references) * 100))
    print('Part of Speech correctness:')
    for temp_pos in pos:
        print('{}:{:.2f}'.format(temp_pos, count[temp_pos][1] / count[temp_pos][0] * 100), end=' ')
    print()


def toy2_structure_analysis(candidate_path, reference_path):
    structures = []
    with open(reference_path, 'r') as f:
        for sentence in f.readlines():
            structures.append(sentence.rstrip())

    new_structures = []
    with open(candidate_path, 'r') as f:
        for sentence in f.readlines():
            new_structures.append(sentence.rstrip())

    in_structures = []
    not_in_structures = []
    for i in range(0, len(new_structures)):
        structure = new_structures[i]
        if structure in structures:
            in_structures.append(structure)
        else:
            not_in_structures.append(structure)
    print('{:d} in structures, {:d} not in structures.'.format(len(in_structures), len(not_in_structures)))


def toy2_analysis(model_path):
    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[1]
        s = s.split(',')

    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))

    root_path = os.path.join(datapath, 'root.txt')
    reference_path = os.path.join(datapath, 'test.unk.txt')
    mean_path = os.path.join(model_path, 'mean.txt')
    signal_path = os.path.join(model_path, 'signal.txt')
    random_path = os.path.join(model_path, 'random.txt')

    toy2_sentence_analysis(mean_path, reference_path, root_path)
    toy2_sentence_analysis(signal_path, reference_path, root_path)
    toy2_sentence_analysis(random_path, reference_path, root_path)

    reference_path = os.path.join(datapath, 'structure.txt')
    mean_path = os.path.join(model_path, 'mean_structures.txt')
    signal_path = os.path.join(model_path, 'signal_structures.txt')
    random_path = os.path.join(model_path, 'random_structures.txt')

    toy2_structure_analysis(mean_path, reference_path)
    toy2_structure_analysis(signal_path, reference_path)
    toy2_structure_analysis(random_path, reference_path)


def sentences_mean(temp, word2index, batch_size, vae, z_dim):
    sentences = []
    maxlen = 0
    for sentence in temp:
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

    mean_plot = [[] for _ in range(z_dim)]
    for x_batch_test in test_dataset:
        enc_embeddings = vae.embeddings(x_batch_test)
        output = vae.encoder.rnn(enc_embeddings)
        mask = tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x_batch_test, 1))
        mask = tf.keras.backend.expand_dims(mask)
        mask = tf.keras.backend.repeat_elements(mask, output.shape[2], axis=2)
        output = tf.keras.backend.sum(output * mask, axis=1)
        mean = vae.encoder.mean_layer(output)

        mean = mean.numpy()
        for j in range(0, mean.shape[1]):
            for i in range(0, mean.shape[0]):
                mean_plot[j].append(mean[i][j])
    return mean_plot


def toy2_disentangling(model_path, seed=0):
    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[1]
        s = s.split(',')

    emb_dim = int(s[0].split()[-1])
    rnn_dim = int(s[1].split()[-1])
    z_dim = int(s[2].split()[-1])
    batch_size = int(s[3].split()[-1])
    lr = float(s[5].split()[-1])
    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
    vocab_size = int(s[-1].split()[-1])

    word2index, index2word = load_dic(datapath)
    vae = load_model(emb_dim, rnn_dim, z_dim, vocab_size, lr, model_path)

    with open(os.path.join(datapath, 'test.unk.txt'), 'r') as f:
        temp = f.readlines()

    mean_plot = [sentences_mean(temp, word2index, batch_size, vae, z_dim)]

    random.seed(seed)
    structures = []
    with open(os.path.join(datapath, 'structure.txt'), 'r') as f:
        for sentence in f.readlines():
            structures.append(sentence.rstrip())

    dic = {}
    with open(os.path.join(datapath, 'root.txt'), 'r') as f:
        for root in f.readlines():
            pos = root[:root.find(':')]
            temp = root[root.find(':') + 1:].split()
            dic[pos] = temp

    sample_index = random.sample(range(0, len(structures)), 3)
    for index in sample_index:
        structure = structures[index]
        print(structure)
        structure = structure.split('+')
        temp = generate_sentences(structure, dic)
        random.shuffle(temp)
        temp = temp[0:20000]

        mean_plot.append(sentences_mean(temp, word2index, batch_size, vae, z_dim))

    color_plot = ['g', 'r', 'c', 'm']
    width = 0.2

    x = []
    y = []
    for dim in range(0, z_dim):
        x = x + [dim + 1 - 1.5 * width for _ in range(len(mean_plot[0][dim]))]
        y = y + mean_plot[0][dim]
    plt.scatter(x, y, s=3, color=color_plot[0], label='test set')
    for i in range(1, len(mean_plot)):
        x = []
        y = []
        for dim in range(0, z_dim):
            x = x + [dim + 1 - 1.5 * width + i * width for _ in range(len(mean_plot[i][dim]))]
            y = y + mean_plot[i][dim]
        plt.scatter(x, y, s=3, marker='s', color=color_plot[i], label='structure'+str(i))
    plt.xlabel('dimension')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.savefig(os.path.join(model_path, 'disentangling_'+str(seed)+'.png'))
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluations for toy datasets.')
    tm_help = 'test mode: ' \
              '0 will analysis reconstruction files, ' \
              '1 will do disentangling for VAEs of toy dataset 2.'
    parser.add_argument('-tm', '--test_mode', default=0, type=int, help=tm_help)
    parser.add_argument('-s', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-m', '--mpath', default='toy2/toy2_Z_16_C_4_0', help='path of model')

    args = parser.parse_args()

    seed = args.seed
    mode = args.test_mode
    model_path = os.path.join(os.path.join(os.getcwd(), 'model'), args.mpath)

    if mode == 0:
        with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
            s = f.readlines()[1]
            s = s.split(',')

        dpath = os.path.basename(s[-2].split()[-1])
        if dpath == 'toy1':
            toy1_analysis(model_path)
        elif dpath == 'toy2':
            toy2_analysis(model_path)
        else:
            print('this model is not a toy-dataset model!')
    elif mode == 1:
        with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
            s = f.readlines()[1]
            s = s.split(',')

        dpath = os.path.basename(s[-2].split()[-1])
        if dpath != 'toy2':
            print('this mode is only for toy dataset2.')
        else:
            toy2_disentangling(model_path, seed=seed)
    else:
        print("wrong mode, please type toy_test.py -h for help")
