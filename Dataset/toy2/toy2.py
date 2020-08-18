import os
import random
import matplotlib.pyplot as plt


def genlol(list_of_list):
    res = []
    if len(list_of_list) > 1:
        for comp in list_of_list[0]:
            temp = genlol(list_of_list[1:])
            for i in range(0, len(temp)):
                res.append(comp+'+'+temp[i])
    elif len(list_of_list) == 1:
        for comp in list_of_list[0]:
            res.append(comp)
    else:
        res = []
    return res


def generate_structure(text_dir):
    basic = ['n.+v.+n.+end-punc.']
    new = []
    for structure in basic:
        words = structure.split('+')
        for i in range(0, len(words)):
            if words[i] == 'n.':
                words[i] = ['n.', 'adj.+n.']
            elif words[i] == 'v.':
                words[i] = ['v.', 'adv.+v.', 'v.+prep.', 'adv.+v.+prep.']
            else:
                words[i] = [words[i]]
        new = new + genlol(words)

    structures = list(new)
    f = open(os.path.join(text_dir, 'structure.txt'), 'w')
    for i in range(0, len(structures)):
        f.writelines(structures[i] + '\n')
        for j in range(0, len(structures)):
            s1 = '+'.join(structures[i].split('+')[:-1])
            s2 = structures[j] + '\n'
            if len(s1.split('+')) + len(s2.split('+')) > 10:
                continue
            f.writelines('conj1.+' + s1 + '+comma+' + s2)
            f.writelines(s1 + '+conj1.+' + s2)
            f.writelines(s1 + '+comma+conj2.+' + s2)
    f.close()


def generate_sentences(structure, dic):
    def generate(voc, n):
        res = []
        if n > 1:
            for i in range(0, len(voc)):
                temp = list(voc)
                word = temp.pop(i)
                s = generate(temp, n - 1)
                for j in range(0, len(s)):
                    res.append(word + '+' + s[j])
        elif n == 1:
            for word in voc:
                res.append(word)
        else:
            res = []
        return res

    pos = []
    order = []
    temp = list(structure)
    while temp:
        pos.append(temp[0])
        order.append([])
        for i in range(0, len(structure)):
            if structure[i] == pos[-1]:
                order[-1].append(i)
        while pos[-1] in temp:
            temp.remove(pos[-1])
    lol = []
    for i in range(0, len(pos)):
        lol.append(generate(dic[pos[i]], len(order[i])))
    res = genlol(lol)
    index = []
    for i in range(0, len(order)):
        index = index + order[i]
    for i in range(0, len(res)):
        temp = res[i].split('+')
        res[i] = list(temp)
        for j in range(0, len(index)):
            res[i][index[j]] = temp[j]
        res[i] = ' '.join(res[i]) + '\n'
    return res


def generate_and_split(text_dir, seed=0):
    structures = []
    with open(os.path.join(text_dir, 'structure.txt'), 'r') as f:
        for sentences in f.readlines():
            structures.append(sentences.rstrip())

    dic = {}
    voc = []
    with open(os.path.join(text_dir, 'root.txt'), 'r') as f:
        for root in f.readlines():
            pos = root[:root.find(':')]
            temp = root[root.find(':') + 1:].split()
            dic[pos] = temp
            for word in temp:
                voc.append(word)

    random.seed(seed)
    random.shuffle(voc)
    with open(os.path.join(text_dir, 'vocab.txt'), 'w') as f:
        f.writelines('\n'.join(voc))

    train_sentences = []
    valid_sentences = []
    test_sentences = []
    for structure in structures:
        structure = structure.split('+')
        temp = generate_sentences(structure, dic)
        random.shuffle(temp)
        temp = temp[:10000]
        train_sentences = train_sentences + temp[:int(0.6*len(temp))]
        valid_sentences = valid_sentences + temp[int(0.6*len(temp)):int(0.8*len(temp))]
        test_sentences = test_sentences + temp[int(0.8*len(temp)):]

    random.shuffle(train_sentences)
    with open(os.path.join(text_dir, 'train.unk.txt'), 'w') as f:
        f.writelines(train_sentences)
    random.shuffle(valid_sentences)
    with open(os.path.join(text_dir, 'valid.unk.txt'), 'w') as f:
        f.writelines(valid_sentences)
    random.shuffle(test_sentences)
    with open(os.path.join(text_dir, 'test.unk.txt'), 'w') as f:
        f.writelines(test_sentences)


def structure_analysis(datapath):
    structures = []
    with open(os.path.join(datapath, 'structure.txt'), 'r') as f:
        for sentences in f.readlines():
            structures.append(sentences.rstrip())

    length = []
    count = []
    for i in range(0, len(structures)):
        temp = structures[i].split('+')
        if 'conj1.' not in temp and 'conj2.' not in temp:
            print('+'.join(temp))
        else:
            if len(temp) not in length:
                length.append(len(temp))
                count.append(0)
            count[length.index(len(temp))] += 1

    for i in range(0, len(length)):
        print('length:{:d}, {:d} complex structures'.format(length[i], count[i]))


def sentence_analysis(datapath):
    voc = []
    with open(os.path.join(datapath, 'vocab.txt'), 'r') as f:
        for vocab in f.readlines():
            vocab = vocab.rstrip()
            voc.append(vocab)

    count = [0 for _ in range(0, len(voc))]
    with open(os.path.join(datapath, 'train.unk.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip()
            sentence = sentence.split()
            for i in range(0, len(sentence)):
                count[voc.index(sentence[i])] += 1

    with open(os.path.join(datapath, 'root.txt'), 'r') as f:
        for root in f.readlines():
            pos = root[:root.find(':')]
            temp = root[root.find(':') + 1:].split()
            sum_temp = 0
            for i in range(0, len(temp)):
                temp[i] = count[voc.index(temp[i])]
                sum_temp += temp[i]
            for i in range(0, len(temp)):
                temp[i] = temp[i] / sum_temp
            bottom = 0
            for i in range(0, len(temp)):
                plt.bar(pos, temp[i], bottom=bottom, label=pos)
                bottom = bottom + temp[i]

    plt.grid(axis='y')
    plt.xlabel('Part of Speech')
    plt.ylabel('Word Proportion')

    plt.savefig(os.path.join(datapath, 'train.png'))
    plt.clf()

    count = [0 for _ in range(0, len(voc))]
    with open(os.path.join(datapath, 'valid.unk.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip()
            sentence = sentence.split()
            for i in range(0, len(sentence)):
                count[voc.index(sentence[i])] += 1

    with open(os.path.join(datapath, 'root.txt'), 'r') as f:
        for root in f.readlines():
            pos = root[:root.find(':')]
            temp = root[root.find(':') + 1:].split()
            sum_temp = 0
            for i in range(0, len(temp)):
                temp[i] = count[voc.index(temp[i])]
                sum_temp += temp[i]
            for i in range(0, len(temp)):
                temp[i] = temp[i] / sum_temp
            bottom = 0
            for i in range(0, len(temp)):
                plt.bar(pos, temp[i], bottom=bottom, label=pos)
                bottom = bottom + temp[i]

    plt.grid(axis='y')
    plt.xlabel('Part of Speech')
    plt.ylabel('Word Proportion')

    plt.savefig(os.path.join(datapath, 'valid.png'))
    plt.clf()

    count = [0 for _ in range(0, len(voc))]
    with open(os.path.join(datapath, 'test.unk.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip()
            sentence = sentence.split()
            for i in range(0, len(sentence)):
                count[voc.index(sentence[i])] += 1

    with open(os.path.join(datapath, 'root.txt'), 'r') as f:
        for root in f.readlines():
            pos = root[:root.find(':')]
            temp = root[root.find(':') + 1:].split()
            sum_temp = 0
            for i in range(0, len(temp)):
                temp[i] = count[voc.index(temp[i])]
                sum_temp += temp[i]
            for i in range(0, len(temp)):
                temp[i] = temp[i] / sum_temp
            bottom = 0
            for i in range(0, len(temp)):
                plt.bar(pos, temp[i], bottom=bottom, label=pos)
                bottom = bottom + temp[i]

    plt.grid(axis='y')
    plt.xlabel('Part of Speech')
    plt.ylabel('Word Proportion')

    plt.savefig(os.path.join(datapath, 'test.png'))
    plt.clf()


if __name__ == '__main__':
    generate_structure(os.getcwd())
    generate_and_split(os.getcwd())
    structure_analysis(os.getcwd())
    sentence_analysis(os.getcwd())