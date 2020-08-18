import os
import random
import matplotlib.pyplot as plt


def generate_and_split(vocab_file, text_dir, seed=0):
    def generate(voc):
        res = []
        if len(voc) > 1:
            for i in range(0, len(voc)):
                temp = list(voc)
                word = temp.pop(i)
                s = generate(temp)
                for j in range(0, len(s)):
                    res.append(word + ' ' + s[j])
        elif len(voc) == 1:
            res.append(voc[0] + '\n')
        else:
            res = []
        return res

    voc = []
    with open(vocab_file, 'r') as f:
        for vocab in f.readlines():
            vocab = vocab.rstrip()
            if vocab != '<unk>':
                voc.append(vocab)

    s = generate(voc)
    random.seed(seed)
    random.shuffle(s)
    with open(os.path.join(text_dir, 'train.unk.txt'), 'w') as f:
        f.writelines(s[:int(0.6*len(s))])
    with open(os.path.join(text_dir, 'valid.unk.txt'), 'w') as f:
        f.writelines(s[int(0.6*len(s)):int(0.8*len(s))])
    with open(os.path.join(text_dir, 'test.unk.txt'), 'w') as f:
        f.writelines(s[int(0.8*len(s)):])


def analysis(vocab_file, text_dir, fig_path):
    voc = []
    with open(vocab_file, 'r') as f:
        for vocab in f.readlines():
            vocab = vocab.rstrip()
            if vocab != '<unk>':
                voc.append(vocab)

    proportion = [[0 for _ in range(0, len(voc))] for _ in range(0, len(voc))]
    with open(text_dir, 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip()
            sentence = sentence.split()
            for i in range(0, len(sentence)):
                proportion[voc.index(sentence[i])][i] += 1

    for j in range(0, len(voc)):
        temp = 0
        for i in range(0, len(voc)):
            temp = temp + proportion[i][j]
        for i in range(0, len(voc)):
            proportion[i][j] = proportion[i][j] / temp

    position = [i+1 for i in range(0, len(voc))]
    bottom = [0 for i in range(0, len(voc))]
    for i in range(0, len(proportion)):
        plt.bar(position, proportion[i], bottom=bottom, label=voc[i])
        for j in range(0, len(voc)):
            bottom[j] = bottom[j] + proportion[i][j]

    plt.grid(axis='y')
    plt.xlabel('Position')
    plt.ylabel('Proportion')

    plt.savefig(fig_path)
    plt.clf()


if __name__ == '__main__':
    generate_and_split(os.path.join(os.getcwd(), 'vocab.txt'), os.getcwd())
    analysis(os.path.join(os.getcwd(), 'vocab.txt'), os.path.join(os.getcwd(), 'train.unk.txt'),
             os.path.join(os.getcwd(), 'train.png'))
    analysis(os.path.join(os.getcwd(), 'vocab.txt'), os.path.join(os.getcwd(), 'valid.unk.txt'),
             os.path.join(os.getcwd(), 'valid.png'))
    analysis(os.path.join(os.getcwd(), 'vocab.txt'), os.path.join(os.getcwd(), 'test.unk.txt'),
             os.path.join(os.getcwd(), 'test.png'))