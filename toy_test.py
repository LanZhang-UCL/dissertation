import os
import argparse


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


def unique(model_path):
    count = 0
    candidates = []
    with open(os.path.join(model_path, 'mean.txt'), 'r') as f:
        for sentence in f.readlines():
            count = count + 1
            if sentence not in candidates:
                candidates.append(sentence)

    with open(os.path.join(model_path, 'unique.txt'), 'w') as f:
        f.writelines(candidates)

    print("Unique rate: {:.2f}".format(len(candidates) / count * 100))

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
        with open(os.path.join(model_path, 'unique.txt'), 'r') as f:
            for sentence in f.readlines():
                sentence = sentence.rstrip()
                sentence = sentence.split()
                for j in range(0, len(sentence)):
                    sentence[j] = dic[sentence[j]]
                structure = '+'.join(sentence)+'\n'
                if structure not in structures:
                    structures.append(structure)

        with open(os.path.join(model_path, 'structures.txt'), 'w') as f:
            f.writelines(structures)


def toy1_analysis(model_path):
    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[1]
        s = s.split(',')

    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))

    voc = []
    with open(os.path.join(datapath, 'vocab.txt'), 'r') as f:
        for vocab in f.readlines():
            vocab = vocab.rstrip()
            voc.append(vocab)

    candidates = []
    with open(os.path.join(model_path, 'mean.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip()
            candidates.append(sentence)

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
    print('There are {:.2f}% correct sentences.'.format(correct_count / len(candidates) * 100))


def toy2_analysis(model_path):
    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[1]
        s = s.split(',')

    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))

    structures = []
    with open(os.path.join(datapath, 'structure.txt'), 'r') as f:
        for sentences in f.readlines():
            structures.append(sentences.rstrip())

    new_structures = []
    with open(os.path.join(model_path, 'structures.txt'), 'r') as f:
        for sentences in f.readlines():
            new_structures.append(sentences.rstrip())

    dic = {}
    with open(os.path.join(datapath, 'root.txt'), 'r') as f:
        for root in f.readlines():
            pos = root[:root.find(':')]
            temp = root[root.find(':') + 1:].split()
            for word in temp:
                dic[word] = pos

    references = []
    with open(os.path.join(datapath, 'test.unk.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip()
            references.append(sentence)

    candidates = []
    with open(os.path.join(model_path, 'mean.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip()
            candidates.append(sentence)

    match_count = 0
    structure_count = 0
    for i in range(0, len(references)):
        reference = references[i].split()
        for j in range(0, len(reference)):
            reference[j] = dic[reference[j]]

        candidate = candidates[i].split()
        for j in range(0, len(candidate)):
            candidate[j] = dic[candidate[j]]

        reference = '+'.join(reference)
        candidate = '+'.join(candidate)
        if reference == candidate:
            match_count = match_count + 1
        if candidate in structures:
            structure_count = structure_count + 1
    print('There are {:.2f}% match.'.format(match_count / len(candidates) * 100))
    print('There are {:.2f}% valid.'.format(structure_count / len(candidates) * 100))

    in_structures = []
    not_in_structures = []
    for i in range(0, len(new_structures)):
        structure = new_structures[i]
        if structure in structures:
            if structure not in in_structures:
                in_structures.append(structure)
        else:
            if structure not in not_in_structures:
                not_in_structures.append(structure)
    print('{:d} in structures, {:d} not in structures.'.format(len(in_structures), len(not_in_structures)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model evaluation')
    parser.add_argument('-tm', '--test_mode', default=0, type=int, help='test mode:')
    parser.add_argument('-m', '--mpath', default='toy2/toy3_Z_4_C_2_0', help='path of model')

    args = parser.parse_args()

    mode = args.test_mode
    model_path = os.path.join(os.path.join(os.getcwd(), 'model'), args.mpath)

    if mode == 0:
        with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
            s = f.readlines()[1]
            s = s.split(',')

        datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
        reference_path = os.path.join(datapath, 'test.unk.txt')

        word_position_match_score(os.path.join(model_path, 'mean.txt'), reference_path)
    elif mode == 1:
        unique(model_path)
    elif mode == 2:
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
    else:
        print("wrong mode, please type toy_test.py -h for help")
