import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cal_mean_std(model_path):
    dpath = os.path.basename(model_path)
    file_path = os.path.join(model_path, dpath + '.csv')
    df = pd.read_csv(file_path)

    index = list(range(0, len(df['model'])))
    while index:
        model = df['model'][index[0]].split(',')
        print(model[0:2])
        data = []
        i = 0
        rm = []
        while i < len(index):
            temp = []
            if df['model'][index[i]].split(',')[0:2] == model[0:2]:
                rm.append(i)
                temp.append(float(df['Epochs'][index[i]]))
                temp.append(float(df['Rec.'][index[i]]))
                temp.append(float(df['KL'][index[i]]))
                if dpath == 'toy1':
                    items = ['WPMS', 'Unique', 'Rule', 'R-WPMS', 'R-Unique', 'R-Rule', 'W-Unique', 'WR-Unique', 'BLEU-2',
                             'R-BLEU-2']
                    for item in items:
                        temp.append(float(df[item][index[i]]))
                elif dpath == 'toy2':
                    temp.append(float(df['SD'][index[i]]))
                    items = ['Match', 'In-Structure', 'Not-In-Structure', 'n.', 'v.', 'adv.', 'adj.', 'prep.', 'conj1.', 'conj2.', 'end-punc.']
                    for item in items:
                        b = str(df[item][index[i]]).split('/')
                        for j in [0, 2]:
                            temp.append(float(b[j]))
                else:
                    temp.append(float(df['SD'][index[i]]))
                    temp.append(float(df['AU'][index[i]]))
                    temp.append(float(df['Rec.(SD)'][index[i]]))
                    b = str(df['Mean'][index[i]]).split('/')
                    for j in range(0, len(b)):
                        temp.append(float(b[j]))
                    b = str(df['Signal'][index[i]]).split('/')
                    for j in range(0, len(b)):
                        temp.append(float(b[j]))
                data.append(temp)
            i = i + 1

        i = len(rm) - 1
        while i >= 0:
            index.pop(rm[i])
            i = i - 1
        data = np.array(data)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        for i in range(0, mean.shape[0]):
            print('mean: {:.2f}, std:{:.2f}'.format(mean[i], std[i]))


def toy2_pos_plot(model_path):
    dpath = os.path.basename(model_path)
    file_path = os.path.join(model_path, dpath + '.csv')
    df = pd.read_csv(file_path)

    marker_plot = ['.', 'o', 's', '*', '+', 'x']
    color_plot = ['b', 'g', 'r', 'c', 'm', 'y']
    c = 0
    index = list(range(0, len(df['model'])))
    items = ['n.', 'v.', 'adv.', 'adj.', 'prep.', 'conj1.', 'conj2.', 'end-punc.']
    while index:
        model = df['model'][index[0]].split(',')
        data = []
        i = 0
        rm = []
        while i < len(index):
            temp = []
            if df['model'][index[i]].split(',')[0:2] == model[0:2]:
                rm.append(i)
                for item in items:
                    b = str(df[item][index[i]]).split('/')
                    temp.append(float(b[0]))
                data.append(temp)
            i = i + 1

        i = len(rm) - 1
        while i >= 0:
            index.pop(rm[i])
            i = i - 1
        data = np.array(data)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        plt.fill_between(items, mean - std, mean + std, alpha=0.2, color=color_plot[c])
        plt.plot(items, mean, marker_plot[c] + color_plot[c] + '-',
                 label='VAE(C={:d},dim={:d})'.format(int(model[0].split('=')[-1]), int(model[1].split('=')[-1])))
        c = c + 1

    plt.legend()
    plt.savefig(os.path.join(model_path, 'toy2_pos.png'))
    plt.clf()


def individual_position_hit_rate(model_path):
    i = 0
    for _, dirs, _ in os.walk(model_path):
        if i == 0:
            break
    marker_plot = ['.', 'o', 's', '*', '+', 'x']
    color_plot = ['b', 'g', 'r', 'c', 'm', 'y']
    c = 0
    for i in range(0, len(dirs)):
        if dirs[i].split('_')[1] == 'collapse':
            path = os.path.join(model_path, dirs[i])
            with open(os.path.join(path, 'epoch_loss.txt'), 'r') as f:
                s = f.readlines()[1]
                s = s.split(',')

            z_dim = int(s[2].split()[-1])
            dpath = os.path.basename(s[-2].split()[-1])
            datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), dpath)
            reference_path = os.path.join(datapath, 'test.unk.txt')
            references = []
            with open(reference_path, 'r') as f:
                for sentence in f.readlines():
                    sentence = sentence.rstrip().split()
                    references.append(sentence)

            length = max([len(references[i]) for i in range(0, len(references))])
            position = [i + 1 for i in range(0, length)]
            dirs.pop(i)
            break

    while dirs:
        temp_dir = dirs[0]
        if temp_dir.split('_')[1] != 'Z':
            dirs.pop(0)
        else:
            if dpath != 'toy1':
                dirs.pop(dirs.index('_'.join(temp_dir.split('_')[0:3]) + '_AE'))
                ae_path = os.path.join(model_path, '_'.join(temp_dir.split('_')[0:3]) + '_AE')

                candidate_path = os.path.join(ae_path, 'mean.txt')
                candidates = []
                with open(candidate_path, 'r') as f:
                    for sentence in f.readlines():
                        sentence = sentence.rstrip().split()
                        candidates.append(sentence)

                count_match = [0 for _ in range(0, length)]
                count = [0 for _ in range(0, length)]
                for reference, candidate in zip(references, candidates):
                    temp = [int(reference[i] == candidate[i]) for i in range(min(len(reference), len(candidate)))]
                    for i in range(min(len(count), len(temp))):
                        count_match[i] += temp[i]
                    for i in range(len(reference)):
                        count[i] += 1

                for i in range(0, len(count)):
                    count_match[i] = count_match[i] / count[i]

                plt.plot(position, count_match, marker_plot[c]+color_plot[c]+'-', color=color_plot[c], label='AE(dim={:d})'.format(z_dim))
                c = (c + 1) % len(color_plot)
                plt.ylim(0, 1)
                plt.grid(axis='y')
                plt.xlabel('Position')
                plt.ylabel('IPHR')

            C_value = []
            for dir in dirs:
                if dir.split('_')[0:3] == temp_dir.split('_')[0:3]:
                    if float(dir.split('_')[4]) not in C_value:
                        C_value.append(float(dir.split('_')[4]))

            C_value.sort()
            while C_value:
                C = C_value[0]
                C_value.pop(0)
                path = os.path.join(model_path, '_'.join(temp_dir.split('_')[0:3])+'_C_'+str(int(C)))
                proportion = []
                for i in range(0, 3):
                    mpath = path + '_' + str(i)
                    dirs.pop(dirs.index(os.path.basename(mpath)))
                    candidate_path = os.path.join(mpath, 'mean.txt')
                    candidates = []
                    with open(candidate_path, 'r') as f:
                        for sentence in f.readlines():
                            sentence = sentence.rstrip().split()
                            candidates.append(sentence)

                    count_match = [0 for _ in range(0, length)]
                    count = [0 for _ in range(0, length)]
                    for reference, candidate in zip(references, candidates):
                        temp = [int(reference[i] == candidate[i]) for i in range(min(len(reference), len(candidate)))]
                        for i in range(min(len(count), len(temp))):
                            count_match[i] += temp[i]
                        for i in range(len(reference)):
                            count[i] += 1

                    for i in range(0, len(count)):
                        count_match[i] = count_match[i] / count[i]

                    proportion.append(count_match)

                proportion = np.array(proportion)
                prop_mean = np.mean(proportion, axis=0)
                prop_std = np.std(proportion, axis=0)
                plt.fill_between(position, prop_mean - prop_std, prop_mean + prop_std, alpha=0.2, color=color_plot[c])
                plt.plot(position, prop_mean, marker_plot[c]+color_plot[c]+'-', label='VAE(C={:d},dim={:d})'.format(int(C), z_dim))
                c = (c + 1) % len(color_plot)

            plt.legend()
            plt.savefig(os.path.join(model_path, '_'.join(temp_dir.split('_')[0:3]) + '.png'))
            plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model evaluation')
    parser.add_argument('-tm', '--test_mode', default=0, type=int, help='test mode:')
    parser.add_argument('-m', '--mpath', default='toy1', help='path of model')

    args = parser.parse_args()

    mode = args.test_mode
    model_path = os.path.join(os.path.join(os.getcwd(), 'model'), args.mpath)

    if mode == 0:
        cal_mean_std(model_path)
        if os.path.basename(model_path) == 'toy2':
            toy2_pos_plot(model_path)
    elif mode == 1:
        individual_position_hit_rate(model_path)
    else:
        print("wrong mode, please type basic_evaluation.py -h for help")