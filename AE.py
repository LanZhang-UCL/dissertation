import tensorflow as tf
import argparse
import time
import os
import matplotlib.pyplot as plt


class Encoder(tf.keras.Model):
    def __init__(self, rnn_dim, z_dim):
        super(Encoder, self).__init__()
        self.rnn = tf.keras.layers.LSTM(rnn_dim, return_state=False, return_sequences=True)
        self.mean_layer = tf.keras.layers.Dense(z_dim)

    def call(self, x, embeddings):
        output = self.rnn(embeddings)
        mask = tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x, 1))
        mask = tf.keras.backend.expand_dims(mask)
        mask = tf.keras.backend.repeat_elements(mask, output.shape[2], axis=2)
        output = tf.keras.backend.sum(output*mask, axis=1)
        z = self.mean_layer(output)
        return z


class Decoder(tf.keras.Model):
    def __init__(self, rnn_dim, vocab_size):
        super(Decoder, self).__init__()
        self.rnn = tf.keras.layers.LSTM(rnn_dim, return_state=True, return_sequences=True)
        self.vocab_prob = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, x, embeddings, z):
        new_z = tf.keras.backend.repeat(z, embeddings.shape[1])
        dec_input = tf.keras.layers.concatenate([embeddings, new_z], axis=-1)
        output, _, _ = self.rnn(dec_input)
        predictions = self.vocab_prob(output)
        rec_loss = self.reconstruction_loss(x, predictions)
        return predictions, rec_loss

    @staticmethod
    def reconstruction_loss(x, predictions):
        mask = 1 - tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x, 0))
        prob = tf.keras.backend.sparse_categorical_crossentropy(x, predictions)*mask
        res = tf.keras.backend.sum(prob, axis=-1)
        return res


class AE(tf.keras.Model):
    def __init__(self, emb_dim, rnn_dim, z_dim, vocab_size):
        super(AE, self).__init__()
        self.embeddings = tf.keras.layers.Embedding(vocab_size, emb_dim)
        self.encoder = Encoder(rnn_dim, z_dim)
        self.decoder = Decoder(rnn_dim, vocab_size)

    def call(self, x):
        enc_embeddings = self.embeddings(x)
        z = self.encoder(x, enc_embeddings)
        y = tf.keras.backend.concatenate([tf.constant(1, shape=(x.shape[0], 1)), x[:, :-1]], axis=-1)
        dec_embeddings = self.embeddings(y)
        predictions, rec_loss = self.decoder(x, dec_embeddings, z)
        return predictions, rec_loss


def load_data(batch_size, path):
    print("loading data")
    dic = {'<pad>': 0, '<eos>': 1}
    index = 2
    with open(os.path.join(path, 'vocab.txt'), 'r') as f:
        for vocab in f.readlines():
            vocab = vocab.rstrip()
            dic[vocab] = index
            index = index + 1

    sentences = []
    maxlen = 0
    with open(os.path.join(path, 'train.unk.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip() + ' <eos>'
            sentence = sentence.split()
            for i in range(len(sentence)):
                sentence[i] = dic[sentence[i]]
            if len(sentence) > maxlen:
                maxlen = len(sentence)
            sentences.append(sentence)

    x_train = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=maxlen, padding='post', truncating='post')

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(len(x_train)).batch(batch_size)

    sentences = []
    maxlen = 0
    with open(os.path.join(path, 'valid.unk.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip() + ' <eos>'
            sentence = sentence.split()
            for i in range(len(sentence)):
                sentence[i] = dic[sentence[i]]
            if len(sentence) > maxlen:
                maxlen = len(sentence)
            sentences.append(sentence)

    x_val = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=maxlen, padding='post', truncating='post')

    val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
    val_dataset = val_dataset.shuffle(len(x_val)).batch(batch_size)

    sentences = []
    maxlen = 0
    with open(os.path.join(path, 'test.unk.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip() + ' <eos>'
            sentence = sentence.split()
            for i in range(len(sentence)):
                sentence[i] = dic[sentence[i]]
            if len(sentence) > maxlen:
                maxlen = len(sentence)
            sentences.append(sentence)

    x_test = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=maxlen, padding='post', truncating='post')

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.shuffle(len(x_test)).batch(batch_size)

    print("number of training data:{:d}, number of validation data:{:d}, number of test data:{:d}"
          .format(len(x_train), len(x_val), len(x_test)))
    return train_dataset, val_dataset, test_dataset, dic


def train(ae, optimizer, epochs, ckpt_man, ckpt_dir, train_dataset, val_dataset, z_dim):

    train_window = []
    val_window = []
    step_count = 1

    for epoch in range(1, epochs+1):
        print("Start of epoch {:d}".format(epoch))
        start_time = time.time()

        total_loss = 0
        for step, x_batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                pred, rec_loss = ae(x_batch_train)

                loss = tf.keras.backend.mean(rec_loss)

                grads = tape.gradient(loss, ae.weights)
                optimizer.apply_gradients(zip(grads, ae.weights))

            total_loss = total_loss + loss

            if step_count % 100 == 0:
                print("step:{:d} train_loss:{:.4f} ".format(step_count, loss))

            step_count = step_count + 1

        train_loss = total_loss/(step+1)
        with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
            f.write("train_loss:{:.4f} ".format(train_loss))
        print("train_loss:{:.4f}".format(train_loss))

        total_loss = 0
        for step, x_batch_val in enumerate(val_dataset):
            pred, rec_loss = ae(x_batch_val)

            loss = tf.keras.backend.mean(rec_loss)

            total_loss = total_loss + loss

        val_loss = total_loss/(step+1)
        with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
            f.write("loss:{:.4f}\n".format(val_loss))
        print("loss:{:.4f}".format(val_loss))

        mean_plot = [[] for _ in range(z_dim)]
        for step, x_batch_val in enumerate(val_dataset):
            enc_embeddings = ae.embeddings(x_batch_val)
            mean = ae.encoder(x_batch_val, enc_embeddings)

            mean = mean.numpy()
            for j in range(0, mean.shape[1]):
                for i in range(0, mean.shape[0]):
                    mean_plot[j].append(mean[i][j])

        color_plot = ['b', 'g', 'r', 'c', 'm']

        for dim in range(0, z_dim):
            dim_plot = [dim + 1 for _ in range(len(mean_plot[dim]))]
            plt.scatter(dim_plot, mean_plot[dim], color=color_plot[dim % len(color_plot)])
        plt.xlabel('dimension')
        plt.ylabel('Mean Value')
        plt.savefig(os.path.join(os.path.join(ckpt_dir, 'figures'), 'mean_' + str(epoch)))
        plt.clf()

        ckpt_man.save()
        print("time taken:{:.2f}s".format(time.time() - start_time))

        if len(train_window) < 3:
            train_window.append(train_loss)
            val_window.append(val_loss)
            continue
        train_window.pop(0)
        val_window.pop(0)
        train_window.append(train_loss)
        val_window.append(val_loss)
        if max(train_window)-min(train_window) < 0.001:
            break
        if min(val_window) == val_window[0]:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE', epilog='start training')
    parser.add_argument('-e','--emb_dim', default=8, type=int, help='embedding dimensions, default: 8')
    parser.add_argument('-r', '--rnn_dim', default=64, type=int, help='RNN dimensions, default: 64')
    parser.add_argument('-z', '--z_dim', default=6, type=int, help='latent space dimensions, default: 6')
    parser.add_argument('-b', '--batch', default=512, type=int, help='batch size, default: 512')
    parser.add_argument('-lr', '--learning_rate', default=0.00075, type=float, help='learning rate, default: 0.00075')
    parser.add_argument('--epochs', default=50, type=int, help='epochs number, default: 50')
    parser.add_argument('--datapath', default='toy1', help='path of data under dataset directory, default: toy1')
    parser.add_argument('-tm', '--training_mode', default=0, type=int, help='training mode')
    parser.add_argument('-s', '--seed', default=0, type=int, help='global random seed')
    parser.add_argument('-m', '--mpath', required=True, help='path of model')

    args = parser.parse_args()

    tm = args.training_mode
    seed = args.seed
    batch_size = args.batch
    lr = args.learning_rate
    epochs = args.epochs
    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), args.datapath)
    ckpt_dir = os.path.join(os.path.join(os.getcwd(), 'model'), args.mpath)

    if tm == 1:
        tf.random.set_seed(seed)

    train_dataset, val_dataset, test_dataset, dic = load_data(batch_size, datapath)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    if os.system('mkdir ' + ckpt_dir) != 0:
        print('This is not first training.')
        exit()
    os.system('mkdir ' + os.path.join(ckpt_dir, 'figures'))
    emb_dim = args.emb_dim
    rnn_dim = args.rnn_dim
    z_dim = args.z_dim
    ae = AE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=len(dic))

    total_loss = 0
    total_kl_loss = 0
    total_rec_loss = 0
    total_vae_loss = 0
    for step, x_batch_val in enumerate(val_dataset):
        pred, rec_loss = ae(x_batch_val)

        loss = tf.keras.backend.mean(rec_loss)

        total_loss = total_loss + loss

    val_loss = total_loss / (step + 1)
    with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
        f.write("loss:{:.4f}\n".format(val_loss))
    print("loss:{:.4f}\n".format(val_loss))

    mean_plot = [[] for _ in range(z_dim)]
    for step, x_batch_val in enumerate(val_dataset):
        enc_embeddings = ae.embeddings(x_batch_val)
        mean = ae.encoder(x_batch_val, enc_embeddings)

        mean = mean.numpy()
        for j in range(0, mean.shape[1]):
            for i in range(0, mean.shape[0]):
                mean_plot[j].append(mean[i][j])

    color_plot = ['b', 'g', 'r', 'c', 'm']

    for dim in range(0, z_dim):
        dim_plot = [dim + 1 for _ in range(len(mean_plot[dim]))]
        plt.scatter(dim_plot, mean_plot[dim], color=color_plot[dim % len(color_plot)])
    plt.xlabel('dimension')
    plt.ylabel('Mean Value')
    plt.savefig(os.path.join(os.path.join(ckpt_dir, 'figures'), 'mean_' + str(0)))
    plt.clf()

    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=ae)
    ckpt_man = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=1, checkpoint_name='ckpt')

    with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
        f.write("training configure: embedding dimension {:d}, RNN dimension {:d}, z dimension {:d}, batch size {:d}, "
                "epoch number {:d}, learning rate {:f}, dataset {}, vocabulary size {:d}\n"
                .format(emb_dim, rnn_dim, z_dim, batch_size, epochs, lr, datapath, len(dic)))

    train(ae, optimizer, epochs, ckpt_man, ckpt_dir, train_dataset, val_dataset, z_dim)

    with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
        f.write('training ends, model at {}\n'.format(ckpt_dir))
    print('training ends, model at {}'.format(ckpt_dir))

    print("model test")
    total_loss = 0
    for step, x_batch_test in enumerate(test_dataset):
        pred, rec_loss = ae(x_batch_test)

        loss = tf.keras.backend.mean(rec_loss)

        total_loss = total_loss + loss

    test_loss = total_loss / (step + 1)
    with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
        f.write("test results\nloss:{:.4f}\n".format(test_loss))
    print("loss:{:.4f}\n".format(test_loss))
    print('test ends, model at {}'.format(ckpt_dir))

    mean_plot = [[] for _ in range(z_dim)]
    for step, x_batch_test in enumerate(test_dataset):
        enc_embeddings = ae.embeddings(x_batch_test)
        mean = ae.encoder(x_batch_test, enc_embeddings)

        mean = mean.numpy()
        for j in range(0, mean.shape[1]):
            for i in range(0, mean.shape[0]):
                mean_plot[j].append(mean[i][j])

    color_plot = ['b', 'g', 'r', 'c', 'm']

    for dim in range(0, z_dim):
        dim_plot = [dim + 1 for _ in range(len(mean_plot[dim]))]
        plt.scatter(dim_plot, mean_plot[dim], color=color_plot[dim % len(color_plot)])
    plt.xlabel('dimension')
    plt.ylabel('Mean Value')
    plt.savefig(os.path.join(os.path.join(ckpt_dir, 'figures'), 'mean'))
    plt.clf()