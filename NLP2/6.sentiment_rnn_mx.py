import collections
from NLP2.Common import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
import os
import random
import tarfile

# 总结：1.对比继承自nn.Block的模型。__init__中的参数用于定义网络层的结构。其它方法可以用于迁移参数。
# forward参数是输入参数，如tf中feedict中key对应的value，如keras中模型的iunput。重载了__call__函数，相当于重载了实例()运算符
# forward中定义网络的流动结构。可认为控制输入、输出维度相同，那模型就可以互换了。不管是RNN还是CNN模型。
#       2.sentiment_rnn 用首尾隐藏状态Nx1x2H ->Nx4H 用于特征提取 + 二分类的dense(2)用于结果输出。
#       3.sentiment_cnn 用GlobalMaxPool1D NxCx(T-k+1) -> NxCx1 -> NxC 来提取每个通道（一个序列中）的最重要特征。
# 再用dense(2)对提取的特征进行线性变换，用于二分类，进行结果输出（情感标签）。

# 本函数已保存在d2lzh包中方便以后使用
# def download_imdb(data_dir='data'):
#     url = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
#     sha1 = '01ada507287d82875905620988597833ad4e0903'
#     fname = gutils.download(url, data_dir, sha1_hash=sha1)
#     with tarfile.open(fname, 'r') as f:
#         f.extractall(data_dir)
#
# download_imdb()

def read_imdb(folder='train'):  # 本函数已保存在d2lzh包中方便以后使用
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join('Data/NLP2/aclImdb/', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data

def get_tokenized_imdb(data):  # 本函数已保存在d2lzh包中方便以后使用
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]

def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return text.vocab.Vocabulary(counter, min_freq=5)

def preprocess_imdb(data, vocab):
    max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])
    return features, labels


batch_size = 64
train_data, test_data = d2l.read_imdb('train'), d2l.read_imdb('test')
vocab = d2l.get_vocab_imdb(train_data)
train_ = preprocess_imdb(train_data, vocab)
test_ = preprocess_imdb(test_data,vocab)
train_set = gdata.ArrayDataset(*train_)
test_set = gdata.ArrayDataset(*test_)
train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_set, batch_size)


for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
'# batches:', len(train_iter)

# 输入：NxT
class BiRNN(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers, bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # inputs：NxT，因为mxnet的LSTM需要将序列作为第一维，所以将输入转置后
        # NxT -> TxN -> TxNxD
        embeddings = self.embedding(inputs.T)
        # TxNxD -> TxNx2H
        outputs = self.encoder(embeddings)
        # 只要序列初始与结束的隐藏状态,连结后作为全连接层输入。它的形状为
        # Nx2H + Nx2H -> Nx4H
        encoding = nd.concat(outputs[0], outputs[-1])
        # Nx4H -> Nx2
        outs = self.decoder(encoding)
        return outs

embed_size, num_hiddens, num_layers, ctx = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
net.initialize(init.Xavier(), ctx=ctx)

glove_embedding = text.embedding.create('glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
# 另外设置参数：迁移预训练的词向量
net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.embedding.collect_params().setattr('grad_req', 'null')

lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
# 数据
# 网络：forward中的参数是上面的数据
# loss：目标函数
# trainer：梯度下降优化器
d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)


def predict_sentiment(net, vocab, sentence):
    sentence = nd.array(vocab.to_indices(sentence), ctx=d2l.try_gpu())
    # 输入是 NxT 预测时，N=1
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'


predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])

predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad'])
