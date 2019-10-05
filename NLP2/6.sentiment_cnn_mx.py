from NLP2.Common import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn

# 总结：1.对比继承自nn.Block的模型。__init__中的参数用于定义网络层的结构。其它方法可以用于迁移参数。
# forward参数是输入参数，如tf中feedict中key对应的value，如keras中模型的iunput。重载了__call__函数，相当于重载了实例()运算符
# forward中定义网络的流动结构。可认为控制输入、输出维度相同，那模型就可以互换了。不管是RNN还是CNN模型。
#       2.sentiment_rnn 用首尾隐藏状态Nx1x2H ->Nx4H 用于特征提取 + 二分类的dense(2)用于结果输出。
#       3.sentiment_cnn 用GlobalMaxPool1D NxCx(T-k+1) -> NxCx1 -> NxC 来提取每个通道（一个序列中）的最重要特征。
# 再用dense(2)对提取的特征进行线性变换，用于二分类，进行结果输出（情感标签）。


batch_size = 64
# d2l.download_imdb()
train_data, test_data = d2l.read_imdb('train'), d2l.read_imdb('test')
vocab = d2l.get_vocab_imdb(train_data)
# 平铺成定长500的序列  T=500    feature：25000x500 label：25000
train_ = d2l.preprocess_imdb(train_data, vocab)
test_ = d2l.preprocess_imdb(test_data, vocab)
train_iter = gdata.DataLoader(gdata.ArrayDataset(*train_), batch_size, shuffle=True)
test_iter = gdata.DataLoader(gdata.ArrayDataset(*test_), batch_size)

# 输入：inputs：NxTxD
# 输出：Nx2
class TextCNN(nn.Block):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # 不参与训练的嵌入层,   这个是为了什么
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = nn.GlobalMaxPool1D()
        self.convs = nn.Sequential()  # 创建多个一维卷积层
        # 100,100,100  3,4,5
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # 将两个形状是(批量大小, 词数, 词向量维度)的嵌入层的输出按词向量连结
        # NxTx2D
        embeddings = nd.concat(self.embedding(inputs), self.constant_embedding(inputs), dim=2)
        # 根据Conv1D要求的输入格式，将词向量维，即一维卷积层的通道维，变换到前一维
        # Nx2DxT   T相当于图像中的W  2D相当于图像中的C   NxCxW
        embeddings = embeddings.transpose((0, 2, 1))
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # NDArray。使用flatten函数去掉最后一维，然后在通道维上连结
        # Nx2DxT Cxk -> NxCx(T-k+1) -> NxCx1 -> NxC -> [NxC, NxC, NxC] -> Nx3C
        encoding = nd.concat(*[nd.flatten(self.pool(conv(embeddings))) for conv in self.convs], dim=1) # dim=1 是相对于concat而说的
        # 应用丢弃法后使用全连接层得到输出
        # Nx3C -> Nx2
        outputs = self.decoder(self.dropout(encoding))
        return outputs


embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
ctx = d2l.try_all_gpus()
net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=ctx)


glove_embedding = text.embedding.create('glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.constant_embedding.weight.set_data(glove_embedding.idx_to_vec)
net.constant_embedding.collect_params().setattr('grad_req', 'null')

lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)

d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])
