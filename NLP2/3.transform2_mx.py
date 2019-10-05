import math, io
from NLP2.Common import d2l
from mxnet import nd, autograd
from mxnet.gluon import nn


# num_steps: T  embed_size: D
embed_size, units, num_layers, dropout = 256, 256, 2, 0.0
batch_size, max_seq_len = 32, 13  # N,T     # 13对应10000样本
lr, num_epochs, ctx = 0.005, 100, d2l.try_gpu()
num_hiddens, num_heads = 128, 4
num_examples=10000
PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'


###### Multi-Head Attention ######
# 总的来说输入输出维度不变，都是 # (batch_size, num_items, units)。
# 单个qkv看也是是 # (batch_size, num_items, units) -> (batch_size, num_items, units)
# Positional Encoding的存在，增加输入词的序列前后属性
# 输入：qkv: NxTxDe(De：Embedding的维度) NxTxDe ->Wqkv  NxTxD
# 输出：1个 NxTxD 这里T不再是1了，V还是加权后平均，使T->1   因为此时Q的T不是1，是T   所以进行了T平方次加权 平均了T词
class MultiHeadAttention(nn.Block):
    def __init__(self, units, num_heads, dropout, **kwargs):  # units = d_o
        super(MultiHeadAttention, self).__init__(**kwargs)
        assert units % num_heads == 0
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Dense(units, use_bias=False, flatten=False)
        self.W_k = nn.Dense(units, use_bias=False, flatten=False)
        self.W_v = nn.Dense(units, use_bias=False, flatten=False)

    # query, key, and value shape: (batch_size, num_items, dim)
    # (batch_size * num_heads, num_items, p), where units = p * num_heads.      512的embedding 8x64 num_heads=8 Df=64
    # 设D=units D=K*Df
    def forward(self, query, key, value, valid_length):
        # N*KxTxDf
        # query: NxTxH  key、value: NxTxH        Q中的T不再是1
        # 经过W: NxTxD              NxTxD
        # trans: N*KxTxDf           N*KxTxDf
        # transponse NxTxD -> N*KxTxDf
        query, key, value = [transpose_qkv(X, self.num_heads) for X in (self.W_q(query), self.W_k(key), self.W_v(value))]
        # print(query.shape)
        if valid_length is not None:
            # Copy valid_length by num_heads times
            if valid_length.ndim == 1:
                valid_length = valid_length.tile(self.num_heads)
            else:
                valid_length = valid_length.tile((self.num_heads, 1))
        # 输入：# Q: NxTxH     K、V：NxTxH
        # 输出：# context： Nx1xH
        # 输入：# Q: N*KxTxDf     K、V：N*KxTxDf
        # 此时：# context: N*KxTxDf 维度同查询Q
        output = self.attention(query, key, value, valid_length)
        # N*KxTxDf -> NxTxD=units=k*p
        return transpose_output(output, self.num_heads)

# transponse NxTxD -> N*KxTxDf
def transpose_qkv(X, num_heads):
    # 0 means copying the shape element, -1 means inferring its value
    # NxTxD -> NxTxKxDf
    X = X.reshape((0, 0, num_heads, -1))
    # NxKxTxDf
    X = X.transpose((0, 2, 1, 3))
    # N*KxTxDf
    return X.reshape((-1, 0, 0), reverse=True)

def transpose_output(X, num_heads):
    # A reversed version of transpose_qkv
    # X: N*KxTxDf -> (N,K,T,Df)
    X = X.reshape((-1, num_heads, 0, 0), reverse=True)
    # (N,T,K,Df)
    X = X.transpose((0, 2, 1, 3))
    # (N,T,D=units=k*p)
    return X.reshape((0, 0, -1))


###### Position-wise Feed-Forward Networks  ######
# 输入：NxTxD
# 输出：NxTxD
class PositionWiseFFN(nn.Block):
    def __init__(self, units, hidden_size, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.ffn_1 = nn.Dense(hidden_size, flatten=False, activation='relu')
        self.ffn_2 = nn.Dense(units, flatten=False)

    def forward(self, X):
        return self.ffn_2(self.ffn_1(X))


##### Add and Norm #####
layer = nn.LayerNorm()
layer.initialize()
batch = nn.BatchNorm()
batch.initialize()
X = nd.array([[1,2],[2,3]])
# compute mean and variance from X in the training mode.
with autograd.record():
    print('layer norm:',layer(X), '\nbatch norm:', batch(X))

###### AddNorm Networks  ######
# 输入：X:词向量(atten): NxTxD  Y: attention的输出v: NxTxD 或 feed_forward的输出 NxTxD
# 输出：NxTxD
class AddNorm(nn.Block):
    def __init__(self, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm()

    def forward(self, X, Y):
        return self.norm(self.dropout(Y) + X)


##### Positional Encoding #####
# 输入：X: NxTxD
# 输出：NxTxD  加上了位置编码  偶数对应sin，奇数索引对应cos
class PositionalEncoding(nn.Block):
    def __init__(self, units, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P, max_len来充当T吗
        # P: (1, max_len, D)
        self.P = nd.zeros((1, max_len, units))
        # X: (max_len, D/2)
        X = nd.arange(0, max_len).reshape((-1,1)) / nd.power(10000, nd.arange(0, units, 2) / units)
        self.P[:, :, 0::2] = nd.sin(X)  # 从0开始间隔2填充P 如0 2 4 偶数序列对应sin
        self.P[:, :, 1::2] = nd.cos(X)  # 从1开始间隔2填充P 如1 3 5 奇数序列对应cos

    # 此X不同于__init__中的X
    def forward(self, X):
        # 1, T, D 同X
        X = X + self.P[:, :X.shape[1], :].as_in_context(X.context)
        return self.dropout(X)


##### Encoder #####
# 输入：NxTxD  和  (T,)   在atten中会对 X和valid_length都进行维度转换后再mask
# 输出：NxTxD  权重后的v再全连接层输出
class EncoderBlock(nn.Block):
    def __init__(self, units, hidden_size, num_heads, dropout, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(units, num_heads, dropout)
        self.add_1 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(units, hidden_size)
        self.add_2 = AddNorm(dropout)

    def forward(self, X, valid_length):
        Y = self.add_1(X, self.attention(X, X, X, valid_length))    # add & norm  norm(X + Y)
        return self.add_2(Y, self.ffn(Y))


##### Transformer Encoder #####
# 输入：NxT  个索引 和  (T,)
# 输出：NxTxD  权重后的v再全连接层输出
class TransformerEncoder(d2l.Encoder):
    # units=32 hidden_size=64 num_heads=4 num_layers=2
    def __init__(self, vocab_size, units, hidden_size, num_heads, num_layers, dropout, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.units = units
        self.embed = nn.Embedding(input_dim=vocab_size, output_dim=units)
        self.pos_encoding = PositionalEncoding(units, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add(
                EncoderBlock(units, hidden_size, num_heads, dropout))

    def forward(self, X, valid_length, *args):
        # NxTxD * 根号D  这里乘上根号D是为什么
        X = self.pos_encoding(self.embed(X) * math.sqrt(self.units))
        for blk in self.blks:
            X = blk(X, valid_length)
        return X


##### Decoder #####
# 输入：NxTxD  和  encode output state: NxTxD
# 输出：NxTxD  权重后的v再全连接层输出
class DecoderBlock(nn.Block):
    # i means it's the i-th block in the decoder
    def __init__(self, units, hidden_size, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention_1 = MultiHeadAttention(units, num_heads, dropout)
        self.add_1 = AddNorm(dropout)
        self.attention_2 = MultiHeadAttention(units, num_heads, dropout)
        self.add_2 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(units, hidden_size)
        self.add_3 = AddNorm(dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lengh = state[0], state[1]
        # state[2][i] contains the past queries for this block
        # print(state[2])
        if state[2][self.i] is None:
            key_values = X  # 每块self-atten起始直接将X作为kv输入
        else:               # 每块self-atten下一次key_value就是连结
            key_values = nd.concat(state[2][self.i], X, dim=1)
        state[2][self.i] = key_values
        if autograd.is_training():
            batch_size, seq_len, _ = X.shape
            # shape: (batch_size, seq_len), the values in the j-th column
            # are j+1
            # 细节点啊，解码器通过valid_length来控制后面的词输出概率为0，只注意先前被翻译的
            valid_length = nd.arange(1, seq_len+1, ctx=X.context).tile((batch_size, 1))
            # print("valid_length: ", valid_length)
        else:
            valid_length = None

        X2 = self.attention_1(X, key_values, key_values, valid_length)
        Y = self.add_1(X, X2)   # add norm 后，一个局部残差结构才结束，意味着产生一个新的输入（相对于下一个局部残差）
        Y2 = self.attention_2(Y, enc_outputs, enc_outputs, enc_valid_lengh)
        Z = self.add_2(Y, Y2)
        return self.add_3(Z, self.ffn(Z)), state


##### Transformer Decoder #####
# 输入：NxT  和  decode state: 1.encoder output: NxTxD 2.valid_len 3.num_layer个[None]
# 输出：NxTxV  decoder block后是NxTxD 再全连接层输出NxTxV
class TransformerDecoder(d2l.Decoder):
    def __init__(self, vocab_size, units, hidden_size, num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, units)
        self.pos_encoding = PositionalEncoding(units, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add(DecoderBlock(units, hidden_size, num_heads, dropout, i))
        self.dense = nn.Dense(vocab_size, flatten=False)

    # def init_state(self, enc_outputs, env_valid_lengh, *args):
    # 解码器初始化得到的参数作为下面forward函数中的state
    def init_state(self, enc_outputs, env_valid_lengh, *args):
        return [enc_outputs, env_valid_lengh, [None]*self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embed(X) * math.sqrt(self.units))
        for blk in self.blks:
            X, state = blk(X, state)
        return self.dense(X), state

#################### Training ####################
import collections, io
from mxnet.contrib import text
from mxnet.gluon import data as gdata, nn


# 将一个序列中所有的词记录在all_tokens中以便之后构造词典，然后在该序列后面添加PAD直到序列
# 长度变为max_seq_len，然后将序列保存在all_seqs中
def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    all_tokens.extend(seq_tokens)
    # 句末加上EOS和补齐用的PAD，使序列长度相等
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)

# 使用所有的词来构造词典。并将所有序列中的词变换为词索引后构造NDArray实例
def build_en_data(all_tokens, all_seqs):
    vocab = text.vocab.Vocabulary(collections.Counter(all_tokens), reserved_tokens=[PAD, BOS, EOS])
    indices = [vocab.to_indices(seq) for seq in all_seqs]
    # print(type(indices), type(vocab.to_indices(PAD)))
    # print(indices != vocab.to_indices(PAD))
    valid_len = (nd.array(indices) != vocab.to_indices(PAD)).sum(axis=1)
    return vocab, nd.array(indices), valid_len

def build_ch_data(all_tokens, all_seqs):
    vocab = text.vocab.Vocabulary(collections.Counter(all_tokens), reserved_tokens=[PAD, BOS, EOS])
    indices = [vocab.to_indices(seq) for seq in all_seqs]
    # print(type(indices), type(vocab.to_indices(PAD)))
    # print(indices != vocab.to_indices(PAD))
    valid_len = (nd.array(indices) != vocab.to_indices(PAD)).sum(axis=1)
    return vocab, nd.array(indices), valid_len
# print(nd.array([1, 1, 0]) != 0)

# def read_data(max_seq_len):
    # in和out分别是input和output的缩写
import jieba
in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
data_path = 'Data/NLP2/translation/cmn.txt'
num_line = 0
with io.open(data_path,encoding='utf-8') as f:
    num_line += 1
    if num_line <= num_examples:
        lines = f.readlines()
for line in lines:
    in_seq, out_seq = line.rstrip().lower().split('\t')
    out_seq = out_seq.translate(str.maketrans("，！？。", "    ")).strip()
    in_seq_tokens, out_seq_tokens = in_seq.split(' '), jieba.lcut(out_seq.replace(" ", ""))
    # in_seq_tokens, out_seq_tokens = in_seq.split(' '), out_seq.split(' ')
    if max(len(in_seq_tokens), len(out_seq_tokens)) > max_seq_len - 1:
        continue  # 如果加上EOS后长于max_seq_len，则忽略掉此样本
    process_one_seq(in_seq_tokens, in_tokens, in_seqs, max_seq_len)
    process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)
    # in_data: 20x7  max_seq_len = 7  20 为行数
src_vocab, in_data, in_valid_length = build_en_data(in_tokens, in_seqs)
tgt_vocab, out_data, out_valid_length = build_ch_data(out_tokens, out_seqs)
dataset = gdata.ArrayDataset(in_data, in_valid_length, out_data, out_valid_length)
    # return in_vocab, out_vocab, in_data, out_data, gdata.ArrayDataset(in_data, in_valid_length, out_data, out_valid_length)


# in_data, out_data输出了，只为了测试
# src_vocab, tgt_vocab, in_data, out_data, dataset = read_data(max_seq_len)
train_iter = gdata.DataLoader(dataset, batch_size)
# train_iter: 1001x10,  1001  1001x10,  1001  后面的表示每个序列的有效长度
# src_vocab, tgt_vocab, train_iter = d2l.load_data_nmt(batch_size, num_steps, num_examples)

# units 是embedding以及feed-forward输出层也即self-atten输出维度，
# embed_size是起始输入self-atten的维度，论文中简单起见，同units。num_hiddens是两层feed-forwd中隐藏层的维度。
# num_layers 表示有多少个self-atten
encoder = TransformerEncoder(len(src_vocab), units, num_hiddens, num_heads, num_layers, dropout)
decoder = TransformerDecoder(len(tgt_vocab), units, num_hiddens, num_heads, num_layers, dropout)
# model 输入enc_X, dec_X
model = d2l.EncoderDecoder(encoder, decoder)
# model(X, Y_input, X_vlen, Y_vlen)
d2l.train_s2s_ch8(model, train_iter, lr, num_epochs, ctx)



##### 预测  ######
# transtions = ['Go .', "Join us .", "I'm OK .", 'I won !']
# for line in transtions:
#     # line = "I'm OK ."
#     # in_seq, out_seq = line.rstrip().split('\t')
#     trans_seq_tokens = line.lower().split(' ')  #
#     # if len(trans_seq_tokens) > max_seq_len - 1:
#     #    continue  # 如果加上EOS后长于max_seq_len，则忽略掉此样本
#     trans_index = src_vocab.to_indices(trans_seq_tokens)
#
#     # seq = [line.lower().split(' ')]
#     # tran_tokens = src_vocab.to_indices(seq)
#     enc_valid_length = nd.array([len(trans_index)], ctx=ctx)
#     trans_indexes = d2l.trim_pad(trans_index, max_seq_len, src_vocab.to_indices(PAD)) # num_steps -> max_seq_len
#     # 1xT
#     enc_X = nd.array(trans_indexes, ctx=ctx)
#     # add the batch_size dimension.
#     # 1xTxD 先处理输入序列，将编码器的输出作为一组kv
#     enc_outputs = model.encoder(enc_X.expand_dims(axis=0), enc_valid_length)
#     # [enc_outputs, enc_valid_length, [None]*num_layer]
#     dec_state = model.decoder.init_state(enc_outputs, enc_valid_length)
#     # 1x1
#     dec_X = nd.array([tgt_vocab.to_indices(BOS)], ctx=ctx).expand_dims(axis=0)
#     # dec_X = nd.array([trans_index[0]], ctx=ctx).expand_dims(axis=0)
#     predict_tokens = []
#     for _ in range(max_seq_len):
#         # 1x1x5116 1x3x32
#         Y, dec_state = model.decoder(dec_X, dec_state)
#         # The token with highest score is used as the next time step input.
#         dec_X = Y.argmax(axis=2)    # 降指定的一维
#         py = dec_X.squeeze(axis=0).astype('int32').asscalar()
#         if py == tgt_vocab.to_indices(EOS):
#             break
#         predict_tokens.append(py)
#         print(' '.join(tgt_vocab.to_tokens(predict_tokens)))
#
# tgt_vocab.to_tokens([24])
src_sentence = "I'm OK ."
src_tokens = src_vocab.to_indices([x.lower() for x in src_sentence.split(' ')])
enc_valid_length = nd.array([len(src_tokens)], ctx=ctx)
src_tokens = d2l.trim_pad(src_tokens, max_seq_len, src_vocab.to_indices('<pad>'))
enc_X = nd.array(src_tokens, ctx=ctx)
# add the batch_size dimension.
enc_outputs = model.encoder(enc_X.expand_dims(axis=0), enc_valid_length)
dec_state = model.decoder.init_state(enc_outputs, enc_valid_length)
dec_X = nd.array([tgt_vocab.to_indices('<bos>')], ctx=ctx).expand_dims(axis=0)
predict_indexs = []
for _ in range(max_seq_len):
    Y, dec_state = model.decoder(dec_X, dec_state)
    # The token with highest score is used as the next time step input.
    dec_X = Y.argmax(axis=2)
    py = dec_X.squeeze(axis=0).astype('int32').asscalar()
    if py == tgt_vocab.to_indices('<eos>'):
        break
    predict_indexs.append(py)
    # tgt_vocab.to_tokens(predict_indexs) 出错， 但tgt_vocab.to_tokens([20]) 又是没问题的
    predict_tokens = [tgt_vocab.idx_to_token[index]  for index in predict_indexs]
    res = ' '.join(predict_tokens)
    print(res)

# for sentence in ['Go .', "Join us .", "I'm OK .", 'I won !']:
#     print(sentence + ' => ' + d2l.predict_s2s_ch8(model, sentence, src_vocab, tgt_vocab, max_seq_len, ctx))
# print(predict_indexs)
# test = [tgt_vocab.idx_to_token[index]  for index in predict_indexs]
# tgt_vocab.to_tokens([20])
