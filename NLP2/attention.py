from Data.DataExtract import load_translation_fenci, load_glove6B
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, \
	Bidirectional, RepeatVector, Concatenate, Dot, Lambda
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
import keras

# 1000个词，epochs=100 中英文去停顿标点，中文短语形式 loss: 0.0193 - acc: 0.9886 - val_loss: 3.0284 - val_acc: 0.6933
# 1000个词，epochs=100 中英文去停顿标点，中文词形式   loss: 0.0185 - acc: 0.9889 - val_loss: 3.2927 - val_acc: 0.6482
# 这里还可以把英文单词再进行拆分的，形成局部单词

# 第二个做attention示例不错， ["i'm", 'thrilled'] ['我', '激', '動', '不', '已']
# 不已的权重都很平均，这是不是意味这些词可以作为去停用词。
# [[0.99383545 0.00409239 0.00107628 0.00099594]
#  [0.50675154 0.27643755 0.13042809 0.08638281]
#  [0.45728102 0.3588085  0.09860806 0.08530243]
#  [0.27644026 0.26376322 0.24046768 0.21932882]
#  [0.25044706 0.25030664 0.24986543 0.24938092]]

# 输入句子： ['he', 'is', 'tall']
# 翻译得到句子： ['他', '高']
# [[9.9938273e-01 4.6381936e-04 9.6236072e-05 5.7242461e-05]
#  [5.1293737e-01 3.5182342e-01 7.2982356e-02 6.2256809e-02]]

# 输入句子： ["i'm", 'curious']
# 翻译得到句子： ['我', '很', '好', '奇']
# [[9.9718082e-01 1.4917660e-03 7.3657365e-04 5.9088302e-04]
#  [9.3813312e-01 4.6988510e-02 7.7440026e-03 7.1343984e-03]
#  [4.9452606e-01 3.0531138e-01 1.0537410e-01 9.4788492e-02]
#  [3.4199092e-01 3.0511385e-01 1.8240550e-01 1.7048974e-01]]

import numpy as np
import matplotlib.pyplot as plt

if len(K.tensorflow_backend._get_available_gpus()) > 0:
	from keras.layers import CuDNNLSTM as LSTM
	from keras.layers import CuDNNGRU as GRU

# jpa.txt: # loss: 0.1806 - acc: 0.8924 - val_loss: 4.6954 - val_acc: 0.6752
# twitter_chat.txt: 100 epochs # 59s 9ms/step - loss: 0.0833 - acc: 0.9952 - val_loss: 2.8875 - val_acc: 0.7297

BATCH_SIZE = 64  # Batch size for training.
EPOCHS = 100
LATENT_DIM = 256
LATENT_DIM_DECODER = 256 # 较seq2seq多出的参数
NUM_SAMPLES = 1000  # 训练样本句子数
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100


word2vec = load_glove6B(EMBEDDING_DIM)
# 翻译的输入句子， 翻译后的输出句子前后分别加标志成为inputs与outputs
# translation_inputs 与 translation_outputs 分别作为 Teacher Forcing 的输入与输出
file_name="cmn.txt"
file_path="Data/NLP2/translation"
input_texts, translation_inputs, translation_outputs = load_translation_fenci(
	file_name=file_name,
	file_path=file_path,
	reserve_punctuation=False,
	word=True,
	sample_num=NUM_SAMPLES
)
# input_texts, translation_inputs, translation_outputs = load_translation(file_name='twitter_chat.txt', sample_num=NUM_SAMPLES)
# 对于jpn.txt 总共44917行 设置10000可以。但对于twitter_chat.txt，总共就8490行，设置10000会导致x,y样本不一样，兼容下
NUM_SAMPLES = min(NUM_SAMPLES, len(input_texts))

# 将输入tokennize
# filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
# lower=True,
# split=' ',
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
print(input_sequences[768])
# 获得词与索引的映射
word2index_inputs = tokenizer_inputs.word_index
print([tokenizer_inputs.index_word[index] for index in input_sequences[768]])
# 获取输入句子中的最大序列长度
max_len_input = max(len(s) for s in input_sequences)

# 将翻译句子tokennize
tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(translation_inputs + translation_outputs)
translation_sequences_outputs = tokenizer_outputs.texts_to_sequences(translation_outputs)
translation_sequences_inputs = tokenizer_outputs.texts_to_sequences(translation_inputs)
print([tokenizer_outputs.index_word[index] for index in translation_sequences_outputs[768]])
# 获得词与索引的映射
word2index_outputs = tokenizer_outputs.word_index
# 翻译句子的独立词数
num_words_translation = len(word2index_outputs) + 1
# 获取翻译句子中的最大序列长度
max_len_translation = max(len(s) for s in translation_sequences_inputs)

# 定长序列 N_en x T_en   N_de x T_de
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input, padding='post')
decoder_inputs = pad_sequences(translation_sequences_inputs, maxlen=max_len_translation, padding='post')
decoder_outputs = pad_sequences(translation_sequences_outputs, maxlen=max_len_translation, padding='post')


# 根据最大词量，重组embedding  +1是因为word2index索引从1开始，embedding索引从0开始。即要复制到的embedding比word2index多一例，索引才对应
num_words = min(MAX_NUM_WORDS, len(word2index_inputs) + 1)
embedding = np.zeros(shape=(num_words, EMBEDDING_DIM))
for word, word_index in word2index_inputs.items():
	if word_index < MAX_NUM_WORDS:
		word_vector = word2vec.get(word)	# 获取词对应的向量
		if word_vector is not None:
			embedding[word_index] = word_vector

# 序列化的输入中获得每个序列的词向量  即加个维度 D
embedding_layer = Embedding(
	input_dim=num_words,	# 独立词个数，也即是embedding中词vector的个数
	output_dim=EMBEDDING_DIM,
	weights=[embedding],	# 初始化参数
	# 后面连接 `Flatten` then `Dense` 必要的参数，没这参数，`Flatten` then `Dense`的shape就无法计算
	# 即要求输入序列长度是固定的
	input_length=max_len_input,
	# trainable=True
)
# embedding_layer.set_weights(embedding)

# N_de x T_de x D_de
decoder_outputs_one_hot = np.zeros(shape=(len(input_texts), max_len_translation, num_words_translation), dtype='float32')

# decoder_ouput即翻译的句子
for n, decoder_output in enumerate(decoder_outputs):
	for t, word_index in enumerate(decoder_output):
		decoder_outputs_one_hot[n, t, word_index] = 1


######## 编码器 #########
# N_en x T_en  		N_en: batch_size  T_en: 输入句子长度
encoder_inputs_placehoder = Input(shape=(max_len_input, ))
# N_en x T_en x D_en
x = embedding_layer(encoder_inputs_placehoder)
# N_en x T_en x 2H_en
encoder = Bidirectional(LSTM(units=LATENT_DIM,					# 将LSTM改为双向LSTM
		 					 return_sequences=True,             # Attention关注所有隐藏状态
			    		     # return_state=True,
))
# N_en x T x 2H_en    因为 return_sequences=True 所以多了T
encoder_outputs = encoder(x)		# return_state为False 无 h1,c1,h2,c2
# encoder_outputs, h = encoder(x) #gru



######## Attention #########
# 输入参数x的shape is N x T x (2*H_en + H_de)
# 对每个T上的 1xD进行softmax运算，
def softmax_over_time(x):
	assert(K.ndim(x) > 2)
	# NxTx(2*H_en + H_de)
	e = K.exp(x - K.max(x, axis=1, keepdims=True))
	# Nx1x(2*H_en + H_de)
	s = K.sum(e, axis=1, keepdims=True)
	# 返回： # NxTx(2*H_en + H_de)
	return e / s

# 输入：N x D -> 输出：N x T x D
attn_repeat_layer = RepeatVector(max_len_input)
attn_concat_layer = Concatenate(axis=-1)
attn_dense1 = Dense(10, activation='tanh')
attn_dense2 = Dense(1, activation=softmax_over_time)
# 计算加权和 alpha[t] * h[t] t是第1维度的坐标，即纵向量的坐标。x.T.dot(y)
attn_dot = Dot(axes=1) 	# Dot详情测试运算见test_func.py

# h 即是编码器所有序列上的状态 1xTx2H_en
def one_step_attention(h, st_1):
	# h = h(1), ..., h(Tx), 每个h(Tx),N个样本， shape = Nx2H_en  总的h shape: N x Tx x 2H_en
	# st_1 = s(t-1), shape = NxH_de

	# 复制s(t-1) Tx 次  NxM -> NxTxH_de
	st_1 = attn_repeat_layer(st_1)
	# 将所有编码器的输出 与attention 中的前一个状态进行连结
	# N x T x (2*H_en + H_de)
	x = attn_concat_layer([h, st_1])
	# N x T x 10
	x = attn_dense1(x)
	# N x T x 1 先通过Dense(unit==1)将 NxTx10 转换为 N x T x 1, 再对T个数（时间序列上）进行softmax
	# 得到的alphas就是 attention 权重，关注多少
	alphas = attn_dense2(x)	# softmax over time

	# 然后 N x T x 1 和 N x T x 2H_en -> Nx1x2H_en
	# T个单词，每个单词都对应双向lstm的输出状态h(1，2M)，而ht1-ht2m的2M个单元（纵向）随横向的前进不断更新参数
	# 每个词都有h，h代表了词的隐形特征
	# alphas 的每一列(T个坐标,每个值对句子中相应词的权重)
	context = attn_dot([alphas, h])
	return context, alphas


######## Attention 后的解码器 #########
# 解码器输入
# N_de x T_de
decoder_inputs_placehoder = Input(shape=(max_len_translation, ))
# 区别：解码器的embedding维度不再使用编码器隐藏层的维度，当然也可以使用
# N_de x T_de x D_de		LATENT_DIM 改为 EMBEDDING_DIM
decoder_embedding = Embedding(num_words_translation, EMBEDDING_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placehoder)

decoder_lstm = LSTM(
	units=LATENT_DIM_DECODER,
	return_state=True
)
decoder_dense = Dense(num_words_translation, activation='softmax')

# 输入的初始状态将会是赋值为0的 tensor
initial_s = Input(shape=(LATENT_DIM_DECODER, ), name='s0')
initial_c = Input(shape=(LATENT_DIM_DECODER, ), name='c0')
context_last_word_concat_layer = Concatenate(axis=2)

# 像seq2seq，用sos、eos偏置，一次性用tensor进行训练
# 这里需要stpe by step，lstm的输出s与记忆单元c每次输入下一个lstm需要更新，而且每步都要考虑全部序列
s = initial_s
c = initial_c

outputs_ = []
# 1xTxD 每次处理 Nx1xD
for t in range(max_len_translation):  # Ty次
	# 通过attention获取context : 1x1x2H_en
	context, _ = one_step_attention(encoder_outputs, s)
	# 获得一个输入句子
	selector = Lambda(lambda x: x[:, t: t+1])
	# NxTxD -> Nx1xD
	xt = selector(decoder_inputs_x)		# 这里其实是yt-1
	print(xt.shape)
	# 前一个状态的yt-1与context作为t时的输入	# N 1 2M+D
	decoder_lstm_input = context_last_word_concat_layer([context, xt])
	# 更新s、c  	o: N 256   无T是因为return_sequence=False
	o, s, c = decoder_lstm(inputs=decoder_lstm_input,
						   initial_state=[s, c])	# lstm输入的向量维度任意
	# 使用dense layer分类，获得下个词的预测
	decoder_outputs = decoder_dense(o)		# N num_words
	print("decoder_outputs:", decoder_outputs.shape)
	outputs_.append(decoder_outputs)

def stack_and_transponse(x):
	# TxNxD
	x = K.stack(x)
	# NxTxD
	x = K.permute_dimensions(x, pattern=(1, 0, 2))
	return x

# 将转换函数作为一层
stacker = Lambda(stack_and_transponse)
outputs = stacker(outputs_)

# 创建模型
model = Model(inputs=[encoder_inputs_placehoder,
					  decoder_inputs_placehoder,
					  initial_s,
					  initial_c],
			  outputs=outputs
)
#  metrics=['accuracy']
# from keras import metrics
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

init_decoder_state = np.zeros(shape=(NUM_SAMPLES, LATENT_DIM_DECODER)) # 初始化 [s, c]

from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)
n = 0
class my_call_back(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		global n
		self.losses.append(logs.get('loss'))
		n += 1
		print("第%d个batch. %f" % (n, logs.get('loss')))


call_back = my_call_back()
r = model.fit(
	x=[encoder_inputs, decoder_inputs, init_decoder_state, init_decoder_state],
	y=decoder_outputs_one_hot,
	batch_size=BATCH_SIZE,
	epochs=EPOCHS,
	validation_split=0.2,
	callbacks=[tensorboard, call_back]
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()



######## 做预测 #########
# 预测时输入发生了改变，从NxT 变为 1xT
# 编码器是独立的，映射输入序列到 NxTx2M 维隐藏状态
# encoder_inputs -> encoder_outputs(双向h状态)
encoder_model = Model(inputs=encoder_inputs_placehoder,
					  outputs=encoder_outputs)
# 编码器输出输入经过Bidirection后的状态
encoder_outputs_as_input = Input(shape=(max_len_input, LATENT_DIM * 2,))
# Attention利用编码器全部输出与st-1生成context
context, alphas = one_step_attention(encoder_outputs_as_input, initial_s)
# yt-1
decoder_inputs_single = Input(shape=(1, ))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

decoder_lstm_input = context_last_word_concat_layer([context, decoder_inputs_single_x])
# context   yt-1  -> 预测 1x1x 2M+D
o, s, c = decoder_lstm(inputs=decoder_lstm_input,
					   initial_state=[initial_s, initial_c])
# 1x1x num_words
decoder_outputs = decoder_dense(o)

decoder_model = Model(inputs=[decoder_inputs_single,
							  encoder_outputs_as_input,
							  initial_s,
							  initial_c],
					  outputs=[decoder_outputs, s, c, alphas, decoder_inputs_single_x]
)

# 为了获取真实的词
index2word_eng = {v: k for k, v in word2index_inputs.items()}
index2word_trans = {v: k for k, v in word2index_outputs.items()}

# 为什么预测时将编码器与解码器分开，有两个model，而训练模型时是从头到尾一步到位，就只有一个model？
# 因为预测时，编码器是独立于解码器的，编码器只需一次产生Bi-lstm状态就好。分开就不用计算T次了。
# 而训练模型时虽然也是采用上述方式计算，但需要更新参数，所以放在一个模型中?
def get_translation(input_seq):
	# 将输入句子序列编码映射输入序列到 NxTx2M 维隐藏状态, Bi-lstm
	enc_output = encoder_model.predict(input_seq)
	# 翻译目标序列长度为1，即产生词    yt-1
	translation_input_word = np.zeros((1, 1))
	# 输入一个随机句子，进行翻译，翻译的句子开始标志是sos。解码器一次只能产生一个词
	# tokenizer 的作用之一就是全部变为小写单词，而不是大写
	translation_input_word[0, 0] = word2index_outputs['<sos>']
	# 如果预测到eos就结束
	eos_index = word2index_outputs['<eos>']

	s = np.zeros((1, LATENT_DIM_DECODER))
	c = np.zeros((1, LATENT_DIM_DECODER))

	# 产生翻译,
	output_sentence = []
	params = []
	embeddings = []
	for _ in range(max_len_translation):
		output_tokens, s, c, alpha, embedding_token = decoder_model.predict(
			x=[translation_input_word, enc_output, s, c]
		)

		# 得到下一个预测的词
		index = np.argmax(output_tokens.flatten())   # output_tokens[0, 0, :]
		# 遇到结束符eos，表示句子翻译结束,eos不加入翻译的句子中
		if index == eos_index:
			break
		word = ''
		if index > 0:
			word = index2word_trans[index]
			output_sentence.append(word)
			params.append(alpha)
			embeddings.append(embedding_token)
		# 更新解码器的输入
		translation_input_word[0, 0] = index
		# 更新解码器输入状态
		# states_value = [h, c]

	return output_sentence, params, embeddings

while True:
	# 随机选择一个句子，并对其进行翻译
	i = np.random.choice(len(input_texts))
	input_seq = encoder_inputs[i: i + 1]  # 区别于encoder_inputs[i] shape:(5,)  前者shape是(1, 5)
	input_sentence = [index2word_eng[word_index] for word_index in input_seq[0] if word_index > 0]
	print("输入句子：", input_sentence)
	output_sentence, params, embeddings = get_translation(input_seq)
	params_ = np.array(params).squeeze()
	print("翻译得到句子：", output_sentence)
	print(params_)
	print(embeddings)
	ans = input("Continue? [Y/n]")
	if ans and ans.lower().startswith('n'):
		break

# params_ = np.array(params).squeeze()
# print(params[0])
# print(params[1])
# print(params[])
# 英文单词数：3518  中单单词数 6766
# input_sentence = ['now', 'i', 'need', 'you', 'to', 'leave']
# output_sentence = ['現在', '我', '需要', '你', '離開']
# output_indexs = [word2index_outputs[word] for word in output_sentence]
