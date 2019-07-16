from Data.DataExtract import load_robert_frost_soseos, load_glove6B
from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, SGD
import keras.backend as K

import  numpy as np
import matplotlib.pyplot as plt

if len(K.tensorflow_backend._get_available_gpus()) > 0:
	from keras.layers import CuDNNLSTM as LSTM
	from keras.layers import CuDNNGRU as GRU

# 参数
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 3000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 200
LATENT_DIM = 25

# 使用训练好的glove当做词向量
word2vec = load_glove6B(dimension=50)
input_sentences, output_sentences = load_robert_frost_soseos()
all_sentences = input_sentences + output_sentences

# Tokensizer: 1.将句子分隔为单词；2.将单词转换为索引
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer.fit_on_texts(all_sentences)
input_sentences = tokenizer.texts_to_sequences(input_sentences)
output_sentences = tokenizer.texts_to_sequences(output_sentences)

word2index = tokenizer.word_index
# +1的原因：后者小时，word2index序号从1开始，对于word_embedding序号从0开始，0是必有的
words_num = min(MAX_VOCAB_SIZE, len(word2index) + 1)

# find max seq length
max_sequence_length_from_data = max(len(s) for s in input_sentences)
max_sequence_length = min(max_sequence_length_from_data, MAX_SEQUENCE_LENGTH)

# 输入序列与输出序列目标， 后续不足的填0
input_sequences = pad_sequences(input_sentences, maxlen=max_sequence_length, padding='post')
output_sequences = pad_sequences(output_sentences, maxlen=max_sequence_length, padding='post')

# 根据最大词量约束，重构词向量空间
embedding = np.zeros(shape=(words_num, EMBEDDING_DIM))
for word, word_index in word2index.items():
	if word_index < MAX_VOCAB_SIZE:
		word_vector = word2vec.get(word)
		if word_vector is not None:
			# 没找到词的词向量是原点，即都为0
			embedding[word_index] = word_vector

# 对目标也进行序列
ont_hot_output = np.zeros(shape=(len(input_sentences), max_sequence_length, words_num))
for i, output_sequence in enumerate(output_sequences):
	for t, word_index in enumerate(output_sequence):
		if word_index > 0:
			ont_hot_output[i, t, word_index] = 1

# 将调整过的glove中的词向量初始化乘Embedding layer
embedding_layer = Embedding(input_dim=words_num,
							output_dim=EMBEDDING_DIM,
							weights=[embedding],
							# trainable=False
)


# 建立模型
# N, max_sequence_length <-> NxT   二维
input_ = Input(shape=(max_sequence_length, ))
initial_h = Input(shape=(LATENT_DIM, ))
initial_c = Input(shape=(LATENT_DIM, ))
# NxT -> NxTxD
x = embedding_layer(input_)
lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
# NxTxD -> NxTxM
x, _, _ = lstm(x, initial_state=[initial_h, initial_c])
dense = Dense(words_num, activation='softmax')
# NxM -> N, word_num
output = dense(x)

model = Model([input_, initial_h, initial_c], output)
model.compile(
	loss='categorical_crossentropy',
	optimizer=Adam(lr=0.01),
	metrics=['accuracy']
)

# 对于训练语言模型的输入状态都初始化为0
z = np.zeros(shape=(len(input_sequences), LATENT_DIM))
r = model.fit(
	x=[input_sequences, z, z],	# NxT,  NxM
	y=ont_hot_output,
	batch_size=BATCH_SIZE,
	epochs=EPOCHS,
	validation_split=VALIDATION_SPLIT
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


# 生成模型，也是采样模型
# 输入是每个词，下个词的生成依赖上次生成的词作为输入
# 1, 1
input2 = Input(shape=(1, ))
# 1x1xD
x = embedding_layer(input2)
# 1x1xM		M: Laten_dim   h: 1xM c: 1xM
x, h, c = lstm(x, initial_state=[initial_h, initial_c])
# 1, 1, word_num
output2 = dense(x)

sampling_model = Model([input2, initial_h, initial_c], [output2, h, c])

index2word = {v: k for k, v in word2index.items()}

np_input = np.array([[ word2index['<sos>'] ]])
h = np.zeros((1, LATENT_DIM))
c = np.zeros((1, LATENT_DIM))

eos_index = word2index['<eos>']

output_sentence = []

for _ in range(max_sequence_length):
	# 1,1,word_num   1,M   1,M
	o, h, c = sampling_model.predict([np_input, h, c])
	# 3000,
	probs = o[0, 0]
	if np.argmax(probs) == 0:
		print('wtf')
	probs[0] = 0
	probs /= probs.sum()	# 均匀分布
	index = np.random.choice(len(probs), p=probs)	# 取样, 不是取概率最大的一个词
	if index == eos_index:
		break

	output_sentence.append(index2word.get(index, '<WTF %s>' % index))

	# 下一个词的预测的输入是前一个输出的词
	np_input[0, 0] = index
