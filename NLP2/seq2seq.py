from Data.DataExtract import load_translation, load_glove6B
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt

if len(K.tensorflow_backend._get_available_gpus()) > 0:
	from keras.layers import CuDNNLSTM as LSTM
	from keras.layers import CuDNNGRU as GRU

BATCH_SIZE = 64  # Batch size for training.
EPOCHS = 100  # Number of epochs to train for.
LATENT_DIM = 256  # Latent dimensionality of the encoding space.
NUM_SAMPLES = 10000  # Number of samples to train on.
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

word2vec = load_glove6B(EMBEDDING_DIM)
# 翻译的输入句子， 翻译后输出句子前后分别加标志成为inputs与outputs
input_texts, translation_inputs, translation_outputs = load_translation(sample_num=NUM_SAMPLES)

# 将输入tokennize
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
# 获得词与索引的映射
word2index_inputs = tokenizer_inputs.word_index
# 获取输入句子中的最大序列长度
max_len_input = max(len(s) for s in input_sequences)

# 将翻译句子tokennize
tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(translation_inputs + translation_outputs)
translation_sequences_outputs = tokenizer_outputs.texts_to_sequences(translation_outputs)
translation_sequences_inputs = tokenizer_outputs.texts_to_sequences(translation_inputs)
# 获得词与索引的映射
word2index_outputs = tokenizer_outputs.word_index
# 翻译句子的独立词数
num_words_translation = len(word2index_outputs) + 1
# 获取翻译句子中的最大序列长度
max_len_translation = max(len(s) for s in translation_sequences_inputs)

# 定长序列
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
decoder_inputs = pad_sequences(translation_sequences_inputs, maxlen=max_len_translation, padding='post')
decoder_outputs = pad_sequences(translation_sequences_outputs, maxlen=max_len_translation, padding='post')


# 根据最大词量，重组embedding  +1是因为word2index从1开始，embedding从0开始
num_words = min(MAX_NUM_WORDS, len(word2index_inputs) + 1)
embedding = np.zeros(shape=(num_words, EMBEDDING_DIM))
for word, word_index in word2index_inputs.items():
	if word_index < MAX_NUM_WORDS:
		word_vector = word2vec.get(word)
		if word_vector is not None:
			embedding[word_index] = word_vector

embedding_layer = Embedding(
	input_dim=num_words,
	output_dim=EMBEDDING_DIM,
	weights=[embedding],
	input_length=max_len_input,
	# trainable=True
)

decoder_outputs_one_hot = np.zeros(shape=(len(input_texts), max_len_translation, num_words_translation), dtype='float32')

# decoder_ouput即翻译的句子
for n, decoder_output in enumerate(decoder_outputs):
	for t, word in enumerate(decoder_output):
		decoder_outputs_one_hot[n, t, word] = 1



# 建立模型
encoder_inputs_placehoder = Input(shape=(max_len_input, ))
x = embedding_layer(encoder_inputs_placehoder)
# dropout 在gpu中不能用
encoder = LSTM(units=LATENT_DIM,
			   return_state=True,
			   # dropout=0.5
)
encoder_outputs, h, c = encoder(x)
# encoder_outputs, h = encoder(x) #gru
# 将编码器的状态与记忆单元作为解码器的初始状态输入
encoder_states = [h,  c]
# encoder_states = [state_h] # gru

# 解码器
decoder_inputs_placehoder = Input(shape=(max_len_translation, ))
decoder_embedding = Embedding(num_words_translation, LATENT_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placehoder)

decoder_lstm = LSTM(units=LATENT_DIM,
					return_sequences=True,
					return_state=True
)

# 将编码器的状态与记忆单元作为解码器的初始状态输入
decoder_outputs, _, _ = decoder_lstm(
	units=decoder_inputs_x,
	initial_state=encoder_states
)
# decoder_outputs, _ = decoder_gru(
#   decoder_inputs_x,
#   initial_state=encoder_states
# )
decoder_dense = Dense(num_words_translation, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs_placehoder, decoder_inputs_placehoder], decoder_outputs)

# 编译
model.compile(
	optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy']
)
r = model.fit(
  [encoder_inputs, decoder_inputs], decoder_outputs_one_hot,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=0.2,
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

# 保存模型
model.save('seq2seq.h5')


##### 做预测 #####
encoder_model = Model(encoder_inputs_placehoder, encoder_states)

decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

decoder_outputs, h, c = decoder_lstm(
	units=decoder_inputs_single_x,
	initial_state=decoder_states_inputs
)
# decoder_outputs, state_h = decoder_lstm(
#   decoder_inputs_single_x,
#   initial_state=decoder_states_inputs
# ) #gru
decoder_states = [h, c]
# decoder_states = [h] # gru
decoder_outputs = decoder_dense(decoder_outputs)


# 采样model
# inputs: y(t-1), h(t-1), c(t-1)
# outputs: y(t), h(t), c(t)
decoder_model = Model(
	[decoder_inputs_single] + decoder_states_inputs,
	[decoder_outputs] + decoder_states
)

index2word_eng = {v:k for k, v in word2index_inputs.items()}
index2word_trans = {v:k for k, v in word2index_outputs.items()}

i = np.random.choice(len(input_texts))
input_seq = encoder_inputs[i:i + 1]
# Encode the input as state vectors.
states_value = encoder_model.predict(input_seq)
# Generate empty target sequence of length 1.
target_seq = np.zeros((1, 1))

# Populate the first character of target sequence with the start character.
# NOTE: tokenizer lower-cases all words
target_seq[0, 0] = word2index_outputs['<sos>']

# if we get this we break
eos = word2index_outputs['<eos>']

# Create the translation
output_sentence = []
for _ in range(max_len_translation):
	output_tokens, h, c = decoder_model.predict(
		[target_seq] + states_value
	)
	# output_tokens, h = decoder_model.predict(
	#     [target_seq] + states_value
	# ) # gru

	# Get next word
	index = np.argmax(output_tokens[0, 0, :])

	# End sentence of EOS
	if eos == index:
		break
	word = ''
	if index > 0:
		word = index2word_trans[index]
		output_sentence.append(word)

	# Update the decoder input
	# which is just the word just generated
	target_seq[0, 0] = index
	# Update states
	states_value = [h, c]