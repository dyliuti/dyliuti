from Data.DataExtract import load_ner
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, GRU
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt

MAX_VOCAB_SIZE = 20000
MAX_TAGS = 100

X, Y = load_ner(split_sequence=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
# 将words转换为索引
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

word2index = tokenizer.word_index
vocab_size = min(MAX_VOCAB_SIZE, len(word2index) + 1)

# 将tags转换为索引
tokenizer_tag = Tokenizer(num_words=MAX_TAGS)
tokenizer_tag.fit_on_texts(Y_train)
Y_train = tokenizer_tag.texts_to_sequences(Y_train)
Y_test = tokenizer_tag.texts_to_sequences(Y_test)

tag2index = tokenizer_tag.word_index
tag_size = min(MAX_TAGS, len(tag2index) + 1)

# 找最长的序列， 对齐序列
sequence_length = max(len(x) for x in X_train + X_test)
X_train = pad_sequences(X_train, maxlen=sequence_length)
Y_train = pad_sequences(Y_train, maxlen=sequence_length)
X_test = pad_sequences(X_test, maxlen=sequence_length)
Y_test = pad_sequences(Y_test, maxlen=sequence_length)

# 转换为one-hot 形式
Y_train_onehot = np.zeros(shape=(len(Y_train), sequence_length, tag_size), dtype='float32')
for n, tags in enumerate(Y_train):
	for t, tag in enumerate(tags):
		Y_train_onehot[n, t, tag] = 1

Y_test_onehot = np.zeros(shape=(len(Y_test), sequence_length, tag_size), dtype='float32')
for n, tags in enumerate(Y_test):
	for t, tag in enumerate(tags):
		Y_test_onehot[n, t, tag] = 1

# 参数
epochs = 30
batch_size = 32
hidden_layer_size = 10
embedding_dim = 10

# build the model
input_ = Input(shape=(sequence_length,))
x = Embedding(vocab_size, embedding_dim)(input_)
x = GRU(hidden_layer_size, return_sequences=True)(x)
output = Dense(tag_size, activation='softmax')(x)


model = Model(input_, output)
model.compile(
	loss='categorical_crossentropy',
	optimizer=Adam(lr=1e-2),
	metrics=['accuracy']
)


print('Training model...')
r = model.fit(
	X_train,
	Y_train_onehot,
	batch_size=batch_size,
	epochs=epochs,
	validation_data=(X_test, Y_test_onehot)
)

# plot
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()