from NLP.Common.Util import all_parity_pairs
from keras.models import Sequential
from keras.layers import Dense
# 12位组成 4096种X组合
X, Y = all_parity_pairs(12)
model = Sequential()
# 宽点
model.add(Dense(2048, activation='relu'))
# 深点
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
	loss='binary_crossentropy',
	metrics=['accuracy'],
	optimizer='rmsprop'
)

history = model.fit(X, Y, batch_size=300, epochs=200)
score = model.evaluate(X, Y)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
