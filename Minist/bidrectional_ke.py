from Data.DataExtract import load_minist_csv
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Bidirectional, GlobalAveragePooling1D, Lambda, Concatenate, Dense
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

# 13s 467us/step - loss: 0.0310 - acc: 0.9912 - val_loss: 0.0764 - val_acc: 0.9771

if len(K.tensorflow_backend._get_available_gpus()) > 0:
	from keras.layers import CuDNNLSTM as LSTM
	from keras.layers import CuDNNGRU as GRU


X_train, X_test, Y_train, Y_test = load_minist_csv(pca=False)
D = np.sqrt(np.shape(X_train)[1]).astype(np.int32)	# 28
M = 20	# hidden layer size
X_train = np.reshape(X_train, newshape=(-1, D, D))
X_test = np.reshape(X_test, newshape=(-1, D, D))


# 图像从上到下，纵轴当做时间序列。
# NxDxD
input_ = Input(shape=(D, D))
# NxDxD MxM -> NxDx2M return_sequences=True.若return_sequences=False 只返回正序与逆序的输出-> Nx2M
rnn1 = Bidirectional(LSTM(M, return_sequences=True))
x1 = rnn1(inputs=input_)
# Nx2M	# 默认channel last, input: (batch_size, steps, features) -> output: (batch_size, features)
# 经过GlobalAveragePooling1D 后，也可以用机器学习算法进行分类了
x1_ = GlobalAveragePooling1D()(x1)

# 图像从左到右，横轴当做时间序列
rnn2 = Bidirectional(LSTM(M, return_sequences=True))
# lamda对象函数, 将1、2轴进行转置
permutor = Lambda(lambda t: K.permute_dimensions(t, pattern=(0, 2, 1)))
x2 = permutor(input_)
# NxDx2M
x2 = rnn2(x2)
# Nx2M
x2_ = GlobalAveragePooling1D()(x2)

# 将输出和2为1  axis: Axis along which to concatenate.
concatentator = Concatenate(axis=1)
x = concatentator([x1_, x2_])

# 将Dense作为分类器
output = Dense(units=10, activation='softmax')(x)

model = Model(inputs=input_, output=output)

model.compile(
	loss='sparse_categorical_crossentropy',
	optimizer='adam',
	metrics=['accuracy']
)

r = model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_split=0.2)

# 测试集与验证集的损失
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# 测试集与验证集的准确率
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()