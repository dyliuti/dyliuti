from keras.models import Model
from keras.layers import Input, LSTM, GRU, Bidirectional
import numpy as np

import keras.backend as K
if len(K.tensorflow_backend._get_available_gpus()) > 0:
  from keras.layers import CuDNNLSTM as LSTM
  from keras.layers import CuDNNGRU as GRU


T = 8
D = 2
H = 3

X = np.random.randn(1, T, D)

input_ = Input(shape=(T, D))
rnn = Bidirectional(LSTM(H, return_state=True, return_sequences=True))
# 是否返回 h、c 又因为有Bidirectional 有了h2、c2       return_sequences 对输出o有影响，就有没中间状态
# rnn = Bidirectional(LSTM(M, return_state=True, return_sequences=False))
x = rnn(input_)

model = Model(inputs=input_, outputs=x)
# h1, c1是forward中的最后一个hidden state（序列的末尾）和
# h2, c2是backward中的最后一个hidden state（序列的起始）和
# o = model.predict(X)
o, h1, c1, h2, c2 = model.predict(X)
print("o:", o)
print("o.shape:", o.shape)		# 1 8 6
print("h1:", h1)				# 1 3
print("c1:", c1)				# 1 3
print("h2:", h2)
print("c2:", c2)
