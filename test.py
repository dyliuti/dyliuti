import mxnet.ndarray as nd

vocab_size = 5
res = nd.one_hot(nd.array([0, 2]), vocab_size)

def to_onehot(X, size):  # 本函数已保存在d2lzh包中方便以后使用
    return [nd.one_hot(x, size) for x in X.T]

X = nd.arange(10).reshape((2, 5))
inputs = to_onehot(X, vocab_size)
len(inputs), inputs[0].shape

a = X.T.reshape((-1,))	# 5 x 2	-> 10
a_ = to_onehot(a, 8)	# 10 -> 10,1 -> 10 8  class_num, V

import numpy as np
# b = np.arange(10).reshape(2, 5)
b = [nd.array([0, 1, 2, 3, 4]), nd.array([5, 6, 7, 8, 9])]
outputs = nd.concat(*b, dim=0)

