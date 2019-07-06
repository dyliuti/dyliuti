import tensorflow as tf

a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
c = a + b

# 注意：allow_soft_placement=True表明：计算设备可自行选择，如果没有这个参数，会报错。
# 因为不是所有的操作都可以被放在GPU上，如果强行将无法放在GPU上的操作指定到GPU上，将会报错。
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())
print(sess.run(c))

max([1, 3, 2])

import numpy as np
z = np.array([[-1, 0, 3, 2], [0, 3, -2, -1]])
print(z < 0)
print(z[z < 0])
z[z < 0] = 0
print(z)
print(z.shape)


logits = np.array([1, -3, 10])
label = np.array([0.1, 0.02, 0.88])
soft = np.exp(logits) / sum(np.exp(logits))
res = - sum(label * np.log(soft))

logits = np.array([1, -3, 10])
label = np.array([0, 0, 1])
soft = np.exp(logits) / sum(np.exp(logits))
res = - sum(label * np.log(soft))

print(7//3)

# 自相关
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

X = np.random.randn(1000)
C = correlate(X, X)
plt.plot(C)
plt.show()

# Y[0] 与 X[100] 相互联系， Y[1] 与 X[101] 相互联系
# Cross-Correlation
Y = np.empty(1000)
Y[0:900] = X[100:]
Y[900:] = X[:100]
C2 = correlate(X, Y)
plt.plot(C2)
plt.show()

from scipy.signal import convolve
C3 = convolve(X, np.flip(Y, 0))
plt.plot(C3)
plt.show()

d = {'aa': 0, 'bb': 1}
test = ((v, k) for k, v in d.items())  # generator object
print(dict(test))

import numpy as np
arr = np.arange(9).reshape((3, 3))
print(arr[0, 2])
print(arr[[0, 2]])
arr[[0,2],[0,2]] = 10
print(arr)
print(arr[:, [0, 2]]) # 输出两列
np.sum(np.array([1, 1]) * arr[:, [0, 2]], axis=1)

a = np.array([1, 3, 4])
a_T = a.T
print(a==a_T)

sentence_size = 12
random_word_sequence = np.random.choice(sentence_size, size=sentence_size, replace=False)

