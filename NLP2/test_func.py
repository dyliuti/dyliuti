import numpy as np
a=np.array([1,2,3])
b=np.array([11,22,33])
res = np.concatenate((a, b),axis=0)
print(res)

a=np.array([[1,2,3],[4,5,6]])
b=np.array([[11,21,31],[7,8,9]])
res = np.concatenate((a, b),axis=-1)
print(res)

import numpy as np
# 对哪个轴进行concatenate，哪个轴的数字就变大
x = [[[1, 2, 3], [2, 2, 2]]]
tmp = np.max(x, axis=1, keepdims=True)
e = np.exp(x - np.max(x, axis=1, keepdims=True))
s = np.sum(e, axis=1, keepdims=True)
res = e / s
print(tmp)

import numpy as np
# 1x2x3  对T=2上的进行softmax
x = [[1, 2, 3], [2, 2, 2]]
tmp = np.max(x, axis=0, keepdims=True)
e = np.exp(x - np.max(x, axis=0, keepdims=True))
s = np.sum(e, axis=0, keepdims=True)
res = e / s
print(tmp)


# Dot 测试，Dot函数核心调用的是batch_dot
import tensorflow as tf
from keras.layers import Dot
from keras import backend as K
import numpy as np
a = np.arange(24).reshape(2,3,4) # a和b的维度有些讲究，具体查看Dot类的build方法
b = np.arange(48).reshape(2,3,8)
# 抛去x，a(x,y,m)的一列分别和b(x,y,n)的每一列求积再加和，得到一行，长度为n
# a有m列，结果就有m行
output = K.batch_dot(K.constant(a), K.constant(b),  axes=1) # (2, 3, 4) (2, 3, 8) -> 2, 4, 8
with tf.Session() as sess:
    output_array = sess.run(output)
    print( output_array  )
    print( output_array.shape )
