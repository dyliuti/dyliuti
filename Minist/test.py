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