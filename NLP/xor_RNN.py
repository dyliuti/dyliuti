import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell
from sklearn.utils import shuffle
from NLP.Common.Util import all_parity_pairs_with_sequence_labels, init_weight


def x2sequence(x, batch_size, T, D):
	x = tf.transpose(x, perm=(1, 0, 2))
	x = tf.reshape(x, (T * batch_size, D))
	x = tf.split(x, T)
	return x

bit_len = 12
hidden_unit_size = 4
batch_size = 20

learning_rate = 0.1
momentum = 0.9
epochs = 100
# X: 4100 bit_len 1  Y: 4100 bit_len
X, Y = all_parity_pairs_with_sequence_labels(bit_len)
N, T, D = X.shape	# T 同bit_len
class_num = len(set(Y.flatten()))

W0 = init_weight(hidden_unit_size, class_num).astype(np.float32)
b0 = np.zeros(class_num, dtype=np.float32)
tfW0 = tf.Variable(W0)
tfb0 = tf.Variable(b0)

# 受限于X的shape，tfX就长这样了，导致之后的一系列维度转换
tfX = tf.placeholder(tf.float32, shape=(batch_size, bit_len, D), name='inputs')
tfY = tf.placeholder(tf.int32, shape=(batch_size, bit_len), name='outputs')

# 将tfX转换为序列 bit_len个lists 每个list里是 batch_size D
sequenceX = x2sequence(tfX, batch_size, bit_len, D)

rnn_units = BasicRNNCell(num_units=hidden_unit_size, activation=tf.nn.sigmoid)

# outputs同sequenceX: bit_len batch_size D  bit_len个二维tensor
outputs_, states = get_rnn_output(rnn_units, sequenceX, dtype=tf.float32)
outputs = tf.transpose(outputs_, perm=(1, 0, 2))
outputs = tf.reshape(outputs, shape=(bit_len * batch_size, hidden_unit_size))

logits = tf.matmul(outputs, tfW0) + tfb0
predict = tf.argmax(logits, axis=1)
targets = tf.reshape(tfY, shape=(bit_len * batch_size, ))

# 损失函数
loss = tf.reduce_mean(
	tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=targets,
		logits=logits
	))
# 优化算法
train_optimize = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
train_setp = train_optimize.minimize(loss)

# batch gradient
losses = []
batches_num = N // batch_size

init = tf.global_variables_initializer()
with tf.Session() as session:
	session.run(init)
	for epoch in range(epochs):
		X, Y = shuffle(X, Y)
		costs = 0
		n_correct = 0
		for i in range(batches_num):
			X_batch = X[i * batch_size: (i + 1) * batch_size]
			Y_batch = Y[i * batch_size: (i + 1) * batch_size]
			_, cost, predict_batch = session.run(fetches=[train_setp, loss, predict], feed_dict={tfX: X_batch, tfY: Y_batch})
			costs += cost
			for b in range(batch_size):
				index = (b + 1) * T - 1		# batch_size * T 看12位最后一位是否预测对
				n_correct += (predict_batch[index] == Y_batch[b][-1])
			# n_correct += tf.reduce_sum(predict_batch == tf.reshape(Y_batch, (batch_size, bit_len)))
		if epoch % 10 == 0:
			print("epoch: ", epoch, ",cost: ", costs, "correct rate: ", float(n_correct)/N)
		if n_correct == N:
			print("epoch: ", epoch, ",cost: ", costs, "correct rate: ", float(n_correct) / N)
			break
		losses.append(costs)


