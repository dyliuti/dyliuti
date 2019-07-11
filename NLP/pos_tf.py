from Data.DataExtract import load_chunking
from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell
import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def flatten(l):
	return [item for sublist in l for item in sublist]


# get the data
X_train, Y_train, X_test, Y_test, word2idx = load_chunking(split_sequence=True)
# tags类别数
class_set = set(flatten(Y_train)) | set(flatten(Y_test))
class_num = len(class_set) + 1
V = len(word2idx) + 2  # vocab size (+1 for unknown, +1 b/c start from 1)


# training config
epochs = 20
learning_rate = 1e-2
mu = 0.99
batch_size = 32
hidden_layer_size = 30  # hideene_layer_size
embedding_dim = 20
# 在训练集与测试集中最长的序列长度
max_sequence_length = max(len(x) for x in X_train + X_test)

# 小于max_sequence_length的序列前面补0
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_sequence_length)
Y_train = tf.keras.preprocessing.sequence.pad_sequences(Y_train, maxlen=max_sequence_length)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_sequence_length)
Y_test = tf.keras.preprocessing.sequence.pad_sequences(Y_test, maxlen=max_sequence_length)

inputs = tf.placeholder(tf.int32, shape=(None, max_sequence_length))
targets = tf.placeholder(tf.int32, shape=(None, max_sequence_length))
samples_num = tf.shape(inputs)[0] # 样本数

# VxD
tfWe = tf.Variable(tf.random_normal(shape=(V, embedding_dim)))
tfW0 = tf.Variable(tf.random_normal(shape=(hidden_layer_size, class_num)))
tfb0 = tf.Variable(tf.zeros(class_num))

rnn_unit = GRUCell(num_units=hidden_layer_size, activation=tf.nn.relu)
# tfWe: VxD  inputs: NxT  -> x: NxTxD
x = tf.nn.embedding_lookup(tfWe, inputs)
# TxNxD
x = tf.unstack(x, num=max_sequence_length, axis=1)
# x：TxNxD -> outputs: TxNxM
outputs, state = get_rnn_output(rnn_unit, x, dtype=tf.float32)
# outputs: NxTxM
outputs = tf.transpose(outputs, perm=(1, 0, 2))
# N*TxM
outputs = tf.reshape(outputs, shape=(samples_num * max_sequence_length, hidden_layer_size))

# 输出
logits = tf.matmul(outputs, tfW0) + tfb0
# N*Tx1
predictions = tf.argmax(logits, axis=1)
# NxT
predictions = tf.reshape(predictions, shape=(samples_num, max_sequence_length))
# NxT -> N*T
labels_flat = tf.reshape(targets, [-1])

cost = tf.reduce_mean(
	tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels_flat,
		logits=logits
	)
)

train_optimize = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_step = train_optimize.minimize(cost)


# 开始训练
session = tf.InteractiveSession()
init = tf.global_variables_initializer()
session.run(init)

losses = []
n_batches = len(Y_train) // batch_size
for epoch in range(epochs):
	n_total, n_correct, loss = 0, 0, 0
	X_train, Y_train = shuffle(X_train, Y_train)

	for i in range(n_batches):
		X = X_train[i * batch_size: (i + 1) * batch_size]
		Y = Y_train[i * batch_size: (i + 1) * batch_size]

		c, p, _ = session.run(fetches=[cost, predictions, train_step],
							  feed_dict={inputs: X, targets: Y})
		cost += c
		# 计算准确率
		for yi, pi in zip(Y, p):
			# 忽略补充0的部分
			yii = yi[yi > 0]
			pii = pi[yi > 0]
			n_correct += np.sum(yii == pii)
			n_total += len(yii)
		if i % 10 == 0:
			print("epoch: ", epoch, ",n: ", i, ",cost: ", cost, "correct rate: ", float(n_correct) / n_total)
	losses.append(cost)
	# 测试集
	p = session.run(predictions, fetches=X_test)
	n_test_correct, n_test_total = 0, 0
	for yi, pi in zip(Y_test, p):
		# 忽略补充0的部分
		yii = yi[yi > 0]
		pii = pi[yi > 0]
		n_test_correct += np.sum(yii == pii)
		n_test_total += len(yii)
	test_acc = float(n_test_correct) / n_test_total
	print("epoch: ", epoch, ",cost: ", cost, ",train acc: %.4f, test acc: %.4f" %(float(n_correct) / n_total, test_acc))

plt.plot(losses)
plt.show()



