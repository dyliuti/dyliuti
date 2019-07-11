from Data.DataExtract import load_chunking
from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell
import tensorflow as tf
from sklearn.utils import shuffle
from datetime import datetime
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def flatten(l):
	return [item for sublist in l for item in sublist]


# get the data
X_train, Y_train, X_test, Y_test, word2idx = load_chunking(split_sequence=True)
V = len(word2idx) + 2  # vocab size (+1 for unknown, +1 b/c start from 1)
K = len(set(flatten(Y_train)) | set(flatten(Y_test))) + 1  # num classes   验证集与训练集共有的tags总数: 45

# training config
epochs = 20
learning_rate = 1e-2
mu = 0.99
batch_size = 32
hidden_layer_size = 30  # hideene_layer_size
embedding_dim = 20
test = [len(x) for x in (X_train + X_test)]  # 8936 + 2012 个 序列长度数
sequence_length = max(len(x) for x in X_train + X_test)  # 最长的序列长度

# pad sequences 小于maxlen的前面填0
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=sequence_length)  # 8936， 78
Y_train = tf.keras.preprocessing.sequence.pad_sequences(Y_train, maxlen=sequence_length)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=sequence_length)  # 2012
Y_test = tf.keras.preprocessing.sequence.pad_sequences(Y_test, maxlen=sequence_length)
print("X_train.shape:", X_train.shape)
print("Y_train.shape:", Y_train.shape)

# inputs
inputs = tf.placeholder(tf.int32, shape=(None, sequence_length))  # (?，78) 固定了最长序列78， 不定长->定长
targets = tf.placeholder(tf.int32, shape=(None, sequence_length))
num_samples = tf.shape(inputs)[0]  # useful for later              # 样本数：8936

# make them tensorflow variables
tfWe = tf.Variable(tf.random_normal(shape=(V, embedding_dim)))		# V 10 	# embedding
tfWo = tf.Variable(tf.random_normal(shape=(hidden_layer_size, K)))	# 10, 45
tfbo = tf.Variable(tf.zeros(K))										# 45	# target：tags

# make the rnn unit
rnn_unit = GRUCell(num_units=hidden_layer_size, activation=tf.nn.relu)

# 一个句子（序列）固定长度78，  M肯定是词向量权重的特征维数 N为输入的样本量: ?
# tfWe: 19124, 20  inputs: ? 78  -> x: ? 78 20 结果会比inputs多出一维(inputs是一个数，输出是一维)
x = tf.nn.embedding_lookup(tfWe, inputs)  # 从所有word embedding的特征空间中  取出输入inputs中那么多个

# T ? 20   T个unstack
x = tf.unstack(x, num=sequence_length, axis=1)  # T NxM   T ?x10

# rnn output:  T ? 30
outputs, states = get_rnn_output(rnn_unit, x, dtype=tf.float32)  # x: ?x10  ?x10, 10x10 -> ? 10    T ? 10

# outputs are now of size (T, N, M)
# so make it (N, T, M)
outputs = tf.transpose(outputs, (1, 0, 2))
outputs = tf.reshape(outputs, (sequence_length * num_samples, hidden_layer_size))  # NT x M         ?Tx10

# final dense layer
logits = tf.matmul(outputs, tfWo) + tfbo  # NT x K       ?Tx10 10xK -> ?TxK  ?T其实就是有这么多个单词
predictions = tf.argmax(logits, axis=1)  # 一维的了 ?T 个索引
predict_op = tf.reshape(predictions, (num_samples, sequence_length))  # ?xT
labels_flat = tf.reshape(targets, [-1])  # target ?xT-> ?T 展开为一维   对每个单词都进行熵计算

cost_optimize = tf.reduce_mean(
	tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=logits,  # ?T个索引
		labels=labels_flat  # ?T个标签
	)
)
train_optimize = tf.train.AdamOptimizer(learning_rate).minimize(cost_optimize)
train_step = train_optimize.minimize(cost_optimize)

# init stuff
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

# training loop
costs = []
n_batches = len(Y_train) // batch_size
for i in range(epochs):
	n_total = 0
	n_correct = 0

	t0 = datetime.now()
	X_train, Y_train = shuffle(X_train, Y_train)
	cost = 0

	for j in range(n_batches):
		x = X_train[j * batch_size:(j + 1) * batch_size]
		y = Y_train[j * batch_size:(j + 1) * batch_size]

		# get the cost, predictions, and perform a gradient descent step
		c, p, _ = sess.run(
			(cost_optimize, predict_op, train_step),
			feed_dict={inputs: x, targets: y})
		cost += c

		# calculate the accuracy
		for yi, pi in zip(y, p):
			# we don't care about the padded entries so ignore them
			yii = yi[yi > 0]
			pii = pi[yi > 0]
			n_correct += np.sum(yii == pii)
			n_total += len(yii)

		# print stuff out periodically
		if j % 10 == 0:
			print(
				"j/N: %d/%d correct rate so far: %f, cost so far: %f\r" %
				(j, n_batches, float(n_correct) / n_total, cost)
			)

	# 训练集训练完了
	# get test acc. too
	p = sess.run(predict_op, feed_dict={inputs: X_test, targets: Y_test})
	n_test_correct = 0
	n_test_total = 0
	for yi, pi in zip(Y_test, p):
		yii = yi[yi > 0]
		pii = pi[yi > 0]
		n_test_correct += np.sum(yii == pii)
		n_test_total += len(yii)
	test_acc = float(n_test_correct) / n_test_total

	print(
		"i:", i, "cost:", "%.4f" % cost,
		"train acc:", "%.4f" % (float(n_correct) / n_total),
		"test acc:", "%.4f" % test_acc,
		"time for epoch:", (datetime.now() - t0)
	)
	costs.append(cost)

plt.plot(costs)
plt.show()


