import tensorflow as tf
from Data.DataExtract import load_robert_frost
from sklearn.utils import shuffle


sentences, word2index = load_robert_frost()
V = len(word2index)		# 单词量
D = 50		# word embedding的维度
hidden_unit_size = 40

# We表示word embedding
We = tf.Variable(tf.random_normal(shape=(V, D)))
Wx = tf.Variable(tf.random_normal(shape=(D, hidden_unit_size)))
Wh = tf.Variable(tf.random_normal(shape=(hidden_unit_size, hidden_unit_size)))
bh = tf.Variable(tf.zeros(hidden_unit_size))
h0 = tf.Variable(tf.zeros(hidden_unit_size))
W0 = tf.Variable(tf.random_normal(shape=(hidden_unit_size, V)))
b0 = tf.Variable(tf.zeros(V))

# 输入、输出都是一个单词的向量
tfX = tf.placeholder(tf.int32, shape=(None, ), name='input')
tfY = tf.placeholder(tf.int32, shape=(None, ), name='output')

# 简单的循环神经网络有意思的是输入不依赖与中间隐藏层，可利用矩阵提前一次性计算
# 这里与numpy相同We[tfX]， tfX是一维列表时结果为多行(多个词向量),几行即几个序列
# 这种形式是one_hot.dot(We)的效率表示
XW = tf.nn.embedding_lookup(We, tfX)
# 计算到隐藏单元的其一输入, 包含了多个序列   为什么没偏差？？？
XW_Wx = tf.matmul(XW, Wx)

def recurrence(h_t, XW_Wx_t):
	# tensorflow矩阵相乘维度要求比numpy严格，h_t一维，XW_Wx_t二维，要先转换
	h_t = tf.reshape(h_t, shape=(1, hidden_unit_size))
	h_t_next = tf.nn.relu(XW_Wx_t + tf.matmul(h_t, Wh) + bh)
	h_t_next = tf.reshape(h_t_next, shape=(hidden_unit_size, ))
	return h_t_next

h = tf.scan(
	fn=recurrence,
	elems=XW_Wx,
	initializer=h0
)

# 输出结果也不依赖中间层，一次性矩阵处理
logits = tf.matmul(h, W0) + b0	# n V 每行的每项是下个词的概率
predictions = tf.argmax(logits, axis=1)
output_probs = tf.nn.softmax(logits)

h = tf.reshape(h, (-1, hidden_unit_size))
labels = tf.reshape(tfY, shape=(-1, 1))

cost = tf.reduce_mean(
	tf.nn.sampled_softmax_loss(
		weights=tf.transpose(W0),
		biases=b0,
		labels=labels,		# Tensor` of type `int64` and shape `[batch_size, num_true]`.
		inputs=h,			# Tensor` of shape `[batch_size, dim]`
		num_sampled=50,		# The number of classes to randomly sample per batch.
		num_classes=V		# The number of possible classes.
	)
)
train_optimize = tf.train.AdamOptimizer(learning_rate=0.01)
train_step = train_optimize.minimize(cost)

# 训练
session = tf.InteractiveSession()
init = tf.global_variables_initializer()
session.run(init)

losses = []
epochs = 50
# 每个input_sequence句子长度之和
n_total = sum([len(sentence) + 1 for sentence in sentences])
for epoch in range(epochs):
	sentences = shuffle(sentences)
	n_correct = 0
	loss =0
	for i in range(len(sentences)):
		# 0 对应 'START', 1 对应 'END'
		input_sequence = [0] + sentences[i]
		output_sequence = sentences[i] + [1]

		_, c, predict_indexs = session.run(fetches=[train_step, cost, predictions],
									   feed_dict={tfX: input_sequence, tfY: output_sequence})
		loss += c
		# 比较句子中每个预测的词
		for predict_index, word_index in zip(predict_indexs, output_sequence):
			if predict_index == word_index:
				n_correct += 1
	losses.append(loss)
	print("epoch: ", epoch, ",cost: ", loss, ",correct rate: ", float(n_correct)/n_total)


import numpy as np
pi = np.zeros(V)
for sentence in sentences:
	pi[sentence[0]] += 1
pi /= pi.sum()

index2word = {v: k for k, v in word2index.items()}
# generate 4 lines at a time
n_lines = 0
# 使用'START' will always yield the same first word!
start_word_index = [np.random.choice(V, p=pi)]
print(index2word[start_word_index[0]], end=" ")

while n_lines < 4:
	probs_ = session.run(output_probs, feed_dict={tfX: start_word_index})
	probs = probs_[-1]
	word_index = np.random.choice(V, p=probs)
	start_word_index.append(word_index)
	if word_index > 1:
		# it's a real word, not start/end token
		word = index2word[word_index]
		print(word, end=" ")
	elif word_index == 1:
		# end token
		n_lines += 1
		print('')
		if n_lines < 4:
			X = [np.random.choice(V, p=pi)]  # reset to start of line
			print(index2word[X[0]], end=" ")

