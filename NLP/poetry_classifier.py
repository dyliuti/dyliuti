import tensorflow as tf
from Data.DataExtract import load_poetry_classifier_data
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

X, Y, V = load_poetry_classifier_data(samples_per_class=500)


class_num = len(set(Y))
test_num = 50
X_test, Y_test = X[-test_num: ], Y[-test_num: ]
X, Y = X[: -test_num], Y[: -test_num]
N = len(X)	# 除去验证集的

# RNN
hidden_unit_size = 30
# 无word embedding Wx从D, . -> V, .
Wx = tf.Variable(tf.random_normal(shape=(V, hidden_unit_size)))
Wh = tf.Variable(tf.random_normal(shape=(hidden_unit_size, hidden_unit_size)))
bh = tf.Variable(tf.zeros(hidden_unit_size))
h0 = tf.Variable(tf.zeros(hidden_unit_size))
# 输出不再是V(可能的词) 而是class_num(两首诗的类别)
W0 = tf.Variable(tf.random_normal(shape=(hidden_unit_size, class_num)))
b0 = tf.Variable(tf.zeros(class_num))

# 输入、输出都是一个单词的向量
tfX = tf.placeholder(tf.int32, shape=(None, ), name='input')
tfY = tf.placeholder(tf.int32, shape=(None, ), name='output')

def recurrence(h_t, x_t):
	# tensorflow矩阵相乘维度要求比numpy严格，h_t一维，XW_Wx_t二维，要先转换
	h_t = tf.reshape(h_t, shape=(1, hidden_unit_size))
	h_t_next = tf.nn.relu(Wx[x_t] + tf.matmul(h_t, Wh) + bh)
	h_t_next = tf.reshape(h_t_next, shape=(hidden_unit_size, ))
	# y_t = tf.nn.softmax(tf.matmul(h_t, W0)) + b0
	return h_t_next

h_ = tf.scan(
	fn=recurrence,
	elems=tfX,
	initializer=h0
)

logits = tf.matmul(h_, W0) + b0	# n V 每行的每项是下个词的概率
# 只要最后一个词后的预测
prediction = tf.argmax(logits, axis=1)[-1]
output_probs = tf.nn.softmax(logits)


h = tf.reshape(h_, (-1, hidden_unit_size))
# labels = tf.reshape(tfY, shape=(-1, 1))	# 一维到二维

# sparse时，labels必须必logits小一维
cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tfY[0], logits=logits[-1])

# cost = -tf.log(output_probs[0][0])
train_optimize = tf.train.AdamOptimizer(learning_rate=0.001)
# train_optimize = tf.train.RMSPropOptimizer(learning_rate=0.001)
train_step = train_optimize.minimize(cost)

# 训练
session = tf.InteractiveSession()
init = tf.global_variables_initializer()
session.run(init)

losses = []
epochs = 1000
for epoch in range(epochs):
	X, Y = shuffle(X, Y)
	n_correct = 0
	loss =0
	for i in range(N):
		_, c, predict = session.run(fetches=[train_step, cost, prediction],
									   feed_dict={tfX: X[i], tfY: [Y[i]]})	# Y[i]只是个init [Y[i]]与tfY维度匹配
		if predict == Y[i]:
			n_correct += 1
		loss += c
	losses.append(loss)
	print("epoch: ", epoch, ",cost: ", loss, ",correct rate: ", float(n_correct)/N)

	# 区别这里做文本分类， xor: 比较句子中每个预测的词
	n_correct_test = 0
	for j in range(test_num):
		predict = session.run(fetches=prediction, feed_dict={tfX: X_test[j]})
		if predict == Y_test[j]:
			n_correct_test += 1
	print("test correct rate: ", float(n_correct_test)/test_num)

plt.plot(losses)
plt.show()