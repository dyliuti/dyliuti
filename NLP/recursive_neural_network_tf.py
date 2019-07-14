from Data.DataExtract import load_parse_tree
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# 训练速度的几点思考：迭代算法、学习率等固然重要。权重的初始化也是很重要的(有规律的)，即使理论上足够多的迭代次数能逼近最优解。

# 保证输出的logits是二维的
def get_logits_recursive(tree, logits):
	# 终止条件
	if tree.word_index is not None:
		# embedding_lookup返回的shape是ids的shape和We的列的维度的组合,如 (ids.shpae, We.shape[1])
		# ids的shape可以是(),(n,),(n,m)...分别对应0维常数，1维列表，2维平面...
		# x的shape需要是(1, D) 若无[]，返回就是(D,) tensorflow维度要求很严格。。。
		x = tf.nn.embedding_lookup(We, ids=[tree.word_index])
		logit = tf.matmul(x, W0) + b0
		logit = tf.reshape(logit, shape=(class_num, ))
		logits.append(logit)
		return x

	# 后序遍历
	x_left = get_logits_recursive(tree.left, logits)
	x_right = get_logits_recursive(tree.right, logits)
	x = tf.nn.relu(tf.matmul(x_left, W_left) + tf.matmul(x_right, W_right) + bh)
	logit = tf.matmul(x, W0) + b0
	logit = tf.reshape(logit, shape=(class_num, ))
	logits.append(logit)
	return x


def get_logits(tree):
	logits = []
	get_logits_recursive(tree, logits)
	# logits = tf.concact(logits)  # 若没logit = tf.reshape(logit, shape=(class_num, ))的话，这样也行
	return logits

# 返回的顺序同logits一样
def get_label_recusive(tree):
	if tree is None:
		return []
	return get_label_recusive(tree.left) + get_label_recusive(tree.right) + [tree.label]


def get_labels(tree):
	labels = get_label_recusive(tree)
	return labels


X_tree, Y_tree, word2index = load_parse_tree()
X_tree = X_tree[:100]
Y_tree = Y_tree[:100]

V = len(word2index)
D = 80
# label从0-4
class_num = 5

# 构建递归神经网络
# word embedding
We = tf.Variable(tf.random_normal(shape=(V, D))/ tf.square(tf.to_float(V + D))) # / tf.square(tf.to_float(V + D))
# 左节点权重, 从隐藏层单一的循环节点转换为左右节点
W_left = tf.Variable(tf.random_normal(shape=(D, D))/ tf.square(tf.to_float(D + D))) #  / tf.square(tf.to_float(D + D))
# 右节点权重
W_right = tf.Variable(tf.random_normal(shape=(D, D))/ tf.square(tf.to_float(D + D))) #  / tf.square(tf.to_float(D + D))
# bias 相对于输出节点(父节点来说的)，单个偏置就够了
bh = tf.zeros(D)
# 输出层前的参数
W0 = tf.Variable(tf.random_normal(shape=(D, class_num))/ tf.square(tf.to_float(class_num + D))) #  / tf.square(tf.to_float(class_num + D))
b0 = tf.zeros(class_num)
params = [We, W_left, W_right, W0]

# 通过每棵树的句子结构进行训练
reg = 0.1
lr = 0.1
all_labels = []
all_predictions = []
train_steps = []
costs = []
i = 0
N = len(X_tree)
for tree in X_tree:
	i += 1
	print("processed: %d/%d" % (i, N))
	logits = get_logits(tree)
	labels = get_labels(tree)
	predictions = tf.argmax(logits, axis=1)
	all_predictions.append(predictions)
	all_labels.append(labels)

	# 定义损失, 交叉熵+L2正则
	cost = tf.reduce_mean(
		tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=labels,
			logits=logits
		)
	)
	# 迭代生成器
	l2_cost = sum(tf.nn.l2_loss(param) for param in params)
	cost += reg * l2_cost
	costs.append(cost)

	# 梯度算法优化
	# train_optimize = tf.train.AdamOptimizer(learning_rate=lr)
	train_optimize = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
	train_step = train_optimize.minimize(cost)
	train_steps.append(train_step)

# 开始训练
epochs = 5
losses = []
tree_losses = []
correct_rates = []
init = tf.global_variables_initializer()
with tf.Session() as session:
#session = tf.Session()
	session.run(init)

	for epoch in range(epochs):
		epoch_cost = 0
		n_correct = 0
		n_total = 0
		i = 0
		train_steps, costs, all_predictions, all_labels = shuffle(train_steps, costs, all_predictions, all_labels)

		for train_step, cost, predictions, labels in zip(train_steps, costs, all_predictions, all_labels):
			# 无feed_dict, 输入不是用占位符(输入sequence)而是用tree生成sequence
			_, c, p = session.run(fetches=[train_step, cost, predictions])
			epoch_cost += c
			tree_losses.append(c)
			n_correct += np.sum(p == labels)
			n_total += len(labels)

			i += 1
			if i % 10 == 0:
				print("epoch: ", epoch, ",n: ", i, ",cost: %.4f" % c, ",correct rate: %.4f" % (float(n_correct) / n_total))

		losses.append(epoch_cost)
		correct_rates.append(float(n_correct) / n_total)
	tf.train.Saver().save(session, 'recursive_neural_network_tf.ckpt')

plt.plot(tree_losses)
plt.title("train_step loss")
plt.show()

plt.plot(losses)
plt.title("epoch loss")
plt.show()

plt.plot(correct_rates)
plt.title("epoch correct rates")
plt.show()


# 评价新的树
def test_score(trees):
	all_predictions = []
	all_labels = []
	N = len(trees)
	for tree in trees:
		logits = get_logits(tree)
		labels = get_labels(tree)
		predictions = tf.argmax(logits, axis=1)
		all_predictions.append(predictions)
		all_labels.append(labels)

	n_correct = 0
	n_total = 0
	with tf.Session() as session:
		tf.train.Saver().restore(session, 'recursive_neural_network_tf.ckpt')
		for predictions, labels in zip(all_predictions, all_labels):
			p = session.run(predictions)
			n_correct += (p[-1] == labels[-1])
			n_total += 1
	return float(n_correct) / n_total

print("test tree correct rate: ", test_score(Y_tree))