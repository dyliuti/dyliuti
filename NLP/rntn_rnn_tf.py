from Data.DataExtract import load_parse_tree
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def tensor_mul(d, x_left, W, x_right):
	# dxdxd -> dxd*d
	W = tf.reshape(W, [d, d*d])
	# 1xdxd
	tmp = tf.matmul(x_left, W)
	# dxd
	tmp = tf.reshape(tmp, [d, d])
	# x_left.W.x_right.T
	tmp = tf.matmul(tmp, tf.transpose(x_right))
	# d 与rntn区别的
	# res = tf.reshape(tmp, [1, d])
	return tf.reshape(tmp, [d, ])


def tree2list(tree, parent_index, is_binary=False):
	if tree is None:
		return [], [], [], []

	words_left, left_child_left, right_child_left, labels_left = tree2list(tree.left, tree.index, is_binary)
	words_right, left_child_right, right_child_right, labels_right = tree2list(tree.right, tree.index, is_binary)

	if tree.word_index is None:
		w = -1
		left = tree.left.index
		right = tree.right.index
	else:
		w = tree.word_index
		left = -1
		right = -1

	words = words_left + words_right + [w]
	left_child = left_child_left + left_child_right + [left]
	right_child = right_child_left + right_child_right + [right]

	if is_binary:
		if tree.label > 2:
			label = 1
		elif tree.label < 2:
			label = 0
		else:
			label = -1 # 中性标签过滤掉
	else:
		label = tree.label
	labels = labels_left + labels_right + [label]

	return words, left_child, right_child, labels

def add_index_to_tree(tree: object, current_index: object) -> object:
	# 后序遍历添加索引
	if tree is None:
		return current_index
	current_index = add_index_to_tree(tree.left, current_index)
	current_index = add_index_to_tree(tree.right, current_index)
	tree.index = current_index
	current_index += 1
	return current_index



X_tree_m, Y_tree_m, word2index = load_parse_tree()
is_binary = True
for tree in X_tree_m:
	add_index_to_tree(tree, 0)
# 包含了words的索引，left_children, right_children, labels
X_list = [tree2list(tree, -1, is_binary) for tree in X_tree_m]
# 过滤掉最后的label是中性的样本
if is_binary:
	X_list = [x for x in X_list if x[3][-1] >= 0]

# 将测试集也进行同样转换
for tree in Y_tree_m:
	add_index_to_tree(tree, 0)
Y_list = [tree2list(tree, -1, is_binary) for tree in Y_tree_m]
if is_binary:
	Y_list = [y for y in Y_list if y[3][-1] >= 0]

Y_tree = Y_tree_m[:1000]

N = len(X_list)
V = len(word2index)
D = 80
# label从0-4
class_num = 5
reg = 0.001
lr = 0.001
epochs = 8

# 构建递归神经网络
# word embedding
We = tf.Variable(tf.random_normal(shape=(V, D))/ tf.square(tf.to_float(V + D))) # / tf.square(tf.to_float(V + D))
# 左节点权重, 从隐藏层单一的循环节点转换为左右节点
W_left = tf.Variable(tf.random_normal(shape=(D, D))/ tf.square(tf.to_float(D + D))) #  / tf.square(tf.to_float(D + D))
# 右节点权重
W_right = tf.Variable(tf.random_normal(shape=(D, D))/ tf.square(tf.to_float(D + D))) #  / tf.square(tf.to_float(D + D))
# quadratic terms
W_left2 = tf.Variable(tf.random_normal(shape=(D, D, D))/ tf.square(tf.to_float(3 * D)))
W_right2 = tf.Variable(tf.random_normal(shape=(D, D, D))/ tf.square(tf.to_float(3 * D)))
W_leftRight = tf.Variable(tf.random_normal(shape=(D, D, D))/ tf.square(tf.to_float(3 * D)))
# bias 相对于输出节点(父节点来说的)，单个偏置就够了
bh = tf.zeros(D)
# 输出层前的参数
W0 = tf.Variable(tf.random_normal(shape=(D, class_num))/ tf.square(tf.to_float(class_num + D))) #  / tf.square(tf.to_float(class_num + D))
b0 = tf.zeros(class_num)
params = [We, W_left, W_right, W_left2, W_right2, W_leftRight, W0]

words = tf.placeholder(tf.int32, shape=(None, ))
left_children = tf.placeholder(tf.int32, shape=(None, ))
right_children = tf.placeholder(tf.int32, shape=(None, ))
labels = tf.placeholder(tf.int32, shape=(None, ))


def dot1(a, B):
	res = tf.matmul(a, B)
	res = tf.reshape(res, [D, ])
	return res

def recursive_net_transform(hiddens, n):
	hidden_left = hiddens.read(index=left_children[n])
	hidden_right = hiddens.read(right_children[n])
	# hidden_left, hidden_right是一维的，不符合tf.matmul，要转换为二维
	hidden_left = tf.reshape(hidden_left, shape=[1, D])
	hidden_right = tf.reshape(hidden_right, shape=[1, D])
	return tf.nn.relu(
		tensor_mul(D, hidden_left, W_left2, hidden_left) +
		tensor_mul(D, hidden_right, W_right2, hidden_right) +
		tensor_mul(D, hidden_left, W_leftRight, hidden_right) +
		dot1(hidden_left, W_left) +
		dot1(hidden_right, W_right) +
		bh	# 这里 (1,D) + (D,)为什么就不可以了? 上面输出也都是tensor啊
	)

def recurrence(hiddens, n):
	word_index = words[n]

	hidden_n = tf.cond(
		word_index >= 0,
		# word_index 0维，输出的hidden_n是1维
		lambda : tf.nn.embedding_lookup(We, word_index),
		lambda : recursive_net_transform(hiddens, n)
	)
	# index: 0 - D.int32 scalar with the index to write to.
	# value: N-D. Tensor of type dtype. The Tensor to write to this index.
	hiddens = hiddens.write(index=n, value=hidden_n)
	n = tf.add(n, 1)
	return hiddens, n

def condition(hiddens, n):
	# 当n < len(words)时，循环继续
	return tf.less(n, tf.shape(words)[0])

hiddens = tf.TensorArray(
	dtype=tf.float32,
	size=0,
	dynamic_size=True,
	clear_after_read=False,
	infer_shape=False
)

outputs, _ = tf.while_loop(
	cond=condition,
	body=recurrence,
	loop_vars=[hiddens, tf.constant(0)],
	parallel_iterations=1
)
h = outputs.stack()
logits = tf.matmul(h, W0) + b0
predictions = tf.argmax(logits, axis=1)

# 定义损失, 交叉熵+L2正则   此次不计算内置节点，只计算根节点损失
cost = tf.reduce_mean(
	tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels[-1],
		logits=logits[-1]
	)
)
# 迭代生成器
l2_cost = sum(tf.nn.l2_loss(param) for param in params)
cost += reg * l2_cost

train_optimize = tf.train.AdamOptimizer(learning_rate=lr)
train_step = train_optimize.minimize(cost)

# 开始训练

losses = []
sequence_indexes = range(N)
init = tf.global_variables_initializer()
with tf.Session() as session:
#session = tf.Session()
	session.run(init)

	for epoch in range(epochs):
		sequence_indexes = shuffle(sequence_indexes)
		epoch_cost = 0
		n_correct = 0
		n_total = 0
		num = 0

		for i in sequence_indexes:
			words_, left, right, lab = X_list[i]
			c, p, _ = session.run(
				fetches=(cost, predictions, train_step),
				feed_dict={words: words_,
						   left_children: left,
						   right_children: right,
						   labels: lab
				}
			)
			if np.isnan(c):
				print("损失值太大了，尝试降低学习率再运行程序")
				for p in params:
					print(p.eval(session).sum())
				exit()
			epoch_cost += c
			n_correct += p[-1] == lab[-1]
			n_total += 1

			num += 1
			if num % 10 == 0:
				print("epoch: ", epoch, ",processed: %d/%d" % (num, N), ",cost: %.4f" % c, ",correct rate: %.4f" % (float(n_correct) / n_total))

		losses.append(epoch_cost)

	n_test_correct = 0
	n_test_total = 0
	for words_, left, right, lab in Y_list:
		p = session.run(predictions, feed_dict={
			words: words_,
			left_children: left,
			right_children: right,
			labels: lab
		})
		n_test_correct += (p[-1] == lab[-1])
		n_test_total += 1

	print("epoch: ", epoch, "cost", cost, ",train acc: %.4f, test acc: %.4f" % (float(n_test_correct)/n_test_total, float(n_correct)/n_total))


plt.plot(losses)
plt.show()


