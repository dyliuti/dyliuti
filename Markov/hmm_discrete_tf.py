import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def random_normalized(d1, d2):
	x = tf.Variable(tf.random_normal(shape=(d1, d2)))
	return tf.nn.softmax(x)

def get_cost(sequence):
	# returns log P(sequence | model)
	# 使用前向后向算法求得 P(Zt|model)
	return session.run(cost, feed_dict={tfsequence: sequence})

def get_cost_multi(sequences):
	return np.array([get_cost(sequence) for sequence in sequences])


# 加载数据
sequences = []
for line in open('Data/Markov/coin_data.txt'):
	# 1 for H, 0 for T
	sequence_ = [1 if e == 'H' else 0 for e in line.rstrip()]
	sequences.append(sequence_)

H = 2
V = max(max(sequence) for sequence in sequences) + 1
sequences_len = len(sequences)

# 明显tf维度要求更严格 54行：pi * trans_mat[:, tfsequence[0]]
pi = tf.nn.softmax(tf.Variable(tf.ones(H)))
trans_mat  = random_normalized(H, H)
emit_mat = random_normalized(H, V)

####### 建立模型 ########
tfsequence = tf.placeholder(tf.int32, shape=(None, ), name='sequence')
# 序列t 多个状态i -> 序列t+1 状态j的概率
# outputs, elem   outputs有两个元素，
def recurrence(sequence_t_state_i, sequence_t):
	sequence_t_state_i = tf.reshape(sequence_t_state_i[0], shape=(1, H))
	# 1xH HxH -> 1xH * M, -> 1xH     # 矩阵相乘，已经包含了 sequence_t_next_state_ij -> sequence_t_next_state_j过程
	sequence_t_next_state_j = tf.matmul(sequence_t_state_i, trans_mat) * emit_mat[:, sequence_t]
	sequence_t_next_state_j = tf.reshape(sequence_t_next_state_j, (H, ))
	sequence_t_next = tf.reduce_sum(sequence_t_next_state_j)
	# sequence_t_next总共H个状态，每个状态占的比例就是 HMM输出了序列T1,T2...Tt+1,并且位于状态j的概率
	return sequence_t_next_state_j / sequence_t_next, sequence_t_next


# alpha, scale 对应recurrence中返回的值（最后一次迭代），所以为标量
alpha, p_sequence_scale = tf.scan(
	fn=recurrence,
	elems=tfsequence[1:],
	# H, H,-> H,  这里tf.Variable跟numpy一样
	# np.float32(1.0)作为初始句子概率，初始值大小不影响结果，只是给输出占个位置
	initializer=(pi * emit_mat[:, tfsequence[0]], np.float32(1)),
)

# tensorflow有意思的点：要使概率最大，可以定义损失函数。其最小时，句子概率最大，用梯度下降来拟合
cost = -tf.log(p_sequence_scale)  # tf.reduce_sum()
train_optimize = tf.train.AdamOptimizer(0.01)
train_step = train_optimize.minimize(cost)

####### 拟合模型以及预测 ########
init = tf.global_variables_initializer()
with tf.Session() as session:
	session.run(init)

	epochs = 2
	costs = []
	# 拟合参数
	for epoch in range(epochs):
		for t in range(sequences_len):
			c = get_cost_multi(sequences).sum()
			print("epoch: ", epoch, "t: ", t, "cost: ", cost)
			costs.append(c)
			session.run(train_step, feed_dict={tfsequence: sequences[t]})
	plt.plot(costs)
	plt.show()

	# tensorfolw: 对于拟合参数求得的极大似然值: 1032.831
	# Baum-Welch: 对于拟合参数求得的极大似然值: -1031.8137967377554
	# 两种方法其实是很接近的
	L = get_cost_multi(sequences).sum()
	print("对于拟合参数求得的极大似然值:", L)



