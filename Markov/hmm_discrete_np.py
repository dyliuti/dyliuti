import numpy as np
import matplotlib.pyplot as plt

def random_normalized(d1, d2):
	x = np.random.random((d1, d2))
	return x / x.sum(axis=1, keepdims=True)

# 加载数据
sequences = []
for line in open('Data/Markov/coin_data.txt'):
	# 1 for H, 0 for T
	sequence_ = [1 if e == 'H' else 0 for e in line.rstrip()]
	sequences.append(sequence_)

H = 2
# 观测量种类  硬币抛掷就两种观测结果   +1是因为序号从0开始
V = max(max(sequence) for sequence in sequences) + 1
sequences_len = len(sequences)

# 初始化EM算法参数
pi = np.ones(H) / H
trans_mat = random_normalized(H, H)
emit_mat = random_normalized(H, V)

epochs = 30
costs = []
# 使用baum-Welch算法（前向后向）来实现EM算法
for epoch in range(epochs):
	alphas = []
	betas = []
	p_sequence = np.zeros(sequences_len)

	for n in range(sequences_len):
		sequence = sequences[n]
		T = len(sequence)
		# 观测序列上，每个状态对应着各种观测结果概率  P(Z0) * P(Y0 | Z0)
		alpha = np.zeros(shape=(T, H))
		alpha[0] = pi * emit_mat[:, sequence[0]]

		for t in range(1, T):
			# 观察者概率：P(Y0..Yt-1, Zt-1) * P(Zt | Zt-1) * P(Yt | Zt) -> P(Y0..Yt, Zt)
			alpha[t] = alpha[t-1].dot(trans_mat) * emit_mat[:, sequence[t]]	# 在观测序列已知的前提下，各状态转换为该序列观测的概率
			tmp = np.zeros(H)
			for i in range(H):
				for j in range(H):
					tmp[j] += alpha[t-1, i] * trans_mat[i, j] * emit_mat[j][sequence[t]]
			print("alpha diff: ", np.abs(alpha[t]-tmp).sum())
		p_sequence[n] = alpha[-1].sum()
		alphas.append(alpha)

		beta = np.zeros(shape=(T, H))
		beta[-1] = 1
		for t in range(T-2, -1, -1):	# 不包括-1 [ )
			# HxH H -> H
			beta[t] = trans_mat.dot(emit_mat[:, sequence[t+1]] * beta[t+1])
			tmp = np.zeros(H)
			for i in range(H):
				for j in range(H):
					tmp[i] += trans_mat[i][j] * emit_mat[j, sequence[t+1]] * beta[t+1][j]
			print("beta diff:", np.abs(beta[t] - tmp).sum())
		betas.append(beta)

	cost = np.sum(np.log(p_sequence))
	costs.append(cost)
	# 利用期望 重新估计pi, trans_mat, emit_mat  
	pi = np.sum((alphas[n][0] * betas[n][0]) / p_sequence[n] for n in range(sequences_len)) / sequences_len

	den_trans = np.zeros(shape=(H, 1))
	den_emit = np.zeros(shape=(H, 1))
	trans_num = 0
	emit_num = 0
	for n in range(sequences_len):
		sequence = sequences[n]
		T = len(sequence)
		# alpha: TxH beta:TxV 这里H=V=2 H表示2个状态  V表示两个观测状态
		# NxTxH alphas[n][: -1]: T-1xH -> T-1xV -> T-1xH ->sum H 因为有keepdims所以 Hx1
		den_trans += (alphas[n][: -1] * betas[n][: -1]).sum(axis=0, keepdims=True).T / p_sequence[n]	# 分母 TxN个和
		# NxTxH alphas[n] TxH -> TxV -> TxH -> Hx1
		den_emit += (alphas[n] * betas[n]).sum(axis=0, keepdims=True).T / p_sequence[n]
		# print("den_emit shape: ", den_emit, np.shape(den_emit))
		# 计算trans_mat的分子
		trans_num_n = np.zeros(shape=(H, H))
		for i in range(H):
			for j in range(H):
				for t in range(T - 1):	# 6-27
					trans_num_n[i, j] += alphas[n][t, i] * trans_mat[i, j] * emit_mat[j, sequence[t+1]] * betas[n][t+1, j]
		trans_num += trans_num_n / p_sequence[n]

		emit_num_n = np.zeros(shape=(H, V))
		for j in range(H):
			for t in range(T):			# 6-28
				emit_num_n[j, sequence[t]] += alphas[n][t, j] * betas[n][t, j]
		emit_num += emit_num_n / p_sequence[n]

	trans_mat = trans_num / den_trans
	emit_mat = emit_num / den_emit
print("trans_mat:", trans_mat)
print("emit_mat:", emit_mat)
print("pi:", pi)


def likelihood(sequence):
	# 使用前向算法求得 P(Y1..T|model)
	T = len(sequence)
	alpha_ = np.zeros(shape=(T, H))
	alpha_[0] = pi * emit_mat[:, sequence[0]]
	for t in range(1, T):
		alpha_[t] = alpha_[t - 1].dot(trans_mat) * emit_mat[:, sequence[t]]
	return alpha_[-1].sum()

def likelihood_multi(sequences):
	return np.array([likelihood(sequence) for sequence in sequences])

def log_likelihood_multi(sequences):
	return np.log(likelihood_multi(sequences))


def get_state_sequence(sequence):
	# 使用Viterbi算法，给定观测序列，返回最有可能的状态序列
	T = len(sequence)
	delta = np.zeros((T, H))
	psi = np.zeros((T, H))
	# H * Hx1 -> H
	delta[0] = pi * emit_mat[:, sequence[0]]
	for t in range(1, T):
		for j in range(H):
			# 前t-1到t时刻状态j最大概率时，对应的观测序列概率 单个j：H->1->O1 多个j：H->H->O1  两次max 下面的是H->1的max 回溯H->O1的max
			delta[t, j] = np.max(delta[t - 1] * trans_mat[:, j]) * emit_mat[j, sequence[t]]
			# 前t-1到t时刻状态j最大概率时的t-1时刻的状态索引  t j状态索引 -> t-1 i状态索引
			psi[t, j] = np.argmax(delta[t - 1] * trans_mat[:, j])

	# 回溯
	states = np.zeros(T, dtype=np.int32)
	# 得到最有可能的最后一个状态（对应观测序列概率最大）
	states[T - 1] = np.argmax(delta[T - 1])
	for t in range(T - 2, -1, -1):
		states[t] = psi[t + 1, states[t + 1]]
	return states


L = log_likelihood_multi(sequences).sum()
print("对于拟合参数求得的极大似然值:", L)

# trans_mat: [[0.69076382 0.30923618]
#  [0.23254533 0.76745467]]
# emit_mat: [[0.59735933 0.40264067]
#  [0.51630239 0.48369761]]
# pi: [0.48190022 0.51809978]
# 对比拟合得到的参数与真实参数，相差其实挺大的。这也说明了EM算法解的不唯一性，解与初值选取有较大关系。

pi = np.array([0.5, 0.5])
L = log_likelihood_multi(sequences).sum()
print("对于拟合参数，初始状态改变为真实参数求得的极大似然值:", L)

trans_mat = np.array([[0.1, 0.9], [0.8, 0.2]])
emit_mat = np.array([[0.6, 0.4], [0.3, 0.7]])
L = log_likelihood_multi(sequences).sum()
print("对于真实参数求得的极大似然值:", L)

# viterbi
print("对于 ", sequences[0], " 概率最大的状态序列：")
print(get_state_sequence(sequences[0]))

plt.plot()
plt.show()
