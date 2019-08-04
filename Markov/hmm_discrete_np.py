import numpy as np
import matplotlib.pyplot as plt

def random_normalized(d1, d2):
	x = np.random.random((d1, d2))
	return x / x.sum(axis=1, keepdims=True)

# 加载数据
sequences = []
for line in open('Markov/coin_data.txt'):
	# 1 for H, 0 for T
	sequence_ = [1 if e == 'H' else 0 for e in line.rstrip()]
	sequences.append(sequence_)

M = 2
# 观测量种类  硬币抛掷就两种观测结果   +1是因为序号从0开始
V = max(max(sequence) for sequence in sequences) + 1
sequences_num = len(sequences)

# 初始化EM算法参数
pi = np.ones(M) / M
trans_mat = random_normalized(M, M)
emit_mat = random_normalized(M, V)

epochs = 30
costs = []
# 使用baum-Welch算法（前向后向）来实现EM算法
for epoch in range(epochs):
	alphas = []
	betas = []
	p_sequence = np.zeros(sequences_num)

	for n in range(sequences_num):
		sequence = sequences[n]
		T = len(sequence)
		# 观测序列上，每个状态对应着各种观测结果概率  P(Z0) * P(Y0 | Z0)
		alpha = np.zeros(shape=(T, M))
		alpha[0] = pi * emit_mat[:, sequence[0]]

		for t in range(1, T):
			# 观察者概率：P(Y0..Yt-1, Zt-1) * P(Zt | Zt-1) * P(Yt | Zt) -> P(Y0..Yt, Zt)
			alpha[t] = alpha[t-1].dot(trans_mat) * emit_mat[:, sequence[t]]
			tmp = np.zeros(M)
			for i in range(M):
				for j in range(M):
					tmp[j] += alpha[t-1, i] * trans_mat[i, j] * emit_mat[j][sequence[t]]
			print("alpha diff: ", np.abs(alpha[t]-tmp).sum())
		p_sequence[n] = alpha[-1].sum()
		alphas.append(alpha)

		beta = np.zeros(shape=(T, M))
		beta[-1] = 1
		for t in range(T-2, -1, -1):	# 不包括-1 [ )
			# MxM M -> M
			beta[t] = trans_mat.dot(emit_mat[:, sequence[t+1]] * beta[t+1])
			tmp = np.zeros(M)
			for i in range(M):
				for j in range(M):
					tmp[i] += trans_mat[i][j] * emit_mat[j, sequence[t+1]] * beta[t+1][j]
			print("beta diff:", np.abs(beta[t] - tmp).sum())
		betas.append(beta)

	cost = np.sum(np.log(p_sequence))
	costs.append(cost)
	# 利用期望 重新估计pi, trans_mat, emit_mat  
	pi = np.sum((alphas[n][0] * betas[n][0]) / p_sequence[n] for n in range(sequences_num)) / sequences_num

	den_trans = np.zeros(shape=(M, 1))
	den_emit = np.zeros(shape=(M, 1))
	trans_num = 0
	emit_num = 0
	for n in range(sequences_num):
		sequence = sequences[n]
		T = len(sequence)
		
		# NxTxM alphas[n][: -1]: T-1xM -> 1xM -> Mx1
		den_trans += (alphas[n][: -1] * betas[n][: -1]).sum(axis=0, keepdims=True).T / p_sequence[n]
		# NxTxM alphas[n] TxM -> 1xM -> Mx1
		den_emit += (alphas[n] * betas[n]).sum(axis=0, keepdims=True).T / p_sequence[n]
		# print("den_emit shape: ", den_emit, np.shape(den_emit))
		# 计算trans_mat的分子
		trans_num_n = np.zeros(shape=(M, M))
		for i in range(M):
			for j in range(M):
				for t in range(T - 1):	# 6-27
					trans_num_n[i, j] += alphas[n][t, i] * trans_mat[i, j] * emit_mat[j, sequence[t+1]] * betas[n][t+1, j]
		trans_num += trans_num_n / p_sequence[n]

		emit_num_n = np.zeros(shape=(M, V))
		for j in range(M):
			for t in range(T):			# 6-28
				emit_num_n[j, sequence[t]] += alphas[n][t, j] * betas[n][t, j]
		emit_num += emit_num_n / p_sequence[n]

	trans_mat = trans_num / den_trans
	emit_mat = emit_num / den_emit
print("trans_mat:", trans_mat)
print("emit_mat:", emit_mat)
print("pi:", pi)


def likelihood(sequence):
	# returns log P(x | model)
	# 使用前向后向算法求得 P(Zt|model)
	T = len(sequence)
	alpha_ = np.zeros(shape=(T, M))
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
	delta = np.zeros((T, M))
	psi = np.zeros((T, M))
	delta[0] = pi * emit_mat[:, sequence[0]]
	for t in range(1, T):
		for j in range(M):
			delta[t, j] = np.max(delta[t - 1] * trans_mat[:, j]) * emit_mat[j, sequence[t]]
			psi[t, j] = np.argmax(delta[t - 1] * trans_mat[:, j])

	# 回溯
	states = np.zeros(T, dtype=np.int32)
	states[T - 1] = np.argmax(delta[T - 1])
	for t in range(T - 2, -1, -1):
		states[t] = psi[t + 1, states[t + 1]]
	return states


L = log_likelihood_multi(sequences).sum()
print("对于拟合参数求得的极大似然值:", L)


pi = np.array([0.5, 0.5])
A = np.array([[0.1, 0.9], [0.8, 0.2]])
B = np.array([[0.6, 0.4], [0.3, 0.7]])
L = log_likelihood_multi(sequences).sum()
print("对于真实参数求得的极大似然值:", L)

# viterbi
print("对于 ", sequences[0], " 最好的状态序列：")
print(get_state_sequence(sequences[0]))

plt.plot(costs)
plt.show()
