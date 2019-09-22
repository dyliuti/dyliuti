import numpy as np
import matplotlib.pyplot as plt

def random_normalized(d1, d2):
	x = np.random.random((d1, d2))
	return x / x.sum(axis=1, keepdims=True)

# 加了归一化
class HMM:
	def __init__(self, H):
		# 隐状态个数
		self.H = H
		
	def fit(self, sequences, epochs=30):
		# 观测量种类  硬币抛掷就两种观测结果   +1是因为序号从0开始
		V = max(max(sequence) for sequence in sequences) + 1
		sequences_len = len(sequences)

		# 初始化EM算法参数
		self.pi = np.ones(self.H) / self.H
		self.trans_mat = random_normalized(self.H, self.H)
		self.emit_mat = random_normalized(self.H, V)

		costs = []
		for epoch in range(epochs):
			alphas = []
			betas = []
			p_sequence = np.zeros(sequences_len)

			for n in range(sequences_len):
				sequence = sequences[n]
				T = len(sequence)
				scale = np.zeros(T)
				# 观测序列上，每个状态对应着各种观测结果概率  P(Z0) * P(Y0 | Z0)
				alpha = np.zeros(shape=(T, self.H))
				alpha[0] = self.pi * self.emit_mat[:, sequence[0]]	# 在观测序列已知的前提下，各状态转换为该序列观测的概率
				for t in range(1, T):
					# 观察者概率：P(Y0..Yt-1, Zt-1) * P(Zt | Zt-1) * P(Yt | Zt) -> P(Y0..Yt, Zt)
					# H, HxH -> H, H, -> H,
					alpha[t] = alpha[t - 1].dot(self.trans_mat) * self.emit_mat[:, sequence[t]]
				p_sequence[n] = alpha[-1].sum()
				alphas.append(alpha)

				beta = np.zeros(shape=(T, self.H))
				beta[-1] = 1
				for t in range(T - 2, -1, -1):  # 不包括-1 [ )
					# HxH H -> H
					beta[t] = self.trans_mat.dot(self.emit_mat[:, sequence[t + 1]] * beta[t + 1])
				betas.append(beta)

			cost = np.sum(np.log(p_sequence))
			costs.append(cost)
			# 利用期望 重新估计self.pi, self.trans_mat, self.emit_mat
			self.pi = np.sum((alphas[n][0] * betas[n][0]) / p_sequence[n] for n in range(sequences_len)) / sequences_len

			den_trans = np.zeros(shape=(self.H, 1))
			den_emit = np.zeros(shape=(self.H, 1))
			trans_num = 0
			emit_num = 0
			for n in range(sequences_len):
				sequence = sequences[n]
				T = len(sequence)

				# NxTxH alphas[n][: -1]: T-1xH -> 1xH -> Hx1
				den_trans += (alphas[n][: -1] * betas[n][: -1]).sum(axis=0, keepdims=True).T / p_sequence[n]
				# NxTxH alphas[n] TxH -> 1xH -> Hx1
				den_emit += (alphas[n] * betas[n]).sum(axis=0, keepdims=True).T / p_sequence[n]
				# print("den_emit shape: ", den_emit, np.shape(den_emit))
				# 计算self.trans_mat的分子
				trans_num_n = np.zeros(shape=(self.H, self.H))
				for i in range(self.H):
					for j in range(self.H):
						for t in range(T - 1):  # 6-27
							trans_num_n[i, j] += alphas[n][t, i] * self.trans_mat[i, j] * self.emit_mat[j, sequence[t + 1]] * \
												 betas[n][t + 1, j]
				trans_num += trans_num_n / p_sequence[n]

				emit_num_n = np.zeros(shape=(self.H, V))
				for j in range(self.H):
					for t in range(T):  # 6-28
						emit_num_n[j, sequence[t]] += alphas[n][t, j] * betas[n][t, j]
				emit_num += emit_num_n / p_sequence[n]

			self.trans_mat = trans_num / den_trans
			self.emit_mat = emit_num / den_emit
		print("trans_mat:", self.trans_mat)
		print("emit_mat:", self.emit_mat)
		print("pi:", self.pi)

		plt.plot(costs)
		plt.show()

	def likelihood(self, sequence):
		# returns log P(x | model)
		# 使用前向后向算法求得 P(Zt|model)
		T = len(sequence)
		alpha_ = np.zeros(shape=(T, self.H))
		alpha_[0] = self.pi * self.emit_mat[:, sequence[0]]
		for t in range(1, T):
			alpha_[t] = alpha_[t - 1].dot(self.trans_mat) * self.emit_mat[:, sequence[t]]
		return alpha_[-1].sum()

	def likelihood_multi(self, sequences):
		return np.array([self.likelihood(sequence) for sequence in sequences])

	def log_likelihood_multi(self, sequences):
		return np.log(self.likelihood_multi(sequences))

	def get_state_sequence(self, sequence):
		# 使用Viterbi算法，给定观测序列，返回最有可能的状态序列
		T = len(sequence)
		delta = np.zeros((T, self.H))
		psi = np.zeros((T, self.H))
		delta[0] = self.pi * self.emit_mat[:, sequence[0]]	# M种最优
		for t in range(1, T):
			for j in range(self.H):
				# delta[t, j] = np.max(delta[t - 1] * self.trans_mat[:, j]) * self.emit_mat[j, sequence[t]]
				# state_i = np.argmax(delta[t - 1] * self.trans_mat[:, j])
				# H, H, 1 -> H,  沿着路径到t, 状态j时，输出O1,O2...Ot的最大概率。
				delta[t, j] = np.max(delta[t - 1] * self.trans_mat[:, j] * self.emit_mat[j, sequence[t]])
				# 记录输出O1,O2...Ot的概率最大时， 记录下t位置,状态j时的前一个状态i（argmax的是M个i状态）
				state_i = np.argmax(delta[t - 1] * self.trans_mat[:, j] * self.emit_mat[j, sequence[t]])
				psi[t, j] = state_i
		# [1 0 1 0 1 1 0 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0] [2.42446276e-14 4.17974985e-14], [9.96230425e-14 8.20578832e-14]
		# [0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1] [3.12739011e-14 4.75255417e-14], [1.39170251e-14 5.36752937e-12]
		print(delta[T-1])
		# 回溯
		states = np.zeros(T, dtype=np.int32)
		states[T - 1] = np.argmax(delta[T - 1])
		for t in range(T - 2, -1, -1):
			states[t] = psi[t + 1, states[t + 1]]
		return states


# 加载数据
sequences = []
for line in open('Data/Markov/coin_data.txt'):
	# 1 for H, 0 for T
	sequence_ = [1 if e == 'H' else 0 for e in line.rstrip()]
	sequences.append(sequence_)

hmm = HMM(2)
hmm.fit(sequences)
L_sequences = hmm.log_likelihood_multi(sequences)
L = L_sequences.sum()
print("对于真实参数求得的极大似然值:", L)

hmm.pi = np.array([0.5, 0.5])
hmm.A = np.array([[0.1, 0.9], [0.8, 0.2]])
hmm.B = np.array([[0.6, 0.4], [0.3, 0.7]])
L_true = hmm.log_likelihood_multi(sequences)
L = L_true.sum()
print("对于真实参数求得的极大似然值:", L)

# viterbi
print("对于 ", sequences[0], " 最好的状态序列：")
print(hmm.get_state_sequence(sequences[0]))

