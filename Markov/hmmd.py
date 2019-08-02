import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def random_normalized(d1, d2):
	x = np.random.random((d1, d2))
	return x / x.sum(axis=1, keepdims=True)

# 加载数据
X = []
for line in open('coin_data.txt'):
	# 1 for H, 0 for T
	x = [1 if e == 'H' else 0 for e in line.rstrip()]
	X.append(x)

M = 2

def fit(sequences, epochs=30):
	# 观测量种类  硬币抛掷就两种观测结果
	V = max(max(sequence) for sequence in sequences) + 1
	sequences_num = len(sequences)

	# 初始化EM算法参数
	pi = np.ones(M) / M
	trans_mat = random_normalized(M, M)
	emit_mat = random_normalized(M, V)

	costs = []
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
			p_sequence = alpha[-1].sum()
			alphas.append(alpha)

			beta = np.zeros(shape=(T, M))
			beta[-1] = 1
			for t in range(T-2, -1, -1):	# 不包括-1 [ )
				# MxM M -> M
				beta[t] = trans_mat.dot(emit_mat[:, sequence[t+1]] * beta[t+1])
				tmp = np.zeros(M)
				for i in range(M):
					for j in range(M):
						tmp[i] += trans_mat[i][j] * emit_mat[j][t+1] * beta[t+1]
				print("diff:", np.abs(beta[t] - tmp).sum())
			betas.append(beta)

		cost = np.sum(np.log(p_sequence))
		costs.append(cost)
		# 重新估计pi, trans_mat, emit_mat
		pi = np.sum((alphas[n][0] * betas[n][0]) / p_sequence[n] for n in range(sequences_num)) / sequences_num

		den_trans = np.zeros(shape=(M, 1))
		den_emit = np.zeros(shape=(M, 1))
		trans_num = 0
		emit_num = 0
		for n in range(sequences_num):
			sequence = sequences[n]
			T = len(sequence)

			den_trans += (alphas[n][: -1] * betas[n][: -1]).sum(axis=0, keepdims=True).T / p_sequence[n]
			den_emit += (alphas[n] * betas[n]).sum(axis=0, keepdims=True).T / p_sequence[n]

			# 计算trans_mat的分子
			trans_num_n = np.zeros(shape=(M, M))
			for i in range(M):
				for j in range(M):
					for t in range(T - 1):
						trans_num_n[i, j] += alphas[n][t, i] * trans_mat[i, j] * emit_mat[j, sequence[t+1]] * betas[n][t+1, j]
			trans_num += trans_num_n / p_sequence[n]

			emit_num_n = np.zeros(shape=(M, V))
			for i in range(M):
				for t in range(T):
					emit_num_n[i, sequence[t]] += alphas[n][t, i] * betas[n][t, i]
			emit_num += emit_num_n / p_sequence[n]

		trans_mat = trans_num / den_trans
		emit_mat = emit_num / den_emit
	print("trans_mat:", trans_mat)
	print("emit_mat:", emit_mat)
	print("pi:", pi)


