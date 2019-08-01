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

	for epoch in range(epochs):
		alpha = []
		beta = []
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