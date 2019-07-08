from Data.DataExtract import load_wiki_with_limit_vocab, load_brown_with_limit_vocab
from NLP.Common.Util import analogy
import numpy as np
import matplotlib.pyplot as plt
import os, json


class Glove:
	def __init__(self, D, V, context_size):
		self.D = D
		self.V = V
		self.context_size = context_size

	def build_cc_matrix(self, indexed_sentences):
		X = np.zeros(shape=(self.V, self.V))
		# 建立共现矩阵
		for sentence in indexed_sentences:
			n = len(sentence)
			for i in range(n):
				word_i_index = sentence[i]

				start_index = max(0, i - self.context_size)
				end_index = min(n, i + self.context_size)

				if i - self.context_size < 0:
					points = 1.0 / (i + 1)
					X[word_i_index, 0] += points
					X[0, word_i_index] += points
				if i + self.context_size > n:
					points = 1.0 / (n - i)
					X[word_i_index, 1] += points
					X[1, word_i_index] += points

				# 句子左半部分
				for j in range(start_index, i):
					word_j_index = sentence[j]
					points = 1.0 / (i - j)
					X[word_i_index, word_j_index] += points
					X[word_j_index, word_i_index] += points
				# 句子右半部分
				for j in range(i + 1, end_index):
					word_j_index = sentence[j]
					points = 1.0 / (j - i)
					X[word_i_index, word_j_index] += points
					X[word_j_index, word_i_index] += points
		return X

	# 建立共现矩阵cc_matrix
	def fit(self, indexed_sentences, cc_matrix=None, learning_rate=1e-4, reg=0.1, xmax=100, alpha=0.75, epochs=10, gradient=False):
		V = self.V
		D = self.D
		X = self.build_cc_matrix(indexed_sentences)

		# 损失函数权重
		fX = np.zeros(shape=(V, V))
		fX[X < xmax] = (X[X < xmax] / float(xmax)) ** alpha
		# 目标函数
		logX = np.log(X + 1)  # 非Nan
		# 初始化参数
		W = np.random.randn(V, D) / np.sqrt(D)
		b = np.zeros(V)
		U = np.random.randn(V, D) / np.sqrt(D)
		c = np.zeros(V)
		mu = np.mean(logX)

		costs = []
		for epoch in range(epochs):
			diff = W.dot(U.T) + b.reshape(V, 1) + c.reshape(1, V) + mu - logX
			cost = np.sum(fX * diff * diff)
			costs.append(cost)
			print("epoch: ", epoch, "cost: ", cost)

		if gradient is True:
			for i in range(V):
				W[i] -= learning_rate  * (fX[i, :] * diff[i, :]).dot(U) #D  1 V .dot V D -> D
			W -= learning_rate * reg * W

			for i in range(V):
				b[i] -= learning_rate * fX[i, :].dot(diff[i, :])

			for j in range(V):
				U[j] -= learning_rate * (fX[:, j] * diff[:, j]).dot(W)
			U -= learning_rate * reg * U

			for j in range(V):
				c[j] -= learning_rate * fX[:, j].dot(diff[:, j])
		else:
			for i in range(V):
				matrix = reg * np.eye(D) + (fX[i, :] * U.T).dot(U)  # 我认为 V * (D, V) -> (D,V).dot( (V, D) )-> D,D
				vector = (fX[i, :] * (logX[i, :] - b[i] - c - mu)).dot(U)  # 1 V.dot( V, D) -> (D, )   1 D
				# 用linalg.solve， matrix必须是正方矩阵(方阵才有机会可逆)，W shape与 vector相同
				W[i] = np.linalg.solve(matrix, vector)  # (D,)  1 D

			for i in range(V):
				denominator = fX[i, :].sum() + reg
				numerator = fX[i, :].dot(logX[i, :] - W[i].dot(U.T) - c - mu)
				b[i] = numerator / denominator

			for j in range(V):
				matrix = reg * np.eye(D) + (fX[:, j] * W.T).dot(W)
				vector = (fX[:, j] * (logX[:, j] - b - c[j] - mu)).dot(W)
				U[j] = np.linalg.solve(matrix, vector)

			for j in range(V):
				denominator = fX[:, j].sum() + reg
				numerator = fX[:, j].dot(logX[:, j] - W.dot(U[j]) - b - mu)
				c[j] = numerator / denominator

		self.W = W
		self.U = U

		plt.plot(costs)
		plt.show()

	def save(self, file):
		arrays = [self.W, self.U.T]
		np.savez(file, *arrays)


def test_model(word2index, W1, W2):
	index2word = {i: w for w, i in word2index.items()}
	# 也可以试 We = W2.T
	for word_embedding in (W1, (W1 + W2.T) / 2):
		print("**********")

		analogy('king', 'man', 'queen', 'woman', word2index, index2word, word_embedding)
		analogy('king', 'prince', 'queen', 'princess', word2index, index2word, word_embedding)
		analogy('miami', 'florida', 'dallas', 'texas', word2index, index2word, word_embedding)
		analogy('einstein', 'scientist', 'picasso', 'painter', word2index, index2word, word_embedding)
		analogy('japan', 'sushi', 'germany', 'bratwurst', word2index, index2word, word_embedding)
		analogy('man', 'woman', 'he', 'she', word2index, index2word, word_embedding)

def main(word_embedding_file, word2vec_file, files_num=100):
	cc_matrix = "cc_matrix_%s.npy" % files_num
	word2index = {}
	if os.path.exists(cc_matrix):
		with open(word2vec_file) as f:
			word2index = json.load(f)
	else:
		sentences, word2index = load_wiki_with_limit_vocab(2000)
		with open(word2vec_file) as f:
			json.dump(word2index, f)

	V = len(word2index)
	model = Glove(100, V, 10)
	# 选择用alternating least squares method
	model.fit(sentences, cc_matrix, epochs=2)
	model.save(word_embedding_file)

if __name__ == "__main__":
	word_emebdding_file = 'glove_model_50.npz'
	word2index_file = 'glove_word2idx_50.json'

	main(word_emebdding_file, word2index_file)

	npz = np.load(word_emebdding_file)
	W = npz['arr_0']
	U = npz['arr_1']

	with open(word2index_file) as f:
		word2index = json.load(f)

		test_model(word2index, W, U)


