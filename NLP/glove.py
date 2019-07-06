from Data.DataExtract import load_wiki_with_limit_vocab, load_brown_with_limit_vocab
import numpy as np

class Glove:
	def __init__(self, D, V, context_size):
		self.D = D
		self.V = V
		self.context_size = context_size

	# 建立共现矩阵cc_matrix
	def fit(self, indexed_sentences, cc_matrix=None, learning_rate=1e-4, reg=0.1, xmax=100, alpha=0.75, epochs=10, gradient=False):
		V = self.V
		D = self.D
		X = np.zeros(shape=(V, V))

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
				if i + self.context_size  > n:
					points = 1.0 / (n - i)
					X[word_i_index, 1] += points
					X[1, word_i_index] += points
					
				# 句子做半部分
				for j in range(start_index, i):
					word_j_index = sentence[j]
					points = 1.0 / (i - j)
					X[word_i_index, word_j_index] += points
					X[word_j_index, word_i_index] += points
				for j in range(i + 1, end_index):
					word_j_index = sentence[j]
					points = 1.0 / (j - i)
					X[word_i_index, word_j_index] += points
					X[word_j_index, word_i_index] += points


