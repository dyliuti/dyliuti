import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


def init_weight(M1, M2):
	return np.random.randn(M1, M2) * np.sqrt(2.0 / M1)

def init_weight_and_bias(M1, M2):
	return np.random.randn(M1, M2) / np.sqrt(M2), np.zeros(M2)


class GloveVectorizer(object):
	def __init__(self):
		# load in pre-trained word vectors
		self.word2vec = {}
		self.index2word = []
		self.embedding = []
		with open('Data/NLP/glove.6B/glove.6B.50d.txt', encoding='utf-8') as f:
			for line in f:
				values = line.split()
				word = values[0]
				vec = np.asarray(values[1:], dtype='float32')
				self.word2vec[word] = vec
				self.index2word.append(word)
				self.embedding.append(vec)
		print("Word vector size: %f." % len(self.word2vec))
		self.embedding = np.array(self.embedding)
		self.V, self.D = self.embedding.shape

	def fit(self, sentences):
		pass

	# 返回句子中所有词向量的加权平均
	def transform(self, sentences):
		X = np.zeros((len(sentences), self.D))
		n = 0
		for sentence in sentences:
			words = sentence.lower().split()
			vecs = []
			for word in words:
				if word in self.word2vec:
					vec = self.word2vec[word]
					vecs.append(vec)
			vecs = np.array(vecs)
			X[n] = vecs.mean(axis = 0) # 词袋模型-评一个句子的相似度，用句子中所有词向量的加权平均
			n += 1
		return X

	def fit_transform(self, sentences):
		self.fit(sentences)
		return self.transform(sentences)

class Word2VecVectorizer(object):
	def __init__(self):
		self.embedding = KeyedVectors.load_word2vec_format(
			'Data/NLP/GoogleNews-vectors-negative300.bin',
			binary=True
		)

	def fit(self, sentences):
		pass

	# 返回句子中所有词向量的加权平均
	def transform(self, sentences):
		v = self.embedding.get_vector('king')
		self.D = v.shape[0]

		X = np.zeros((len(sentences), self.D))
		n = 0
		for sentence in sentences:
			words = sentence.lower().split()
			vecs = []
			for word in words:
				try:
					vec = self.embedding.get_vector(word)
					vecs.append(vec)
				except KeyError:
					pass
			vecs = np.array(vecs)
			X[n] = vecs.mean(axis = 0) # 词袋模型-评一个句子的相似度，用句子中所有词向量的加权平均
			n += 1
		return X

	def fit_transform(self, sentences):
		self.fit(sentences)
		return self.transform(sentences)
