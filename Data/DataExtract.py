import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from nltk.corpus import brown

# 返回标准化了的训练集与测试集
def load_minist_csv(pca=True):
	train_file_path = 'Data/Minist/train.csv'
	if not os.path.exists(train_file_path):
		print('%s not exist.' % train_file_path)

	train_file = pd.read_csv(train_file_path)
	train_np = train_file.values.astype(np.float32)		# float 不是 int 是有原因的
	np.random.shuffle(train_np)

	Y = train_file['label'].values.astype(np.int32)		# seris -> np
	X_pd = train_file.drop('label', axis=1)
	X = X_pd.values.astype(np.float32) / 255.0		# Min-Max Scaling -> normalization
	# X = train_file.values[:, 1:]  # 同上X, 但没标准化

	# 训练集中分训练、验证集
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
	# PCA降维，丢弃杂质
	if pca is True:
		pca = PCA()
		X_train = pca.fit_transform(X_train)
		X_test = pca.transform(X_test)
		plot_cumulative_variance(pca)
	# 各自Normalize服从正态分布   梯度图变圆有利于梯度下降
	X_train = (X_train - np.mean(X_train)) / np.std(X_train)
	X_test = (X_test - np.mean(X_test)) / np.std(X_test)
	return X_train, X_test, Y_train, Y_test

def plot_cumulative_variance(pca):
	P = []
	# 用奇异值S来解释方差
	for p in pca.explained_variance_ratio_:
		if len(P) == 0:
			P.append(p)
		else:
			P.append(p + P[-1])
	plt.plot(P)
	plt.show
	return P

def load_facial_expression_data(balance_ones=True):
	# images are 48x48 = 2304 size vectors
	train_file_path = 'Data/FacialExpression/fer2013.csv'
	if not os.path.exists(train_file_path):
		print('%s not exist.' % train_file_path)

	Y = []
	X = []
	first = True
	for line in open(train_file_path):
		if first:
			first = False
		else:
			row = line.split(',')
			Y.append(int(row[0]))
			X.append([int(p) for p in row[1].split()])

	X, Y = np.array(X) / 255.0, np.array(Y)

	if balance_ones:
		# balance the 1 class
		X0, Y0 = X[Y!=1, :], Y[Y!=1]
		X1 = X[Y==1, :]
		X1 = np.repeat(X1, 9, axis=0)
		X = np.vstack([X0, X1])
		Y = np.concatenate((Y0, [1]*len(X1)))

	return X, Y

# 返回indexed_sentences: 用索引表示的句子的集合, 不包括额外添加的'START', 'END'
# 返回word2index: 字符 词 映射到索引， 包括'START', 'END'
def load_brown():
	sentences = brown.sents()

	index = 2
	indexed_sentences = []
	word2index = {'START': 0, 'END': 1}
	for sentence in sentences:
		indexed_sentence = []
		for word in sentence:
			if word not in word2index:
				word2index[word] = index
				index += 1

			indexed_sentence.append(word2index[word])
		indexed_sentences.append(indexed_sentence)

	return indexed_sentences, word2index


KEEP_WORDS = set(['king', 'man', 'queen', 'woman',
				  'italy', 'rome', 'france', 'paris',
				  'london', 'britain', 'england',])

# 保留出现次数最多的前n_vocab个词
def load_brown_with_limit_vocab(n_vocab=2000, keep_words=KEEP_WORDS):
	sentences = brown.sents()

	index = 2
	indexed_sentences = []
	word2index = {'START': 0, 'END': 1}
	index2word = ['START', 'END']
	# 保留'START'与'END'，词索引映射词数量
	word_index2count = {
		0: float('inf'),
		1: float('inf')
	}
	for sentence in sentences:
		indexed_sentence = []
		for word in sentence:
			word = word.lower()		# 必须的，不然word2index['italy']会报keyValueError异常
			if word not in word2index:
				index2word.append(word)
				word2index[word] = index
				index += 1

			# 若词索引不在字典key中，value置0；否则，出现次数+1
			word_index = word2index[word]
			word_index2count[word_index] = word_index2count.get(word_index, 0) + 1
			indexed_sentence.append(word2index[word])
		indexed_sentences.append(indexed_sentence)

	# 保留需保留的词
	for word in keep_words:
		word_index2count[word2index[word]] = float('inf')
	# 取出现次数最多的n_vocab个词
	sorted_word_index2count = sorted(word_index2count.items(), key=lambda k: k[1], reverse=True)
	# sorted_word_idx2count = sorted(word_index2count.items(), key=operator.itemgetter(1), reverse=True)

	word2index_limit = {}
	new_index = 0
	old_index2new_index = {}

	# 生成新的word2index； 同时生成old_index2new_index，为句子索引转换做准备
	for index, count in sorted_word_index2count[:n_vocab]:
		word = index2word[index]
		word2index_limit[word] = new_index
		old_index2new_index[index] = new_index
		new_index += 1
	# 'UNKNOW'为最后一个词
	word2index_limit['UNKNOW'] = new_index # n_vocab
	unknow_index = new_index

	# 将句子中的‘旧索引’更新为'新索引'
	sentences_limit = []
	for sentence in indexed_sentences:
		if len(sentence) > 1:
			new_sentence = [old_index2new_index[word_index]
							if word_index in old_index2new_index else unknow_index
							for word_index in sentence]
			sentences_limit.append(new_sentence)

	return  sentences_limit, word2index_limit


