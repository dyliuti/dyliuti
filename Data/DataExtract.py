import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from nltk.corpus import brown
from nltk import pos_tag, word_tokenize
import string
from glob import glob
import tarfile, re

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
		# test1 = pca.explained_variance_
		# test2 = pca.explained_variance_ratio_
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
	plt.show()
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

def remove_punctuation(s):
	return s.translate(str.maketrans('', '', string.punctuation))

# 返回indexed_sentences: 用索引表示的句子的集合, 频数不在前n_vocab个的词转换为'UNKNOW'的索引
# 返回word2index: 字符 词 映射到索引， 额外包括'UNKNOW'
def load_wiki_with_limit_vocab(n_vocab=20000):
	path = 'Data/NLP/WikiData/'
	dirs = os.listdir(path)
	files = [glob(path + dir_path + '/wiki*') for dir_path in dirs]
	all_word_counts = {}
	for f_list in files:  # str
		for f in f_list:
			for line in open(f, encoding='utf-8'):  # str
				if line and line[0] not in '[*-|=\{\}<':
					s = remove_punctuation(line).lower().split()
					if len(s) > 1:
						for word in s:
							if word not in all_word_counts:
								all_word_counts[word] = 0
							all_word_counts[word] += 1
	print("finished counting")

	n_vocab = min(n_vocab, len(all_word_counts))

	# 按词出现频数进行降序排序
	all_word_counts = sorted(all_word_counts.items(), key=lambda k: k[1], reverse=True)
	# 选取前n_vocab个词，并生成word2index
	top_words  = [w for w, count in all_word_counts[: n_vocab - 1]] + ['UNKNOW']
	word2index = {w: i for i, w in enumerate(top_words)}
	unkonw_index = word2index['UNKNOW']

	# 将句子中的单词转换为索引，频数不在前n_vocab个的词转换为'UNKNOW'的索引
	indexed_sentences = []
	for f_list in files:
		for f in f_list:
			for line in open(f, encoding='utf-8'):
				if line and line[0] not in '[*-|=\{\}':
					s = remove_punctuation(line).lower().split()
					if len(s) > 1:
						# 如果无context，就不训练
						sent = [word2index[w] if w in word2index else unkonw_index for w in s]  # word embedding
						indexed_sentences.append(sent)
	return indexed_sentences, word2index

def load_robert_frost():
	word2index = {'START': 0, 'END': 1}
	current_index = 2
	sentences = []
	for line in open('Data/NLP/robert_frost.txt', 'r', encoding='utf-8'):
		line = line.strip()
		if line:
			tokens = remove_punctuation(line.lower()).split()
			sentence = []
			for t in tokens:
				if t not in word2index:
					word2index[t] = current_index
					current_index += 1
				idx = word2index[t]
				sentence.append(idx)
			sentences.append(sentence)
	return sentences, word2index

def load_robert_frost_soseos():
	input_sentences = []
	output_sentences = []
	for line in open('Data/NLP/robert_frost.txt', 'r', encoding='utf-8'):
		line = line.strip()
		if line:
			# 前后空格别忘了
			input_sentence = '<sos> ' + line
			output_sentence = line + ' <eos>'
			input_sentences.append(input_sentence)
			output_sentences.append(output_sentence)
	return input_sentences, output_sentences

# 分词后对词性进行标注
def get_tags(s):
	tuples = pos_tag(word_tokenize(s))
	return [y for x, y in tuples]

def load_poetry_classifier_data(samples_per_class, load_cached=False, save_cached=True):
	datafile = 'Data/NLP/poetry_classifier_data.npz'
	if load_cached and os.path.exists(datafile):
		npz = np.load(datafile)
		X = npz['arr_0']
		Y = npz['arr_1']
		V = int(npz['arr_2'])
		return X, Y, V

	tag2index = {}
	current_index = 0
	X = []
	Y = []
	for fn, label in zip(('Data/NLP/edgar_allan_poe.txt', 'Data/NLP/robert_frost.txt'), (0, 1)):
		count = 0
		for line in open(fn, encoding='utf-8'):
			line = line.rstrip()
			if line:
				print(line)
				# tokens = remove_punctuation(line.lower()).split()
				tags = get_tags(line)
				if len(tags) > 1:
					# scan doesn't work nice here, technically could fix...
					for tag in tags:
						if tag not in tag2index:
							tag2index[tag] = current_index
							current_index += 1
					sequence = np.array([tag2index[t] for t in tags])
					X.append(sequence)  # 一个sequence代表一行
					Y.append(label)
					count += 1          # 行数 从1开始
					print(count, label)
					# quit early because the tokenizer is very slow
					if count >= samples_per_class:
						break
	if save_cached:
		np.savez(datafile, X, Y, current_index)
	return X, Y, current_index

# train都是词的索引，test都是词对应的tag的索引
def load_chunking(split_sequence=True):
	train_file = 'Data/NLP/chunking/train.txt'
	test_file = 'Data/NLP/chunking/test.txt'
	word2index, tag2index = {}, {}
	word_index, tag_index = 0, 0
	X_train, Y_train, current_X, current_Y = [], [], [], []
	for line in open(train_file, encoding='utf-8'):
		line = line.rstrip()	# 删除右侧空字符
		if line:
			r = line.split()
			word, tag, _ = r
			if word not in word2index:
				word2index[word] = word_index
				word_index += 1
			current_X.append(word2index[word])

			if tag not in tag2index:
				tag2index[tag] = tag_index
				tag_index += 1
			current_Y.append(tag2index[tag])
		elif split_sequence:
			X_train.append(current_X)
			Y_train.append(current_Y)
			current_X, current_Y = [], []
	if not split_sequence:
		X_train = current_X
		Y_train = current_Y

	# 加载测试集
	X_test, Y_test, current_X, current_Y = [], [], [], []
	for line in open(test_file, encoding='utf-8'):
		line = line.rstrip()  # 删除右侧空字符
		if line:
			r = line.split()
			word, tag, _ = r
			if word in word2index:
				current_X.append(word2index[word])
			else:
				current_X.append(word_index)	# 将最后词索引当做'UNKNOW'索引
			current_Y.append(tag2index[tag])
		elif split_sequence:
			X_test.append(current_X)
			Y_test.append(current_Y)
			current_X, current_Y = [], []
	if not split_sequence:
		X_test = current_X
		Y_test = current_Y

	return X_train, Y_train, X_test, Y_test, word2index


# 返回word，tag 非索引
def load_ner(split_sequence=True):
	file = 'Data/NLP/ner.txt'
	words, tags, word_list, tag_list = [], [], [], []
	for line in open(file, encoding='utf-8'):
		line = line.rstrip()  # 删除右侧空字符
		if line:
			r = line.split()
			word, tag = r
			word = word.lower()
			word_list.append(word)
			tag_list.append(tag)
		elif split_sequence:
			words.append(word_list)
			tags.append(tag_list)
			word_list = []
			tag_list = []
	if not split_sequence:
		words = word_list
		tags = tag_list

	return words, tags


class Tree:
	def __init__(self, word_index, label):
		self.left = None
		self.right = None
		self.word_index = word_index
		self.label = label


current_idx = 0
def str2tree(s, word2index):
	# take a string that starts with ( and MAYBE ends with )
	# return the tree that it represents
	# EXAMPLE: "(3 (2 It) (4 (4 (2 's) (4 (3 (2 a) (4 (3 lovely) (2 film))) (3 (2 with) (4 (3 (3 lovely) (2 performances)) (2 (2 by) (2 (2 (2 Buy) (2 and)) (2 Accorsi))))))) (2 .)))"
	# NOTE: not every node has 2 children (possibly not correct ??)
	# NOTE: not every node has a word
	# NOTE: every node has a label
	# NOTE: labels are 0,1,2,3,4
	# NOTE: only leaf nodes have words
	# s[0] = (, s[1] = label, s[2] = space, s[3] = character or (

	global current_idx

	label = int(s[1])
	if s[3] == '(':
		# 前序遍历
		t = Tree(None, label)
		child_s = s[3:]
		t.left = str2tree(child_s, word2index)

		i = 0
		depth = 0
		for c in s:
			i += 1
			if c == '(':
				depth += 1
			elif c == ')':
				depth -= 1
				if depth == 1:
					break
		# print "index of right child", i

		t.right = str2tree(s[i+1:], word2index)
		return t
	else:
		# this has a word, so it's a leaf
		r = s.split(')', 1)[0]
		word = r[3:].lower()
		# print "word found:", word

		if word not in word2index:
			word2index[word] = current_idx
			current_idx += 1

		t = Tree(word2index[word], label)
		return t


# word2index: word到index的映射
# train: 以树表示的句子
def load_parse_tree():
	word2index = {}
	train, test = [], []

	# train set first
	for line in open('Data/NLP/trees/train.txt'):
		line = line.rstrip()
		if line:
			t = str2tree(line, word2index)
			train.append(t)
			# break

	# test set
	for line in open('Data/NLP/trees/test.txt'):
		line = line.rstrip()
		if line:
			t = str2tree(line, word2index)
			test.append(t)
	return train, test, word2index


def load_glove6B(dimension):
	word2vec = {}
	with open('Data/NLP/glove.6B/glove.6B.%sd.txt' % dimension, encoding='utf-8') as f:
		for line in f:
			values = line.split()
			word = values[0]
			vec = np.asarray(values[1:], dtype='float32')
			word2vec[word] = vec
	return word2vec

def load_translation(file_name='jpn.txt', sample_num=float('inf')):
	file = 'Data/NLP/translation/' + file_name
	input_texts = []
	translation_inputs = []
	translation_outputs = []
	num = 0
	for line in open(file, encoding='utf-8'):
		num += 1
		if num > sample_num:
			break

		if '\t' not in line:
			continue

		# 分离输入句子和输出的翻译句子
		input_text, translation = line.rstrip().split('\t')
		translation_input = '<sos> ' + translation
		translation_output = translation + ' <eos>'

		input_texts.append(input_text)
		translation_inputs.append(translation_input)
		translation_outputs.append(translation_output)

	return input_texts, translation_inputs, translation_outputs

def tokenize(sent):
	# tokenize('Bob dropped the apple. Where is the apple?')
	# ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
	return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
}

# 数据集：序号 句子   若有tab，句子就表示提问，后面还有回答  答案所在的句子序号， 不过这个序号不用
# 若句子中没tab，就是故事story
# 一个故事中有多个提问
# 返回：data每项都是对应着一个提问，以及提问前的故事（不包括提问，但提问占序号）
def load_bAbI_challange_data(challenge_type='single_supporting_fact_10k', data_type='train'):
	path = 'Data/NLP/Memory/tasks.tar.gz'
	tar = tarfile.open(path)
	challenge = challenges[challenge_type]
	f = tar.extractfile(challenge.format(data_type))

	data = []
	# 保持到目前提问前的story
	story = []
	for line in f:
		line = line.decode('utf-8').strip()
		nid, sentence = line.split(' ', 1)
		# 新的story
		if int(nid) == 1:
			story = []

		if '\t' not in sentence:
			# 添加story句
			story.append(tokenize(sentence))
		else:
			# 处理问答句
			question, answer, supporting = sentence.split('\t')
			question = tokenize(question)
			# 去除前面提问(if s)后的story  句子所在序号比文本中的少1
			story_so_far = [[str(i)] + s for i, s in enumerate(story) if s]
			data.append((story_so_far, question, answer))
			# sotry还是要添加，保持行数对应
			story.append('')
	return data
