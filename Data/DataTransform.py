import numpy as np

# 将data进行one_hot编码
def y2one_hot(data, class_num=1):
	sample_num = len(data)
	res = np.zeros(shape=(sample_num, class_num))
	data = data.astype(np.int32)
	for i in range(sample_num):
		res[i, data[i]] = 1
	return res.astype('float32')

def softmax(a):
	a = a - a.max()
	exp_a = np.exp(a)
	return exp_a / exp_a.sum(axis=1, keepdims=True)


def smoothed_loss(x, decay=0.99):
	y = np.zeros(len(x))
	last = 0
	for t in range(len(x)):
		z = decay * last + (1 - decay) * x[t]
		y[t] = z / (1 - decay ** (t + 1))
		last = z
	return y

def q_to_b(q_str):
	"""全角转半角"""
	b_str = ""
	for uchar in q_str:
		# ord是 chr函数的配对函数，它以一个字符串（Unicode 字符）作为参数，返回对应的 ASCII 数值，或者 Unicode 数值。
		inside_code = ord(uchar)
		if inside_code == 12288:  # 全角空格直接转换
			inside_code = 32
		elif 65374 >= inside_code >= 65281:  # 全角字符（除空格）根据关系转化
			inside_code -= 65248
		b_str += chr(inside_code)
	return b_str

def b_to_q(b_str):
	"""半角转全角"""
	q_str = ""
	for uchar in b_str:
		inside_code = ord(uchar)
		if inside_code == 32:  # 半角空格直接转化
			inside_code = 12288
		elif 126 >= inside_code >= 32:  # 半角字符（除空格）根据关系转化
			inside_code += 65248
		q_str += chr(inside_code)
	return q_str

def read_corpus_from_file(file_path):
	"""读取语料"""
	f = open(file_path, 'r', encoding='utf-8')
	lines = f.readlines()
	f.close()
	return lines

def write_corpus_to_file(data, file_path):
	"""写语料"""
	f = open(file_path, 'wb')
	f.write(data)
	f.close()

def pos_to_tag(_maps, p):
	"""由词性提取标签"""
	t = _maps.get(p, None)
	return t if t else u'O'

def tag_perform(tag, index):
	"""标签使用BIO模式"""
	if index == 0 and tag != u'O':
		return u'B_{}'.format(tag)
	elif tag != u'O':
		return u'I_{}'.format(tag)
	else:
		return tag

def pos_perform(_maps, pos):
	"""去除词性携带的标签先验知识"""
	if pos in _maps.keys() and pos != u't':
		return u'n'
	else:
		return pos

def extract_feature(word_grams):
	"""特征选取"""
	features, feature_list = [], []
	for index in range(len(word_grams)):
		for i in range(len(word_grams[index])):
			word_gram = word_grams[index][i]
			feature = {u'w-1': word_gram[0], u'w': word_gram[1], u'w+1': word_gram[2],
					   u'w-1:w': word_gram[0] + word_gram[1], u'w:w+1': word_gram[1] + word_gram[2],
					   # u'p-1': self.pos_seq[index][i], u'p': self.pos_seq[index][i+1],
					   # u'p+1': self.pos_seq[index][i+2],
					   # u'p-1:p': self.pos_seq[index][i]+self.pos_seq[index][i+1],
					   # u'p:p+1': self.pos_seq[index][i+1]+self.pos_seq[index][i+2],
					   u'bias': 1.0}
			feature_list.append(feature)
		features.append(feature_list)
		feature_list = []
	return features

def segment_by_window(words_list=None, window=3):
	"""窗口切分"""
	words = []
	begin, end = 0, window
	for _ in range(1, len(words_list)):
		if end > len(words_list):
			break
		words.append(words_list[begin: end])
		begin = begin + 1
		end = end + 1
	return words


# hard labels
# purity最高为1, 越高越好
def purity2(Y, R):
	C = np.argmax(R, axis=1) # cluster数

	N = len(Y) # 数据量
	K = len(set(Y)) # 分类数

	total = 0.0
	for k in range(K):
		max_intersection = 0
		for j in range(K):
			intersection = ((C == k) & (Y == j)).sum()
			if intersection > max_intersection:
				max_intersection = intersection
		total += max_intersection
	return total / N

# purity最高为1, 越高越好
def purity(Y, R):
	N, K = R.shape
	p = 0
	for k in range(K):
		best_target = -1
		max_intersection = 0
		for j in range(K):
			intersection = R[Y==j, k].sum()
			if intersection > max_intersection:
				max_intersection = intersection
				best_target = j
		p += max_intersection
	return p / N


# hard labels
def DBI2(X, R):
	N, D = X.shape
	_, K = R.shape

	# 先计算sigmas, means first
	sigma = np.zeros(K)
	M = np.zeros((K, D))
	assignments = np.argmax(R, axis=1)
	for k in range(K):
		Xk = X[assignments == k]
		M[k] = Xk.mean(axis=0)
		# assert(Xk.mean(axis=0).shape == (D,))
		n = len(Xk)
		diffs = Xk - M[k]
		sq_diffs = diffs * diffs
		sigma[k] = np.sqrt( sq_diffs.sum() / n )

	# 计算 Davies-Bouldin Index
	dbi = 0
	for k in range(K):
		max_ratio = 0
		for j in range(K):
			if k != j:
				numerator = sigma[k] + sigma[j]
				denominator = np.linalg.norm(M[k] - M[j])
				ratio = numerator / denominator
				if ratio > max_ratio:
					max_ratio = ratio
		dbi += max_ratio
	return dbi / K



def DBI(X, M, R):
	# 2个聚类间std偏差和/聚类均值间距离的比值 越低越好
	N, D = X.shape
	K, _ = M.shape

	# 先计算sigmas
	sigma = np.zeros(K)
	for k in range(K):
		diffs = X - M[k]
		squared_distances = (diffs * diffs).sum(axis=1)
		weighted_squared_distances = R[:,k]*squared_distances
		sigma[k] = np.sqrt( weighted_squared_distances.sum() / R[:,k].sum() )

	# 计算 Davies-Bouldin Index
	dbi = 0
	for k in range(K):
		max_ratio = 0
		for j in range(K):
			if k != j:
				numerator = sigma[k] + sigma[j]
				denominator = np.linalg.norm(M[k] - M[j])
				ratio = numerator / denominator
				if ratio > max_ratio:
					max_ratio = ratio
		dbi += max_ratio
	return dbi / K



