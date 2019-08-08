import re
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.externals import joblib
from Data.DataTransform import q_to_b, b_to_q, read_corpus_from_file, write_corpus_to_file, pos_to_tag, tag_perform, \
	pos_perform, extract_feature, segment_by_window

def process_k(words):
	"""处理大粒度分词,合并语料库中括号中的大粒度分词,类似：[国家/n  环保局/n]nt """
	pro_words = []
	index = 0
	temp = u''
	while True:
		word = words[index] if index < len(words) else u''
		if u'[' in word:
			temp += re.sub(pattern=u'/[a-zA-Z]*', repl=u'', string=word.replace(u'[', u''))
		elif u']' in word:
			w = word.split(u']')
			temp += re.sub(pattern=u'/[a-zA-Z]*', repl=u'', string=w[0])
			pro_words.append(temp + u'/' + w[1])
			temp = u''
		elif temp:  # [后面跟着的，但又不含]的词
			temp += re.sub(pattern=u'/[a-zA-Z]*', repl=u'', string=word)
		elif word:	# 非大粒度分词
			pro_words.append(word)
		else:
			break
		index += 1
	return pro_words

def process_nr(words):
	""" 处理姓名，合并语料库分开标注的姓和名，类似：温/nr  家宝/nr"""
	pro_words = []
	index = 0
	while True:
		word = words[index] if index < len(words) else u''
		if u'/nr' in word:
			next_index = index + 1
			if next_index < len(words) and u'/nr' in words[next_index]:
				pro_words.append(word.replace(u'/nr', u'') + words[next_index])
				index = next_index
			else:
				pro_words.append(word)
		elif word:
			pro_words.append(word)
		else:
			break
		index += 1
	return pro_words

def process_t(words):
	"""处理时间,合并语料库分开标注的时间词，类似： （/w  一九九七年/t  十二月/t  三十一日/t  ）/w   """
	pro_words = []
	index = 0
	temp = u''
	while True:
		word = words[index] if index < len(words) else u''
		if u'/t' in word:
			temp = temp.replace(u'/t', u'') + word
		elif temp:
			pro_words.append(temp)
			pro_words.append(word)
			temp = u''
		elif word:
			pro_words.append(word)
		else:
			break
		index += 1
	return pro_words


dir_ = "Data/NLP/Chinese/"
train_corpus_path = dir_ + "1980_01rmrb.txt"
process_corpus_path = dir_ + "result-rmrb.txt"
maps = {u't': u'T', u'nr': u'PER', u'ns': u'ORG', u'nt': u'LOC'}

lines = read_corpus_from_file(train_corpus_path)
new_lines = []
for line in lines:
	words = q_to_b(line.strip()).split(u'  ')
	pro_words = process_t(words)
	pro_words = process_nr(pro_words)
	pro_words = process_k(pro_words)
	new_lines.append('  '.join(pro_words[1:]))
write_corpus_to_file(data='\n'.join(new_lines).encode('utf-8'), file_path=process_corpus_path)

"""初始化 """
lines = read_corpus_from_file(process_corpus_path)
words_list = [line.strip().split('  ') for line in lines if line.strip()]  #  if line.strip() 保证不是空格行
# del lines

"""初始化字序列、词性序列、标记序列 """
words_seqs = [[word.split(u'/')[0] for word in words] for words in words_list]
# 迈向/v 充满/v -> 'v' 'v'
pos_seqs = [[word.split(u'/')[1] for word in words] for words in words_list]	# 二维列表
tag_seqs = [[maps.get(pos, 'O') for pos in pos_seq] for pos_seq in pos_seqs]		 # pos_to_tag(maps, pos) <-> maps.get(pos, 'O')
# 迈向/v 充满/v -> ['v', 'v'], ['v', 'v']
pos_seq_ = [[[pos_seqs[index][i] for _ in range(len(words_seqs[index][i]))]
				 for i in range(len(pos_seqs[index]))] for index in range(len(pos_seqs))]
# pos_seq = [[[] for pos in pos_seq] for pos_seq in pos_seqs]
# 标签采用“BIO”体系，即实体的第一个字为 B_*，其余字为 I_*，非实体字统一标记为 O
tag_seq_ = [[[tag_perform(tag_seqs[index][i], word_index) for word_index in range(len(words_seqs[index][i]))]
				 for i in range(len(tag_seqs[index]))] for index in range(len(tag_seqs))]
# 模型采用 tri-gram 形式，所以在字符列中，要在句子前后加上占位符。
# 迈向/v 充满/v -> ['un', v', 'v', 'v', 'v', 'un']
pos_seq = [[u'un'] + [pos_perform(maps, p) for pos in pos_seq for p in pos] + [u'un']
		   for pos_seq in pos_seq_]
# 三维->二维 [['O', 'O'], ['B_T', 'I_T','I_T']] -> ['O', 'O' 'B_T', 'I_T','I_T']
tag_seq = [[t for tag in tag_seq for t in tag] for tag_seq in tag_seq_]
# ['迈向' '充满'] -> ['<BOS>' '迈' '向' '充' '满' '<EOS>']
word_seqs_ = [[u'<BOS>'] + [w for word in word_seq for w in word] + [u'<EOS>'] for word_seq in words_seqs]

#### 训练模型 ###
algorithm = "lbfgs"
c1 = float("0.1")
c2 = float("0.1")
max_iterations = 100
model = sklearn_crfsuite.CRF(algorithm=algorithm, c1=c1, c2=c2, max_iterations=max_iterations, all_possible_transitions=True)

"""训练数据"""
word_grams = [segment_by_window(word_seq) for word_seq in word_seqs_]
features = extract_feature(word_grams)
x, y = features, tag_seq
x_train, y_train = x[500:], y[500:]
x_test, y_test = x[:500], y[:500]
model.fit(x_train, y_train)
labels = list(model.classes_)
labels_test = list(model.classes_) # 多了个非实体字统一标记 'O'
# labels.remove('O') 去除后f1_score低点，更准确
y_predict = model.predict(x_test)
scores = metrics.flat_f1_score(y_test, y_predict, average='weighted', labels=labels)
sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))


def predict(sentence):
	"""预测"""
	# 同训练一样，先做数据预处理
	u_sent = q_to_b(sentence)
	word_lists = [[u'<BOS>'] + [c for c in u_sent] + [u'<EOS>']]
	word_grams = [segment_by_window(word_list) for word_list in word_lists]
	features = extract_feature(word_grams)
	# 预测
	y_predict = model.predict(features)
	print(y_predict)
	entity = u''
	for index in range(len(y_predict[0])):
		if y_predict[0][index] != u'O':
			if index > 0 and y_predict[0][index][-1] != y_predict[0][index - 1][-1]:
				entity += u' '
			entity += u_sent[index]
		elif entity[-1] != u' ':
			entity += u' '
	return entity

predict(u'新华社北京十二月三十一日电(中央人民广播电台记者刘振英、新华社记者张宿堂)今天是一九九七年的最后一天。')
predict(u'一九四九年，国庆节，毛泽东同志在天安门城楼上宣布中国共产党从此站起来了！')


