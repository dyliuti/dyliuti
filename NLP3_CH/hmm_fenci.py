import pickle
import json


STATES = {'B', 'M', 'E', 'S'}
EPS = 0.0001
# 定义停顿标点
seg_stop_words = {" ","，","。","“","”",'“', "？", "！", "：", "《", "》", "、", "；", "·", "‘ ", "’", "──", ",", ".", "?", "!", "`", "~", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "+", "=", "[", "]", "{", "}", '"', "'", "<", ">", "\\", "|" "\r", "\n","\t"}

class HMM_Model:
	def __init__(self):
		self.trans_mat = {}
		self.emit_mat = {}
		self.init_vec = {}
		self.state_count = {}
		self.states = {}
		self.inited = False

	# 初始化
	# StatusSet：状态值集合，状态值集合为 (B, M, E, S)，其中 B 为词的首个字，M 为词中间的字，
	# E 为词语中最后一个字，S 为单个字，B、M、E、S 每个状态代表的是该字在词语中的位置。
	# emit_mat：观测矩阵，emit_mat[state][char] 表示训练集中单字 char 被标注为 state 的次数。
	# init_vec：初始状态分布向量，init_vec[state] 表示状态 state 在训练集中出现的次数。初始状态只可能是B,S
	# state_count：状态统计向量，state_count[state]表示状态 state 出现的次数。
	def setup(self):
		for state in self.states:
			# build trans_mat
			self.trans_mat[state] = {}
			for target in self.states:
				self.trans_mat[state][target] = 0.0	# M x M  state->target 频数
			self.emit_mat[state] = {}				# M		 observer: 频数
			self.init_vec[state] = 0				# M
			self.state_count[state] = 0				# M
		self.inited = True

	# 模型保存
	def save(self, filename, code):
		fw = open(filename, 'w', encoding='utf-8')
		data = {
			"trans_mat": self.trans_mat,
			"emit_mat": self.emit_mat,
			"init_vec": self.init_vec,
			"state_count": self.state_count
		}
		if code == "json":
			txt = json.dumps(data)
			txt = txt.encode('utf-8').decode('unicode-escape')
			fw.write(txt)
		elif code == "pickle":
			pickle.dump(data, fw)
		fw.close()
	# 模型加载
	def load(self, filename, code):
		fr = open(filename, 'r', encoding='utf-8')
		if code == "json":
			txt = fr.read()
			model = json.loads(txt)
		elif code == "pickle":
			model = pickle.load(fr)
		self.trans_mat = model["trans_mat"]
		self.emit_mat = model["emit_mat"]
		self.init_vec = model["init_vec"]
		self.state_count = model["state_count"]
		self.inited = True
		fr.close()

	# 模型训练
	# 使用的标注数据集， 因此可以使用更简单的监督学习算法，训练函数输入观测序列和状态序列进行训练，
	# 依次更新各矩阵数据。类中维护的模型参数均为频数而非频率， 这样的设计使得模型可以进行在线训练，
	# 使得模型随时都可以接受新的训练数据继续训练，不会丢失前次训练的结果。
	def do_train(self, observes, states):
		if not self.inited:
			self.setup()

		for i in range(len(states)):
			if i == 0:
				# 初始状态，与状态计数
				self.init_vec[states[0]] += 1
				self.state_count[states[0]] += 1
			else:
				# 转移矩阵, 与状态计数
				self.trans_mat[states[i - 1]][states[i]] += 1
				self.state_count[states[i]] += 1
				# self.emit_mat[states[i]][observes[i]] = self.emit_mat[states[i]].get(observes[i], 0) + 1
				if observes[i] not in self.emit_mat[states[i]]:
					self.emit_mat[states[i]][observes[i]] = 1
				else:
					self.emit_mat[states[i]][observes[i]] += 1
	# HMM计算
	# 预测前，需将数据结构的频数转换为频率
	def get_prob(self):
		init_vec = {}
		trans_mat = {}
		emit_mat = {}
		default = max(self.state_count.values())

		for state in self.init_vec:
			if self.state_count[state] != 0:	# 防止被除数是0
				# 每个词语的开头状态的频率  只有B、S非零
				init_vec[state] = float(self.init_vec[state]) / self.state_count[state]
			else:
				init_vec[state] = float(self.init_vec[state]) / default

		for state_prev in self.trans_mat:
			trans_mat[state_prev] = {}
			for state_next in self.trans_mat[state_prev]:
				if self.state_count[state_prev] != 0:
					# count(state_prev -> state_next) / count(state_prev) 条件概率
					trans_mat[state_prev][state_next] = float(self.trans_mat[state_prev][state_next]) / self.state_count[state_prev]
				else:
					trans_mat[state_prev][state_next] = float(self.trans_mat[state_prev][state_next]) / default

		for state in self.emit_mat:
			emit_mat[state] = {}
			for target in self.emit_mat[state]:
				if self.state_count[state] != 0:
					# count(state -> target) / count(state)	条件概率
					emit_mat[state][target] = float(self.emit_mat[state][target]) / self.state_count[state]
				else:
					emit_mat[state][target] = float(self.emit_mat[state][target]) / default
		return init_vec, trans_mat, emit_mat

	# 模型预测
	# 采用 Viterbi 算法求得最优路径
	def do_predict(self, sequence):
		tab = [{}]
		path = {}
		init_vec, trans_mat, emit_mat = self.get_prob()

		# 初始化
		for state in self.states:
			# P(Y, Z) = P(Z) * P(Y | Z)
			tab[0][state] = init_vec[state] * emit_mat[state].get(sequence[0], EPS)
			path[state] = [state]

		# 创建动态搜索表
		for t in range(1, len(sequence)):
			tab.append({})
			new_path = {}
			for state in self.states:
				items = []
				for state_prev in self.states:
					if tab[t - 1][state_prev] == 0:
						continue
					# P(Y0..Yt-1, Zt-1) * P(Zt | Zt-1) * P(Yt | Zt) -> P(Y0..Yt, Zt, Zt-1)   多个Zt-1 -> 每个Zt
					prob = tab[t - 1][state_prev] * trans_mat[state_prev].get(state, EPS) * emit_mat[state].get(sequence[t], EPS)
					items.append((prob, state_prev))	# 多个Zt-1 可到 单个Zt的所有概率大小
				print(items)
				best = max(items)	# 多个Zt-1 可到 单个Zt，取概率最大的，即最可能的
				tab[t][state] = best[0]	# t-1时刻（序列） 到 P(Y0..Yt, Zt) 时最大的概率
				new_path[state] = path[best[1]] + [state]	# 状态前插
			path = new_path
		print(tab)
		# 搜索最优路径
		prob, state = max([(tab[len(sequence) - 1][state], state) for state in self.states])
		print("prob", prob)
		print("state: ", state)
		print(path)
		print(path[state])
		return path[state]

# 对输入的训练语料中的每个词进行标注，因为训练数据是空格隔开的，可以进行转态标注
# src代表一个词语，词语进行转换得到如['B', 'M', 'M', 'E']
def get_tags(src):
	tags = []
	if len(src) == 1:
		tags = ['S']
	elif len(src) == 2:
		tags = ['B', 'E']
	else:
		m_num = len(src) - 2
		tags.append('B')
		tags.extend(['M'] * m_num)
		tags.append('E')
	return tags

# 根据预测得到的标注序列将输入的句子分割为词语列表，也就是预测得到的状态序列，解析成一个 list 列表进行返回
def cut_sent(src, tags):
	word_list = []
	start = -1
	started = False

	if len(tags) != len(src):
		return None

	if tags[-1] not in {'S', 'E'}:
		if tags[-2] in {'S', 'E'}:
			tags[-1] = 'S'
		else:
			tags[-1] = 'E'

	for i in range(len(tags)):
		if tags[i] == 'S':
			if started:
				started = False
				word_list.append(src[start:i])
			word_list.append(src[i])
		elif tags[i] == 'B':
			if started:
				word_list.append(src[start:i])
			start = i
			started = True
		elif tags[i] == 'E':
			started = False
			word = src[start:i+1]
			word_list.append(word)
		elif tags[i] == 'M':
			continue
	return word_list

class HMM_FenCi(HMM_Model):
	def __init__(self, *args, **kwargs):
		super(HMM_FenCi, self).__init__(*args, **kwargs)
		self.states = STATES
		self.data = None

	# 加载训练数据
	def read_txt(self, filename):
		self.data = open(filename, 'r', encoding="utf-8")

	# 模型训练函数
	# 根据单词生成观测序列和状态序列，并通过父类的 do_train() 方法进行训练
	def train(self):
		if not self.inited:
			self.setup()

		for line in self.data:
			line = line.strip()
			if not line:
				continue

			# 观测序列
			observes = []
			for i in range(len(line)):
				if line[i] == " ":
					continue
				observes.append(line[i])

			# 状态序列
			words = line.split(" ")

			states = []
			for word in words:
				# 去停用词
				if word in seg_stop_words:
					continue
				# 获得每行已知的状态
				states.extend(get_tags(word))
			# 开始训练每行
			if (len(observes) >= len(states)):
				self.do_train(observes, states)
			else:
				pass

	# 模型分词预测
	# 模型训练好之后，通过该方法进行分词测试
	def lcut(self, sentence):
		try:
			tags = self.do_predict(sentence)
			return cut_sent(sentence, tags)
		except:
			return sentence


FenCi = HMM_FenCi()
FenCi.read_txt("Data/NLP/Chinese/syj_trainCorpus_utf8.txt")
FenCi.train()

FenCi.lcut("中国的人工智能发展进入高潮阶段。")
FenCi.lcut("中文自然语言处理是人工智能技术的一个重要分支。")