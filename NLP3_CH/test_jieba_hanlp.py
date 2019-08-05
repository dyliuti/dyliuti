import jieba

# https://github.com/fxsjy/jieba
# https://github.com/hankcs/pyhanlp

##################### 分词测试 #####################
content = "现如今，机器学习和深度学习带动人工智能飞速的发展，并在图片处理、语音识别领域取得巨大成功。"
# 精确分词：精确模式试图将句子最精确地切开，精确分词也是默认分词。
segs_1_gen = jieba.cut(content, cut_all=False)
segs_1_list = jieba.lcut(content, cut_all=False)
print("/".join(segs_1_gen))
print(segs_1_list)

# 全模式分词：把句子中所有的可能是词语的都扫描出来，速度非常快，但不能解决歧义。
# 是否全模式用cut_all来控制
segs_2_gen = jieba.cut(content, cut_all=True)
segs_2_list = jieba.lcut(content, cut_all=True)
print("/".join(segs_2_gen))
print(segs_2_list)

# 搜索引擎模式：在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
segs_3 = jieba.cut_for_search(content)
segs_3_list = jieba.lcut_for_search(content)
print("/".join(segs_3))
print(segs_3_list)

# 并行分词：并行分词原理为文本按行分隔后，分配到多个 Python 进程并行分词，最后归并结果。
jieba.enable_parallel(4)
jieba.disable_parallel()

######### 获取词性 #########
import jieba.posseg as psg
print([(x.word, x.flag) for x in psg.lcut(content)])  # 无[]就是生成器，加[]就是把生成器转换为列表

sentence = u'''上线三年就成功上市,拼多多上演了互联网企业的上市奇迹,却也放大平台上存在的诸多问题，拼多多在美国上市。'''
kw = jieba.analyse.extract_tags(sentence, topK=10, withWeight=True, allowPOS=('n', 'ns'))
for item in kw:
	print(item[0], item[1])

kw = jieba.analyse.textrank(sentence, topK=20, withWeight=True, allowPOS=('ns', 'n'))
for item in kw:
	print(item[0], item[1])

# 用hanlp提取词
from pyhanlp import *

sentence = u'''上线三年就成功上市,拼多多上演了互联网企业的上市奇迹,却也放大平台上存在的诸多问题，拼多多在美国上市。'''
analyzer = PerceptronLexicalAnalyzer()
segs = analyzer.analyze(sentence)
arr = str(segs).split(" ")

def get_result(arr):
	re_list = []
	ner = ['n', 'ns']
	for x in arr:
		temp = x.split("/")
		if temp[1] in ner:
			re_list.append(temp[0])
	return re_list

result = get_result(arr)
print(result)


######### 统计词频 #########
from collections import Counter
top5 = Counter(segs_3_list).most_common(5)
print(top5)

######### 自定义添加词和字典 #########
txt = "铁甲网是中国最大的工程机械交易平台。"
print(jieba.lcut(txt))
jieba.add_word("铁甲网")		# 可识别“铁甲网了”
print(jieba.lcut(txt))

# 加载文件，添加词和字典，提高效率
jieba.load_userdict(['NLP3_CH/user_dict.txt'])  # 加[]是利用源码，不想改动源码处理str的部分
print(jieba.lcut(txt))


##################### hanlp分词测试 #####################
from pyhanlp import *

content = "现如今，机器学习和深度学习带动人工智能飞速的发展，并在图片处理、语音识别领域取得巨大成功。"
print(HanLP.segment(content))

######### 自定义添加词和字典 #########
txt = "铁甲网是中国最大的工程机械交易平台。"
print(HanLP.segment(txt))
CustomDictionary.add("铁甲网")
CustomDictionary.insert("工程机械", "nz 1024")
CustomDictionary.add("交易平台", "nz 1024 n 1")
print(HanLP.segment(txt))


##################### 关键词提取 #####################

######### TF-IDF #########
# TF-IDF:TF-IDF 倾向于过滤掉常见的词语，保留重要的词语。
# 例如，某一特定文件内的高频率词语，以及该词语在整个文件集合中的低文件频率，可以产生出高权重的 TF-IDF。
import jieba.analyse
sentence = "人工智能（Artificial Intelligence），英文缩写为AI。它是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。人工智能是一门极富挑战性的科学，从事这项工作的人必须懂得计算机知识，心理学和哲学。人工智能是包括十分广泛的科学，它由不同的领域组成，如机器学习，计算机视觉等等，总的说来，人工智能研究的一个主要目标是使机器能够胜任一些通常需要人类智能才能完成的复杂工作。但不同的时代、不同的人对这种“复杂工作”的理解是不同的。2017年12月，人工智能入选“2017年度中国媒体十大流行语”。"
keywords = jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())
keywords_weigths = jieba.analyse.extract_tags(sentence, topK=20, withWeight=True, allowPOS=())
keywords = "  ".join(keywords)
print(keywords)

# 只获取名词和动词
keywords =(jieba.analyse.extract_tags(sentence , topK=10, withWeight=True, allowPOS=(['n','v'])))
print(keywords)

######### TextRank #########
# TextRank：由 PageRank 改进而来，核心思想将文本中的词看作图中的节点，通过边相互连接，
# 不同的节点会有不同的权重，权重高的节点可以作为关键词。
jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
result = "  ".join(jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v')))
print(result)

######### LDA #########

#引入库文件
import jieba.analyse as analyse
import jieba
import pandas as pd
from gensim import corpora, models, similarities
import gensim
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# 设置文件路径
dir_ = "./Data/NLP/Chinese/lda/"
file_desc = "".join([dir_,'car.csv'])
stop_words = "".join([dir_,'stopwords.txt'])
# 定义停用词
stopwords = pd.read_csv(stop_words,index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords = stopwords['stopword'].values
# 加载语料
df = pd.read_csv(file_desc, encoding='gbk')
# 删除nan行
df.dropna(inplace=True)
lines = df.content.values.tolist()
# 开始分词
sentences = []
for line in lines:
	try:
		# 分隔line，然后对分词后的词
		segs_ = jieba.lcut(line)
		segs = [v for v in segs_ if not str(v).isdigit()]		# 去数字
		segs = list(filter(lambda x: x.strip(), segs))   		# 去左右空格
		segs = list(filter(lambda x: x not in stopwords, segs)) # 去掉停用词
		sentences.append(segs)
	except Exception:
		print(line)
		continue
# 构建词袋模型
dictionary = corpora.Dictionary(sentences)
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]
# lda模型，num_topics是主题的个数，这里定义了5个
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)
# 我们查一下第1号分类，其中最常出现的5个词是：
print(lda.print_topic(1, topn=5))
# 我们打印所有5个主题，每个主题显示8个词
for topic in lda.print_topics(num_topics=10, num_words=8):
	print(topic[1])


# 显示中文matplotlib
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 在可视化部分，我们首先画出了九个主题的7个词的概率分布图
num_show_term = 8 # 每个主题下显示几个词
num_topics  = 10
for i, k in enumerate(range(num_topics)):
	ax = plt.subplot(2, 5, i+1)
	# 列表，列表里是字典项  （index，频率）
	index_freq = lda.get_topic_terms(topicid=k)
	#
	index_freq_num = np.array(index_freq[:num_show_term])
	ax.plot(range(num_show_term), index_freq_num[:, 1], 'b*')		# 展示评率
	indexs = index_freq_num[:, 0].astype(np.int)						# 词索引
	words = [dictionary.id2token[index] for index in indexs]
	ax.set_ylabel(u"概率")
	for j in range(num_show_term):
		# 纵坐标频率，s是words
		ax.text(x=j, y=index_freq_num[j, 1], s=words[j], bbox=dict(facecolor='green',alpha=0.1))
plt.suptitle(u'9个主题及其7个主要词的概率', fontsize=18)
plt.show()
