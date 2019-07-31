import jieba

# 定义停用词、标点符号
punctuation = ["，","。", "：", "；", "？"]
# 定义语料
content = ["机器学习带动人工智能飞速的发展。",
		   "深度学习带动人工智能飞速的发展。",
		   "机器学习和深度学习带动人工智能飞速的发展。"
		  ]

# 分词
segs_1 = [jieba.lcut(con) for con in content]
print(segs_1)

# 去除标点符号，句子单词合并
tokenized = []
for sentence in segs_1:
	words = []
	for word in sentence:
		if word not in punctuation:
			words.append(word)
	tokenized.append(words)
print(tokenized)

# 求并集
bag_of_words = [x for item in segs_1 for x in item if x not in punctuation]
# 去重
bag_of_words = list(set(bag_of_words))
print(bag_of_words)

# 获得词袋向量
bag_of_word2vec = []
for sentence in tokenized:
	tokens = [1 if token in sentence else 0 for token in bag_of_words]
	bag_of_word2vec.append(tokens)



########## Gensim库词袋模型分词 ##########
from gensim import corpora
import gensim

# tokenized是去标点之后的
dictionary = corpora.Dictionary(tokenized)
# 保存词典
dictionary.save('deerwester.dict')
print(dictionary)

# 查看词典和下标 id 的映射
print(dictionary.token2id)

# doc2bow作用：计算每个不同单词的出现次数，将单词转换为其序号 个数 返回
corpus = [dictionary.doc2bow(sentence) for sentence in segs_1]
print(corpus )


########## 词向量 ##########
from gensim.models import Word2Vec
import jieba

# 定义停用词、标点符号
punctuation = [",", "。", ":", ";", ".", "'", '"', "’", "?", "/", "-", "+", "&", "(", ")"]
sentences = [
	"长江是中国第一大河，干流全长6397公里（以沱沱河为源），一般称6300公里。流域总面积一百八十余万平方公里，年平均入海水量约九千六百余亿立方米。以干流长度和入海水量论，长江均居世界第三位。",
	"黄河，中国古代也称河，发源于中华人民共和国青海省巴颜喀拉山脉，流经青海、四川、甘肃、宁夏、内蒙古、陕西、山西、河南、山东9个省区，最后于山东省东营垦利县注入渤海。干流河道全长5464千米，仅次于长江，为中国第二长河。黄河还是世界第五长河。",
	"黄河,是中华民族的母亲河。作为中华文明的发祥地,维系炎黄子孙的血脉.是中华民族民族精神与民族情感的象征。",
	"黄河被称为中华文明的母亲河。公元前2000多年华夏族在黄河领域的中原地区形成、繁衍。",
	"在兰州的“黄河第一桥”内蒙古托克托县河口镇以上的黄河河段为黄河上游。",
	"黄河上游根据河道特性的不同，又可分为河源段、峡谷段和冲积平原三部分。 ",
	"黄河,是中华民族的母亲河。"
]

sentences = [jieba.lcut(sen) for sen in sentences]
tokenized = []
for sentence in sentences:
	words = []
	for word in sentence:
		if word not in punctuation:
			words.append(word)
	tokenized.append(words)

# sg=1 是 skip-gram 算法，对低频词敏感；默认 sg=0 为 CBOW 算法。
# size 是输出词向量的维数，值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间。
# min_count 是对词进行过滤，频率小于 min-count 的单词则会被忽视，默认值为5。
model = Word2Vec(tokenized, sg=1, size=100,  window=5,  min_count=2,  negative=0, sample=0.001, hs=1, workers=4)

model.save('model')  # 保存模型
model = Word2Vec.load('model')  # 加载模型
print(model.wv.similarity('黄河', '母亲河'))
print(model.wv.similarity('黄河', '长江'))
# ，预测与黄河和母亲河最接近，而与长江不接近的词：
print(model.wv.most_similar(positive=['黄河', '母亲河'], negative=['长江']))



# 定义数据预处理类，作用是给每个文章添加对应的标签
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
doc_labels = ["长江", "黄河", "黄河", "黄河", "黄河", "黄河", "黄河"]
class LabeledLineSentence(object):
	def __init__(self, doc_list, labels_list):
		self.labels_list = labels_list
		self.doc_list = doc_list

	def __iter__(self):
		for idx, doc in enumerate(self.doc_list):
			yield LabeledSentence(words=doc, tags=[self.labels_list[idx]])


	# model = Doc2Vec(documents, dm=1, size=100, window=8, min_count=5, workers=4)
	# model.save('model')
	# model = Doc2Vec.load('model')

iter_data = LabeledLineSentence(tokenized, doc_labels)

# dm = 0 或者 dm=1 决定调用 DBOW 还是 DM
model = Doc2Vec(dm=1, size=100, window=8, min_count=5, workers=4)
model.build_vocab(iter_data)
model.train(iter_data, total_examples=model.corpus_count,epochs=1000,start_alpha=0.01,end_alpha =0.001)

#根据标签找最相似的，这里只有黄河和长江，所以结果为长江，并计算出了相似度
print(model.docvecs.most_similar('黄河'))
print(model.docvecs.similarity('黄河','长江'))