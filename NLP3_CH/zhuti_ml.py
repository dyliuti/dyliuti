import random
import jieba
import pandas as pd
# 指定文件目录
dir_ = "./Data/NLP_CH/NB_SVM/"
# 指定语料
stop_words = "".join([dir_,'stopwords.txt'])
laogong = "".join([dir_,'beilaogongda.csv'])  #被老公打
laopo = "".join([dir_,'beilaopoda.csv'])  #被老婆打
erzi = "".join([dir_,'beierzida.csv'])   #被儿子打
nver = "".join([dir_,'beinverda.csv'])    #被女儿打
# 加载停用词
stopwords = pd.read_csv(stop_words,index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values
# 加载语料
laogong_df = pd.read_csv(laogong, encoding='utf-8', sep=',')
laopo_df = pd.read_csv(laopo, encoding='utf-8', sep=',')
erzi_df = pd.read_csv(erzi, encoding='utf-8', sep=',')
nver_df = pd.read_csv(nver, encoding='utf-8', sep=',')
# 删除语料的nan行
laogong_df.dropna(inplace=True)
laopo_df.dropna(inplace=True)
erzi_df.dropna(inplace=True)
nver_df.dropna(inplace=True)
# 转换
laogong = laogong_df.segment.values.tolist()
laopo = laopo_df.segment.values.tolist()
erzi = erzi_df.segment.values.tolist()
nver = nver_df.segment.values.tolist()
# 定义分词和打标签函数preprocess_text
# 参数content_lines即为上面转换的list
# 参数sentences是定义的空list，用来储存打标签之后的数据
# 参数category 是类型标签
def preprocess_text(content_lines, sentences, category):
	for line in content_lines:
		try:
			segs = jieba.lcut(line)
			segs = [v for v in segs if not str(v).isdigit()]		# 去数字
			segs = list(filter(lambda x: x.strip(), segs))   		# 去左右空格
			segs = list(filter(lambda x: len(x)>1, segs)) 			# 长度为1的字符
			segs = list(filter(lambda x: x not in stopwords, segs)) # 去掉停用词
			# ('词 词 词'， '类别')形式
			sentences.append((" ".join(segs), category))			# 打标签
		except Exception:
			print("Exception: ", line)
			continue
# 调用函数、生成训练数据
sentences = []
preprocess_text(laogong, sentences, 'laogong')
preprocess_text(laopo, sentences, 'laopo')
preprocess_text(erzi, sentences, 'erzi')
preprocess_text(nver, sentences, 'nver')

# 打散数据，生成更可靠的训练集
random.shuffle(sentences)

# 控制台输出前10条数据，观察一下
for sentence in sentences[: 10]:
	print(sentence[0], sentence[1])
# 用sk-learn对数据切分，分成训练集和测试集
from sklearn.model_selection import train_test_split
x, y = zip(*sentences)	# *[3, 4] -> 3, 4     **{'a': 3, 'b', 4} -> 3, 4
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)

# 抽取特征，我们对文本抽取词袋模型特征
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(
    analyzer='word', # tokenise by character ngrams
	ngram_range=[1, 2],
    max_features=4000,  # keep the most common 1000 ngrams
)
vec.fit(x_train)
# 用朴素贝叶斯算法进行模型训练
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
x_inputs = vec.transform(x_train)
classifier.fit(x_inputs, y_train)
# 对结果进行评分
print(classifier.score(vec.transform(x_test), y_test))

# 用svm分类
from sklearn.svm import SVC
svm = SVC(kernel='linear')
svm.fit(x_inputs, y_train)
print(svm.score(vec.transform(x_test), y_test))





def preprocess_text(content_lines, sentences):
	for line in content_lines:
		try:
			segs = jieba.lcut(line)
			segs = [v for v in segs if not str(v).isdigit()]		# 去数字
			segs = list(filter(lambda x: x.strip(), segs))   		# 去左右空格
			segs = list(filter(lambda x: len(x)>1, segs)) 			# 长度为1的字符
			segs = list(filter(lambda x: x not in stopwords, segs)) # 去掉停用词
			# ('词 词 词'， '类别')形式
			sentences.append((" ".join(segs)))			# 打标签
		except Exception:
			print("Exception: ", line)
			continue


###########  聚类  ###########
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 调用函数、生成训练数据
sentences_ = []
preprocess_text(laogong, sentences_)
preprocess_text(laopo, sentences_)
preprocess_text(erzi, sentences_)
preprocess_text(nver, sentences_)

# 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
# 统计每个词语的tf-idf权值
transformer = TfidfTransformer()
# 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(sentences_))
# 获取词袋模型中的所有词语
word = vectorizer.get_feature_names()
# 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
weight = tfidf.toarray()
# 查看特征大小
print('Features length: ' + str(len(word)))

numClass = 4  # 聚类分几簇
clf = KMeans(n_clusters=numClass, max_iter=10000, init="k-means++", tol=1e-6)  # 这里也可以选择随机初始化init="random"
pca = PCA(n_components=10)  # 降维
TnewData = pca.fit_transform(weight)  # 载入N维
s = clf.fit(TnewData)

def plot_cluster(result,newData,numClass):
	plt.figure(2)
	Lab = [[] for i in range(numClass)]
	index = 0
	for labi in result:
		Lab[labi].append(index)
		index += 1
	color = ['oy', 'ob', 'og', 'cs', 'ms', 'bs', 'ks', 'ys', 'yv', 'mv', 'bv', 'kv', 'gv', 'y^', 'm^', 'b^', 'k^',
			 'g^'] * 3
	for i in range(numClass):
		x1 = []
		y1 = []
		for ind1 in newData[Lab[i]]:
			# print ind1
			try:
				y1.append(ind1[1])
				x1.append(ind1[0])
			except:
				pass
		plt.plot(x1, y1, color[i])

	#绘制初始中心点
	x1 = []
	y1 = []
	for ind1 in clf.cluster_centers_:
		try:
			y1.append(ind1[1])
			x1.append(ind1[0])
		except:
			pass
	plt.plot(x1, y1, "rv") #绘制中心
	plt.show()


pca = PCA(n_components=2)  # 输出两维
newData = pca.fit_transform(weight)  # 载入N维
result = list(clf.predict(TnewData))
plot_cluster(result, newData, numClass)


###### TSNE降维 #########
from sklearn.manifold import TSNE

ts = TSNE(2)
newData = ts.fit_transform(weight)
result = list(clf.predict(TnewData))
plot_cluster(result, newData, numClass)

from sklearn.manifold import TSNE

# 为了更好的表达和获取更具有代表性的信息，在展示（可视化）高维数据时，更为一般的处理，常常先用 PCA 进行降维，再使用 TSNE：
newData = PCA(n_components=4).fit_transform(weight)  # 载入N维
newData = TSNE(2).fit_transform(newData)
result = list(clf.predict(TnewData))
plot_cluster(result, newData, numClass)