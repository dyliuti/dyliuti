import pandas as pd

# 目的：对比Glove与Word2Vec词向量化效果
# 用训练好的Glove或Word2Vec得到每句中每个词的词向量
# 将句子中的各词向量求加权平均 表示一个句子
# 用常用的机器学习算法预测 句子的类别，

train = pd.read_csv('Data/NLP/r8-train-all-terms.txt', header=None, sep='\t')
test = pd.read_csv('Data/NLP/r8-test-all-terms.txt', header=None, sep='\t')
train.columns = ['label', 'content']
test.columns = ['label', 'content']

from NLP.Common.Model import GloveVectorizer
glove = GloveVectorizer()
X_train = glove.fit_transform(train.content)
Y_train = train.label
X_test = glove.fit_transform(test.content)
Y_test = test.label

# from NLP.Common.Model import Word2VecVectorizer
# word2vec = Word2VecVectorizer()
# X_train = word2vec.fit_transform(train.content)
# Y_train = train.label
# X_test = word2vec.fit_transform(test.content)
# Y_test = test.label

#测试 LogisticRegression RandomForest xgboost
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = [('LogisticRegression', LogisticRegressionCV(Cs=10, cv=5)),  # Cs正则约束，越小越强  cv:交叉验证
          ('RandomForest', RandomForestClassifier(n_estimators=50)),    # , criterion='gini'
          ('XGBoost', XGBClassifier(max_depth=3, n_estimators=50, silent=True, objective='multi:softmax'))]
for name, model in models:
	model.fit(X_train, Y_train)
	print(name, '训练集正确率：', model.score(X_train, Y_train))
	print(name, '测试集正确率：', model.score(X_test, Y_test))