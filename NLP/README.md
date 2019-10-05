**“纸上得来终觉浅，绝知此事要躬行。”  —— 陆游**

**"Practice，practice，practice and summary makes perfect" —— dyliuti**

------



**文件说明：**

1.bigram: 利用贝叶斯估计求得bigram矩阵，用bigram矩阵估计一个句子的可能性（对数似然）。

1.bigram_nn:利用神经网络训练bigram，二分类问题。以及二分类的迭代效率优化。

2.word2vec和2.glove分别用负采样和共现矩阵方式实现词向量。然后用词向量进行词类比。

2.model_test分别用word2vec或glove的预训练词向量，将句子加权合并到一个词，进行语句分类。

3.开头的是用rnn来做异或校验和训练语言模型。

4.开头的是用解析树表示句子，用递归神经网络做语义情感分析。



