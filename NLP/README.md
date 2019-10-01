**“纸上得来终觉浅，绝知此事要躬行。”  —— 陆游**

**"Practice，practice，practice and summary makes perfect" —— dyliuti**

------



**自然语言处理：**



<br>

**总结：**



<br>

**文件说明：**

1.bigram: 利用贝叶斯估计求得bigram矩阵，用bigram矩阵估计一个句子的可能性（对数似然）。

1.bigram_nn:利用神经网络训练bigram，二分类问题。以及二分类的迭代效率优化。

2.word2vec和2.glove分别用负采样和共现矩阵方式实现词向量。然后用词向量进行词类比。

2.model_test分别用word2vec或glove的预训练词向量，将句子加权合并到一个词，进行语句分类。



<br>

**数据集下载：**

[Markov数据集下载，解压后将Markov文件夹放在Data文件夹下](https://drive.google.com/file/d/1G3rmYtY7Io754vVogcEdtskvTqYfsiuF/view?usp=sharing)

