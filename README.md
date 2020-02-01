**“纸上得来终觉浅，绝知此事要躬行。”  —— 陆游**

**"Practice，practice，practice and summary make perfect" —— dyliuti**

------



**简介：**

个人在学习深度学习时写过、测试过的代码。还有一些基于实践上的思考、总结。

**感谢：**

这一年感谢真正富有学识，又乐于分享的大神们。你们总是在想能给予别人什么，怎么成就别人，让人敬佩。也感谢本素不相识，却给过我鼓励的人们，谢谢了！

<br>

**视频链接：**

[个人Bilibili主页，点我](https://space.bilibili.com/33760281)

<br>

**常用简化符号说明：**

T:   序列长度。

N:  随机梯度，N=1；批量随机梯度，N=batch_size; 全量，N=examples_num

H:  hidden_size，LSTM或GRU隐藏层维度。

D: 对应Embedding的维度

V: 对应语料库中不重复词的数量，相当于预测时dense的输出维度

<br>

**文件结构：**

Data：存放模型、模型数据、素材等的文件夹。

1.Minist：多种框架、多种神经网络（ANN、CNN、RNN），以及一些机器学习方法实现Minist分类。

2.CNN：主要是使用CNN的成熟网络结构，得到一些应用。如用迁移学习的方法、进行style_transform的图片生成。如用ssd检测图片目标、使用ssd检测视频中的目标、并标注于视频中等。

3.Markov：练习Markov与隐Markov三类问题的解法。

4.NLP：练习NLP的基础，如词向量的构建方法，测试词类比，语句可能性等。也用RNN测试语言模型等。

5.NLP2：主要是练习自然语言处理的下游应用。如用seq2seq、attention、transformer等进行翻译、对话问答。用memory问答推理。命名实体、情感分析、文档分类、句法依存等应用。

6.NLP3_CH：测试中文自然语言处理需要用到的一些库。如jieba、hanlp等等。也练习一些简单的中文自然语言处理。

7.BERT：使用BERT进行命名实体检测。

8.BERT-Classifier：使用BERT进行分类。

<br>

**运行环境：**

Pycharm中以dyliuti为根目录。文件找不到现象，是因为不是以dyliuti为根目录，需要手动改下路径。

PyCharm：Windows 2018community

tensorflow：1.14.0-gpu		CUDA 10.0.0		cuDNN 7.4.1

keras：2.2.4

mxnet：1.5.0
