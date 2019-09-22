**“纸上得来终觉浅，绝知此事要躬行。”  —— 陆游**

**"Practice，practice，practice and summary makes perfect" —— dyliuti**

------



**简介：**

个人在学习深度学习时写过、测试过的代码。还有一些基于实践上的思考、总结。

<br>

**简化符号说明：**

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

4.NLP：练习NLP的基础，如词向量的构建方法，测试词类比，语句可能性等。也用RNN测试语言模型，命名实体训练等。

5.NLP2：主要是练习自然语言处理的下游应用。如用seq2seq、attention、transformer等进行翻译、对话问答。使用memory进行问答等。

6.NLP3_CH：测试中文自然语言处理需要用到的一些库。如jieba、hanlp等等。也练习一些简单的中文自然语言处理。