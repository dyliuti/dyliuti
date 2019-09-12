**“纸上得来终觉浅，绝知此事要躬行。”  —— 陆游**

**"Practice，practice，practice and summary makes perfect" —— dyliuti**

------

<br>

优化算法：**

梯度下降是目标函数在自变量当前位置下降最快的方向。对于多维梯度，容易误入“歧途”，如往最低点的梯度下降较别的自变量的梯度下降平缓得多。梯度下降法会变得非常缓慢，一是学习率要低点，二是容易陷入局部低点。

momentum是为了解决梯度下降上述的问题的。momentum核心思想就是更新下次梯度时，叠加上些上次梯度值，这样可加速走出“歧途”或叫局部低点吧。画下等高图就容易理解了。指数加权移动平均注意下。

RMSProp说明前，先看下两个公式：

Adagrad的cache：cache = cache + gradient^2

RMSProp的cache：cache = decay * cache + (1-decay) * gradient^2

两者自变量更新公式是一样的：Xt = Xt-1 - 学习率 * 损失函数对X的梯度 / 根号(cache + 很小的值)

Adagrad与RMSProp的思想：因为各方向的梯度大小不同，梯度很小的方向学习率大点有利于加速学习，用cache自适应学习率。cache都使用了二阶梯度，RMSProp另外多了指数衰减参数decay，有利于防止cache增长得太快太大，自变量学习不动了，即更新太慢或更新不动了。

Adam的思想：不仅对二阶梯度进行指数移动衰减， 对一阶梯度也进行了指数移动衰减。且对指数衰减后的值进行变换：

cache = cache / (1 - decay^t)  t表示迭代次数。该操作可校正初始每步可移动平均的尴尬。

从Adam和RMSProp的整体公式上来看，Adam的损失函数对自变量的梯度，转换为了类momentum，所以有人称Adam是“RMSProp with momentum”。

<br>

**文件说明：**

1.开头的是一些优化算法。

2.开头的是用感知机进行训练预测手写数字。

3.开头的是用卷积神经网络进行手写数字预测。两种结构，一种是kaggle top8%版的，[cnn_ke.py引用Yassine Ghouzam, PhD的分享](https://www.kaggle.com/woshiliziming/minist)；cnn_mx.py我用mxnet重写了上述卷积结构。还有一种就是用典型的Lenet预测手写数字。数据读取方式也试了下，如 mxnet直接读图和先制作成RecordIO文件，再读取。tf，mx数据增强。

4.开头的是用Bi-LSTM进行预测。分别将横纵与纵轴当做序列，经过Bi-LSTM，输出每个序列的隐藏状态。通过GlobalAveragePooling1D进行序列中的特征进行压缩提取，然后将横轴序列特征组合下经过Dense进行10分类。

<br>

**数据集下载：**

[Minist文件夹用到的数据集下载](https://drive.google.com/file/d/1dQk9YIUDQZbubn4a3cay6hctYoBJjpiu/view?usp=sharing)

