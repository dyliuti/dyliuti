**“纸上得来终觉浅，绝知此事要躬行。”  —— 陆游**

**"Practice，practice，practice and summary makes perfect" —— dyliuti**



**优化算法：**

梯度下降是目标函数在自变量当前位置下降最快的方向。对于多维梯度，容易误入“歧途”，如往最低点的梯度下降较别的自变量的梯度下降平缓得多。梯度下降法会变得非常缓慢，一是学习率要低点，二是容易陷入局部低点。

momentum是为了解决梯度下降上述的问题的。momentum核心思想就是更新下次梯度时，叠加上些上次梯度值，这样可加速走出“歧途”或叫局部低点吧。画下等高图就容易理解了。指数加权移动平均注意下。

RMSProp说明前，先看下两个公式：

Adagrad的cache：cache = cache + gradient^2

RMSProp的cache：cache = decay * cache + (1-decay) * gradient^2

两者自变量更新公式是一样的：Xt = Xt-1 - 学习率 * 损失函数对X的梯度 / 根号(cache + 很小的值)

Adagrad与RMSProp的思想：因为各方向的梯度大小不同，梯度很小的方向学习率大点有利于加速学习，用cache自适应学习率。cache都使用了二阶梯度，RMSProp另外多了指数衰减参数decay，有利于防止cache增长得太快太大，自变量学习不动了，即更新太慢或更新不动了。

Adam的思想：不仅对二阶梯度进行指数移动衰减， 对一阶梯度也进行了指数移动衰减。且对指数衰减后的值进行变换：

cache = cache / (1 - decay^t)  t表示迭代次数。该操作可校正初始没步可移动平均的尴尬。

从Adam和RMSProp的整体公式上来看，Adam对损失函数对自变量的梯度，即一阶导，也采用了cache或叫momentum，所以有人称Adam是“RMSProp with momentum”。



_np: numpy	tf: tensorflow	ke:  keras	mx: mxnet



**后缀说明：**

_np: numpy	tf: tensorflow	ke:  keras	mx: mxnet



**文件说明：**

1.开头的是一些优化算法。

[Minist文件夹用到的数据集下载](https://drive.google.com/file/d/1dQk9YIUDQZbubn4a3cay6hctYoBJjpiu/view?usp=sharing)

**文件结构：**

按照dive into deep learning的章节，代码基本都是关于gulon的。