import random
import jieba
import pandas as pd

# 指定文件目录
dir_path = "./Data/NLP_CH/NB_SVM/"
# 指定语料
stop_words = "".join([dir_path,'stopwords.txt'])
laogong = "".join([dir_path,'beilaogongda.csv'])  	#被老公打
laopo = "".join([dir_path,'beilaopoda.csv'])  		#被老婆打
erzi = "".join([dir_path,'beierzida.csv'])   		#被儿子打
nver = "".join([dir_path,'beinverda.csv'])    		#被女儿打
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
def preprocess_text(content_lines, sentences, category=None):
	for line in content_lines:
		try:
			segs = jieba.lcut(line)
			segs = [v for v in segs if not str(v).isdigit()]  # 去数字
			segs = list(filter(lambda x: x.strip(), segs))  # 去左右空格
			segs = list(filter(lambda x: len(x) > 1, segs))  # 长度为1的字符
			segs = list(filter(lambda x: x not in stopwords, segs))  # 去掉停用词
			if category is not None:
				sentences.append((" ".join(segs), category))  # 打标签
		except Exception:
			print(line)
			continue

# 调用函数、生成训练数据
sentences = []
preprocess_text(laogong, sentences, 0)
preprocess_text(laopo, sentences, 1)
preprocess_text(erzi, sentences, 2)
preprocess_text(nver, sentences, 3)

# 打散数据，生成更可靠的训练集
random.shuffle(sentences)

# 控制台输出前10条数据，观察一下
for sentence in sentences[:10]:
	print(sentence[0], sentence[1])
# 特征(序列)
all_texts = [sentence[0] for sentence in sentences]
# 标签
all_labels = [sentence[1] for sentence in sentences]

# 第四步，使用LSTM对数据进行分类：
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import LSTM, Embedding, GRU
from keras.models import Sequential
import numpy as np

# 预定义变量
MAX_SEQUENCE_LENGTH = 100  # 最大序列长度
EMBEDDING_DIM = 200  # embdding 维度
VALIDATION_SPLIT = 0.16  # 验证集比例
TEST_SPLIT = 0.2  # 测试集比例
# keras的sequence模块文本序列填充
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)
sequences = tokenizer.texts_to_sequences(all_texts)
# V=480
word_index = tokenizer.word_index
# 1674x100
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# 1674x4
labels = to_categorical(np.asarray(all_labels))


# 数据切分
p1 = int(len(data) * (1 - VALIDATION_SPLIT - TEST_SPLIT))
p2 = int(len(data) * (1 - TEST_SPLIT))
x_train = data[:p1]
y_train = labels[:p1]
x_val = data[p1:p2]
y_val = labels[p1:p2]
x_test = data[p2:]
y_test = labels[p2:]

# LSTM训练模型
model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
# 4,
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()
# 模型编译
model.compile(loss='categorical_crossentropy',
			  optimizer='rmsprop',
			  metrics=['acc'])
print(model.metrics_names)

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128)
model.save('lstm.h5')
# 模型评估 返回loss value 与 metrics中的值
print("损失", "准确率")
print(model.evaluate(x_test, y_test))


label_dict = {0:"被老公打", 1:"被老婆打", 2:"被儿子打", 3:"被女儿打"}
preds = np.argmax(model.predict(x_test), axis=1)
labels = np.argmax(y_test, axis=1)
pair = [(label_dict[preds[i]], label_dict[labels[i]]) for i, _ in enumerate(labels)]
import pandas as pd
pd.DataFrame(pair, columns=["预测", "真值"])
