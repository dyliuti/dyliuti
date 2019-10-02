from Data.DataExtract import load_bAbI_challange_data
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Lambda, Reshape, add, dot, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, RMSprop
import keras.backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# loss: 0.1388 - acc: 0.9643 - val_loss: 0.3998 - val_acc: 0.9020

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

def should_flatten(data_item):
	return not isinstance(data_item, (str, bytes))

# <data 'tuple'>: ([['0', 'Mary', 'moved', 'to', 'the', 'bathroom', '.'], ['1', 'John', 'went', 'to', 'the', 'hallway', '.']],
# ['Where', 'is', 'Mary', '?'], 'bathroom')
# <vocab 'list'>: ['<PAD>', '.', '0', '1', '10', '12', '13', '3', '4', '6', '7', '9', '?', 'Daniel', 'John', 'Mary', 'Sandra', 'Where', 'back',
# 'bathroom', 'bedroom', 'garden', 'hallway', 'is', 'journeyed', 'kitchen', 'moved', 'office', 'the', 'to', 'travelled', 'went']
def flatten(data):
	for data_item in data:
		# print(data_item)
		if should_flatten(data_item):
			# 若果是story或query，就返回story或query每句中的单词
			yield from flatten(data_item)
		else:
			# 如果是answer（str），就返回answer
			yield data_item

# 对sotry中的每句索引化后进行pad，对query索引化后进行pad， 结果转换为索引
def vectorize_data(data, word2idx, max_story_sents_len, max_query_len):
	inputs, queries, answers = [], [], []
	for story, query, answer in data:
		inputs.append( [[word2idx[word] for word in sentence] for sentence in story])
		queries.append([word2idx[word] for word in query])
		answers.append([word2idx[answer]])
	# 前面补0
	return (
		[pad_sequences(sentence, maxlen=max_story_sents_len) for sentence in inputs],
		pad_sequences(queries, maxlen=max_query_len),
		np.array(answers)
	)

# 1000x10x8 将句子的个数对齐
def stack_inputs(inputs, max_story_len, max_story_sents_len):
	for i, story in enumerate(inputs):
		# 没满足 max_story_len x max_story_sents_len 的，进行填充
		inputs[i] = np.concatenate( [story, np.zeros((max_story_len - story.shape[0], max_story_sents_len), 'int')] )
	return np.stack(inputs)

# 数据集改为two_support了
train_data = load_bAbI_challange_data(challenge_type='two_supporting_facts_10k', data_type='train')
test_data = load_bAbI_challange_data(challenge_type='two_supporting_facts_10k', data_type='test')

data = train_data + test_data

max_story_sents_len = max((len(sentence) for stories, _, _ in data for sentence in stories))
max_story_len = max((len(stories) for stories, _, _ in data))
max_query_len = max(len(question) for _, question, _ in data)

# <data 'tuple'>: ([['0', 'Mary', 'moved', 'to', 'the', 'bathroom', '.'], ['1', 'John', 'went', 'to', 'the', 'hallway', '.']], ['Where', 'is', 'Mary', '?'], 'bathroom')
# <vocab 'list'>: ['<PAD>', '.', '0', '1', '10', '12', '13', '3', '4', '6', '7', '9', '?', 'Daniel', 'John', 'Mary', 'Sandra', 'Where', 'back', 'bathroom', 'bedroom', 'garden', 'hallway', 'is', 'journeyed', 'kitchen', 'moved', 'office', 'the', 'to', 'travelled', 'went']
flatten_data = flatten(data)	# 生成器
vocab = sorted(set(flatten_data))
vocab.insert(0, '<PAD>')		# 前面
vocab_size = len(vocab)

word2idx = {c: i for i, c in enumerate(vocab)}

# 将句子长度对齐，对sotry中的每句索引化后进行pad，对query索引化后进行pad， 结果转换为索引
# N x max_story_len x T		N x max_query_Len
inputs_train_, queries_train, answers_train = vectorize_data(
    train_data,
    word2idx,
    max_story_sents_len,
    max_query_len
)
inputs_test_, queries_test, answers_test = vectorize_data(
    test_data,
    word2idx,
    max_story_sents_len,
    max_query_len
)

# 将story句子数一致，从list转换为 3-D numpy arrays
inputs_train = stack_inputs(inputs_train_, max_story_len, max_story_sents_len)
inputs_test = stack_inputs(inputs_test_, max_story_len, max_story_sents_len)




############### 建立模型 ###############
embedding_dim = 30

# 将story多个句子映射到一个句子，将question单个句子映射到一个词向量
def embedded_and_sum(input, axis):
	# embedded_story.shape        = (N, num sentences, embedding_dim)
	# embedded_question.shape     = (N, 1, embedding_dim)
	# story：? T 8 30		quesetion: ? 4 30
	embedded_input = Embedding(vocab_size, embedding_dim)(input)
	# story：? T 30			quesetion: ? 30
	embedded_input = Lambda(lambda x: K.sum(x, axis=axis))(embedded_input)
	return embedded_input

# 将story转换为一系列的embedding vector，每个story就像“bag of words”
input_story = Input(shape=(max_story_len, max_story_sents_len))
# 将question转换为embedding，也是bag of words
input_question = Input(shape=(max_query_len, ))
# story：? T 30			quesetion: ? 30
# 创建embedding代表故事中的每一行
embedded_story_ = embedded_and_sum(input_story, 2)
embedded_question_ = embedded_and_sum(input_question, 1)


def hop(query, story):
	# ? 30  ->  ? 1 30 为了和story点积
	embedded_question = Reshape(target_shape=(1, embedding_dim))(query)
	# 计算每个story中句子的权重
	# ? T 30   ? 1 30   ->  ? T, 1
	x__ = dot(inputs=[story, embedded_question], axes=2)
	# ? T
	x_ = Reshape(target_shape=(max_story_len, ))(x__)		# x 表示故事中每个映射到词向量的每个句子的权重
	x = Activation('softmax')(x_)
	# 再次unflatten，因为之后还需要点积
	# ? T 1
	story_weights = Reshape(target_shape=(max_story_len, 1))(x)

	# 区别：使用新的Embedding,产生新的story与ans -> ? T 30
	# embedding代表第二个故事中的每一行不然。作用：hop产生的weights与返回的ans不一样
	embedded_story2_ = embedded_and_sum(input_story, 2)
	# ? T 1   ? T M -> ? 1 M								# 顺序明朗，把story_weights与embedded_story位置换了，结果也是对的
	ans = dot(inputs=[story_weights, embedded_story2_], axes=1)
	ans = Reshape(target_shape=(embedding_dim, ))(ans)
	# ? M -> ? vocab_size									# 再加一层Dense，又是逻辑回归，低维映射到高维，做分类
	# ans = Dense(vocab_size, activation='softmax')(x_ans_)
	# 区别： Dense，activation类别，之前这里没有
	# ？M -> ? M 虽然shape每变，但在这里加Dense有助于做个转换，更利于下一步hop   去掉的话准确率会将降低
	# 编码解码不是特别合适，可以理解为特征转换
	ans = Dense(embedding_dim, activation='elu')(ans)
	return ans, embedded_story2_, story_weights

ans1, embedded_story, story_weights1 = hop(embedded_question_, embedded_story_)
# 第一个hop产生的ans传给第二个hop，可以确保weights不同
ans2, _,              story_weights2 = hop(ans1,               embedded_story)
# 不能返回ans1_index 模型的输出值必须是一个 Keras `Layer`
# ans1_index = K.argmax(ans1)

ans = Dense(vocab_size, activation='softmax')(ans2)


# ? T 8   ? 4   ans:一个词
model = Model([input_story, input_question], ans)

model.compile(
	optimizer=RMSprop(lr=5e-3),	# 训练集准确率不够高，降低学习率试试
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy']
)

# 训练模型
r = model.fit(
	[inputs_train, queries_train],
	answers_train,
	epochs=30,
	batch_size=32,
	validation_data=([inputs_test, queries_test], answers_test)
)

# 查看每个sotry中句子的权重
debug_model = Model([input_story, input_question], [story_weights1, story_weights2, ans1])

# 随机选一个story
story_index = np.random.choice(len(train_data))

# 从模型中取出权重
# 1xTx8
input_ = inputs_train[story_index: story_index + 1]
# 1x4
ques = queries_train[story_index: story_index + 1]
# 1xTx1
w1_, w2_, ans1_indexs= debug_model.predict([input_, ques])
# T
w1 = w1_.flatten()
w2 = w2_.flatten()

one_story, one_question, ans = train_data[story_index]
print("story:\n")
for i, line in enumerate(one_story):
	print("{:1.5f}".format(w1[i]), "\t", "{:1.5f}".format(w2[i]), "\t", " ".join(line))

print("question:", " ".join(one_question))
# argmax返回值与ans1_indexs有相同维度，ans1_indexs一维，返回值也1维
# ans1没经过Dense进行解码 没M->num_words, 每以ans1为目标进行梯度下降迭代。 取argmax没意义
# ans1_index = np.argmax(ans1_indexs)
# print("middle answer:", vocab[ ans1_index ])
print("answer:", ans)
pred_indexs = model.predict([input_, ques])	# 输出ans
pred_index = np.argmax(pred_indexs)
print("prediction:", vocab[pred_index])

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()