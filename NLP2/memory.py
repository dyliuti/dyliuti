from Data.DataExtract import load_bAbI_challange_data
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Lambda, Reshape, add, dot, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, RMSprop
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

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

# 1000x10x8
def stack_inputs(inputs, max_story_len, max_story_sents_len):
	for i, story in enumerate(inputs):
		# 没满足 max_story_len x max_story_sents_len 的，进行填充
		inputs[i] = np.concatenate( [story, np.zeros((max_story_len - story.shape[0], max_story_sents_len), 'int')] )
	return np.stack(inputs)

train_data = load_bAbI_challange_data(challenge_type='single_supporting_fact_10k', data_type='train')
test_data = load_bAbI_challange_data(challenge_type='single_supporting_fact_10k', data_type='test')

data = train_data + test_data

max_story_sents_len = max((len(sentence) for stories, _, _ in data for sentence in stories))
max_story_len = max((len(stories) for stories, _, _ in data))
max_query_len = max(len(question) for _, question, _ in data)

# <data 'tuple'>: ([['0', 'Mary', 'moved', 'to', 'the', 'bathroom', '.'], ['1', 'John', 'went', 'to', 'the', 'hallway', '.']], ['Where', 'is', 'Mary', '?'], 'bathroom')
# <vocab 'list'>: ['<PAD>', '.', '0', '1', '10', '12', '13', '3', '4', '6', '7', '9', '?', 'Daniel', 'John', 'Mary', 'Sandra', 'Where', 'back', 'bathroom', 'bedroom', 'garden', 'hallway', 'is', 'journeyed', 'kitchen', 'moved', 'office', 'the', 'to', 'travelled', 'went']
vocab = sorted(set(flatten(data)))
vocab.insert(0, '<PAD>')
vocab_size = len(vocab)

word2idx = {c: i for i, c in enumerate(vocab)}

# 对sotry中的每句索引化后进行pad，对query索引化后进行pad， 结果转换为索引
# NxTxLen
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

# 从list转换为 3-D numpy arrays
inputs_train = stack_inputs(inputs_train_, max_story_len, max_story_sents_len)
inputs_test = stack_inputs(inputs_test_, max_story_len, max_story_sents_len)




############### 建立模型 ###############
embedding_dim = 15

# 将story转换为一系列的embedding vectgor，每个story就像“bag of words”
input_story = Input(shape=(max_story_len, max_story_sents_len))
embedded_story = Embedding(vocab_size, embedding_dim)(input_story)
embedded_story = Lambda(lambda x: K.sum(x, axis=2))(embedded_story)
print("input_story_.shape, embedded_story.shape:", input_story.shape, embedded_story.shape)

# 将question转换为embedding，也是bag of words
input_question = Input(shape=(max_query_len, ))
embedded_question = Embedding(vocab_size, embedding_dim)(input_question)
embedded_question = Lambda(lambda x: K.sum(x, axis=1))(embedded_question)

# keras.core 为了可以和embedded_story进行点积
embedded_question = Reshape(target_shape=(1, embedding_dim))(embedded_question)
print("inp_q.shape, emb_q.shape:", input_question.shape, embedded_question.shape)

# embedded_story.shape        = (N, num sentences, embedding_dim)
# embedded_question.shape     = (N, 1, embedding_dim)
# 计算每个story中句子的权重
x = dot(inputs=[embedded_story, embedded_question], axes=2)
# flatten成一个vector
x = Reshape(target_shape=(max_story_len, ))(x)
x = Activation('softmax')(x)
# 再次unflatten，因为之后还需要点积
story_weights = Reshape(target_shape=(max_story_len, 1))(x)
print("story_weights.shape:", story_weights.shape)

x = dot(inputs=[story_weights, embedded_story], axes=1)
x = Reshape(target_shape=(embedding_dim, ))(x)
ans = Dense(vocab_size, activation='softmax')(x)

model = Model([input_story, input_question], ans)

model.compile(
	optimizer=RMSprop(lr=1e-2),
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy']
)

# train the model
r = model.fit(
	[inputs_train, queries_train],
	answers_train,
	epochs=4,
	batch_size=32,
	validation_data=([inputs_test, queries_test], answers_test)
)

# Check how we weight each input sentence given a story and question
debug_model = Model([input_story, input_question], story_weights)

# choose a random story
story_idx = np.random.choice(len(train_data))

# get weights from debug model
i = inputs_train[story_idx:story_idx+1]
q = queries_train[story_idx:story_idx+1]
w = debug_model.predict([i, q]).flatten()

story, question, ans = train_data[story_idx]
print("story:\n")
for i, line in enumerate(story):
	print("{:1.5f}".format(w[i]), "\t", " ".join(line))

print("question:", " ".join(question))
print("answer:", ans)


# pause so we can see the output
input("Hit enter to continue\n\n")