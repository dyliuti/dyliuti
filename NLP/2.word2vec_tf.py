from Data.DataExtract import load_wiki_with_limit_vocab, load_brown_with_limit_vocab
from NLP.Common.Util import analogy
import numpy as np
from datetime import datetime
from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt
import os, json
import tensorflow as tf

# 获取词的分布，频率高的词越容易抽中
def get_negative_sampling_distribution(sentences, vocab_size):
	word_freq = np.zeros(vocab_size)
	for indexed_sentence in sentences:
		for indexed_word in indexed_sentence:
			word_freq[indexed_word] += 1

	# smooth 0.75是个不错的经验值
	# 从假设为均匀分布进行调整，不再是对均匀分布的极大似然估计
	p_word = word_freq ** 0.75
	p_word = p_word / np.sum(p_word)
	return p_word

# 获得中心词左右window_size个上下文词
def get_context(pos, sentence, window_size):
	start = max(0, pos - window_size)
	end = min(len(sentence), pos + window_size)

	context = []
	for context_pos, context_index in enumerate(sentence[start: end], start=start):
		if context_pos != pos:
			context.append(context_index)
	return context

def stochastic_gradient(input, output, label, learning_rate, W1, W2):
	# W2:DxV V中每一列是一个单词o的潜在语义Dx1 对输入单词i潜在语义W1:1xD的相互作用 得到输出单词o的概率
	# 换句话说，W2中只有部分对结果有效
	# input:nxV W1:VxD W2:DxV -> nxV 效率化后
	# input:W1	W1:nxD W2:DxV -> nxV 令n=1 (input样本为1个)
	# input:W1  W1:D   W2:DxV -> V	 输出对结果有影响的是Vc列(output所在的列，因为要与output比较，Vc为output维度)
	# input:W1  W1:D   W2:DxVc -> Vc
	# Vc = min(len(sentence), 2 x window_size)  Vc不是2000 Vc针对一个句子
	try:
		activation = W1[input].dot(W2[:, output])	# 1xD DxVc -> Vc
		prob = sigmoid(activation)	# Vc
		# gradients
		# W1.T.dot(output) <-> Dx1.dot(1xVc) <-> D.outer(Vc)
		gW2 = np.outer(W1[input], prob - label)		# DxVc
		# output.dot(W2.T) <-> 1xVc.dot(VcxD(W2.T)) -> sum(Vc * DxVc(W2), axis=1)
		gW1 = np.sum((prob - label) * W2[:, output], axis=1) # D

		W2[:, output] -= learning_rate * gW2 # DxN
		W1[input] -= learning_rate * gW1	# D

		# return cost(binary cross entropy)
		cost = label * np.log(prob + 1e-10) + (1 - label) * np.log(1 - prob + 1e-10)
	except IndexError:
		print('input:', input, 'output:', output)
	else:
		return -cost.sum()

def train_model():
	indexed_sentences, word2index = load_wiki_with_limit_vocab(n_vocab=2000)
	print("句子总数: ", len(indexed_sentences))
	# indexed_sentences, word2index = load_brown_with_limit_vocab(n_vocab=2000)
	vocab_size = len(word2index)

	# 参数
	window_size = 5
	epochs = 1
	D = 100	# word embedding size

	p_word = get_negative_sampling_distribution(indexed_sentences, vocab_size)
	# 重新对每个句子进行采样，频率越大，丢弃的概率越大(近似于直接把很高频的丢弃)
	threshold = 1e-5
	p_drop = 1 - np.sqrt((threshold / p_word))

	W1 = (np.random.randn(vocab_size, D) / np.sqrt(D)).astype(np.float32)
	W2 = (np.random.randn(D, vocab_size) / np.sqrt(vocab_size)).astype(np.float32)
	tfW1 = tf.Variable(W1)
	tfW2_T = tf.Variable(W2.T)
	tf_word = tf.placeholder(tf.int32, shape=(None, ))	# 中间词
	tf_negative_word = tf.placeholder(tf.int32, shape=(None,))	# 固定上下文，替换中间词
	tf_context = tf.placeholder(tf.int32, shape=(None,))

	tfW1_word = tf.nn.embedding_lookup(tfW1, tf_word)	# D
	tfW2_T_context = tf.nn.embedding_lookup(tfW2_T, tf_context)	# N x D
	# 矩阵相乘转换为elemetn-wise
	correct_logits = tf.reduce_sum(tfW1_word * tfW2_T_context, axis=1) # N
	# binary cross-entropy
	word_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones(tf.shape(correct_logits)),
												   		logits=correct_logits)
	# binary cross-entropy
	tfW1_negative = tf.nn.embedding_lookup(tfW1, tf_negative_word)
	negative_logits = tf.reduce_sum(tfW1_negative * tfW2_T_context, axis=1)
	negative_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones(tf.shape(correct_logits)),
															logits=negative_logits)

	loss = tf.reduce_mean(word_loss) + tf.reduce_mean(negative_loss)
	train_optimize = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9)
	train_step = train_optimize.minimize(loss)

	session = tf.Session()
	init = tf.global_variables_initializer()
	session.run(init)

	costs = []
	t0 = datetime.now()
	for epoch in range(epochs):
		np.random.shuffle(indexed_sentences)
		cost = 0
		num = 0
		words = []
		negative_words = []
		contexts = []
		for sentence in indexed_sentences:
			# 预处理每个句子，极大降低高频词汇出现概率
			sentence = [indexed_word for indexed_word in sentence
						if np.random.random() < (1 - p_drop[indexed_word])]
			# 避免output为None出现异常
			if len(sentence) < 2:
				continue

			# 均匀不放回抽样
			sentence_size = len(sentence)
			random_word_sequence = np.random.choice(sentence_size, size=sentence_size, replace=False)
			for word_pos in random_word_sequence:
				# 中心词
				indexed_word = sentence[word_pos]
				# 中心词的上下文
				context = get_context(word_pos, sentence, window_size)
				# negitive word
				negative_word = np.random.choice(vocab_size, p=p_word)	# 替代中心词indexed_word的位置

				n = len(context)
				# 三者长度一样, 单个word的话，乘不乘n无所谓，element-wise运算都不会错。但多个word，W1、W2 shape就要一样
				words += [indexed_word] * n
				negative_words += [negative_word] * n
				contexts += context

			if len(words) >= 128:
				_, c = session.run(fetches=(train_step, loss),
								   feed_dict={tf_word: words, tf_negative_word: negative_words, tf_context: contexts})

				cost += c
				words = []
				negative_words = []
				contexts = []

			num += 1
			if num % 100 == 0: 	# 每处理100个句子打印一次
				print("epoch: %d. proceessed %d/%d." % (epoch, num, len(indexed_sentences)), "cost: ", cost)
		print("Elapsed time: ", datetime.now() - t0, "cost: ", cost)
		costs.append(cost)

	plt.plot(costs)
	plt.show()

	W1, W2_T = session.run((tfW1, tfW2_T))
	return word2index, W1, W2_T.T

def save_model(savedir, word2index, W1, W2):
	if not os.path.exists(savedir):
		os.mkdir(savedir)
	with open('%s/word2index.json' % savedir, 'w') as f:
		json.dump(word2index, f)
	np.savez('%s/weights.npz' % savedir, W1, W2)

def load_model(savedir):
	if os.path.exists(savedir):
		with open('%s/word2index.json' % savedir) as f:
			word2idx = json.load(f)
		npz = np.load('%s/weights.npz' % savedir)
		W1 = npz['arr_0']
		W2 = npz['arr_1']
		return word2idx, W1, W2


def test_model(word2index, W1, W2):
	index2word = {i: w for w, i in word2index.items()}
	# 也可以试 We = W2.T
	for word_embedding in (W1, (W1 + W2.T) / 2):
		print("**********")

		analogy('king', 'man', 'queen', 'woman', word2index, index2word, word_embedding)
		analogy('king', 'prince', 'queen', 'princess', word2index, index2word, word_embedding)
		analogy('miami', 'florida', 'dallas', 'texas', word2index, index2word, word_embedding)
		analogy('einstein', 'scientist', 'picasso', 'painter', word2index, index2word, word_embedding)
		analogy('japan', 'sushi', 'germany', 'bratwurst', word2index, index2word, word_embedding)
		analogy('man', 'woman', 'he', 'she', word2index, index2word, word_embedding)

def main():
	savedir = 'NLP/word2vec_tf_model'
	if not os.path.exists(savedir):
		word2index, W1, W2 = train_model()
		test_model(word2index, W1, W2)
		save_model(savedir, word2index, W1, W2)
	else:
		word2index, W1, W2 = load_model(savedir)
		test_model(word2index, W1, W2)

if  __name__ == '__main__':
	main()
