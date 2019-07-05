from Data.DataExtract import load_wiki_with_limit_vocab
import numpy as np
from datetime import datetime

# 获取词的分布，频率高的词越容易抽中
def get_negative_sampling_distribution(sentences, vocab_size):
	word_freq = np.zeros(vocab_size)
	for indexed_sentence in sentences:
		for indexed_word in indexed_sentence
			word_freq[indexed_word] += 1

	# smooth 0.75是个不错的经验值
	p_word = word_freq ** 0.75
	p_word = p_word / np.sum(p_word)
	return p_word

# 获得中心词左右window_size个上下文词
def get_context(pos, sentence, window_size):
	start = max(0, pos - window_size)
	end = min(len(sentence), pos + window_size)

	contex = []
	for contex_pos, contex_index in enumerate(sentence[start: end], start=start):
		if contex_pos != pos:
			contex.append(contex_index)
	return contex

def stochastic_gradient(input, output, label, learning_rate, W1, W2):
	activation = W1[input].dot(W2[:, output])
	prob = sigmoid(activation)

	# gradients
	gW2 = np.outer(W1[input], prob - label)		# DxN
	gW1 = np.sum((prob - label) * W2[:, output], axis=1) # D

	W2[:, output] -= learning_rate * gW2 # DxN
	W1[input] -= learning_rate * gW1	# D

	# return cost(binary cross entropy)
	cost = label * np.log(prob + 1e-10) + (1 - label) * np.log(1 - prob + 1e-10)
	return cost.sum()

indexed_sentences, word2index = load_wiki_with_limit_vocab(n_vocab=20000)
vocab_size = len(word2index)

# 参数
window_size = 5
learning_rate = 0.025
final_learning_rate = 0.0001
epochs = 20
D = 50	# word embedding size

# 线性递减学习率
learning_rate_delta = (learning_rate - final_learning_rate) / epochs

W1 = np.random.randn(vocab_size, D) / np.sqrt(D)
W2 = np.random.randn(D, vocab_size) / np.sqrt(vocab_size)

p_word = get_negative_sampling_distribution()
# 重新对每个句子进行采样
threshold = 1e-5
p_drop = 1 - np.sqrt((threshold / p_word))


for epoch in range(epochs):
	np.random.shuffle(indexed_sentences)

	cost = 0
	t0 = datetime.now()
	for sentence in indexed_sentences:
		sentence = [indexed_word for indexed_word in sentence
					if np.random.random() < (1 - p_drop[indexed_word])]

		if len(sentence) < 2:
			continue

	# 均匀不放回抽样
	sentence_size = len(sentence)
	random_word_sequence = np.random.choice(sentence_size, size=sentence_size, replace=False)

	for word_pos in random_word_sequence:
		# 中心词
		word = sentence[word_pos]
		# 中心词的上下文
		contex = get_context(word_pos, sentence, window_size)
		# negitive word
		negative_word = np.random.choice(vocab_size, p=p_word)
		# 输出
		output = np.array(contex)

		cost_word = stochastic_gradient(word, output, True, learning_rate, W1, W2)
		cost_context = stochastic_gradient(negative_word, output, False, learning_rate, W1, W2)
		cost += cost_word + cost_context
