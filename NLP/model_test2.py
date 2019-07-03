from Data.DataExtract import load_brown_with_limit_vocab
from Data.DataTransform import softmax, smoothed_loss
from NLP.Common.Util import get_bigram_prob
import numpy as np
import random
import matplotlib.pyplot as plt

# 一个词预测下一个词

indexed_sentences, word2index = load_brown_with_limit_vocab()
# 单词数
V = len(word2index)
start_index = word2index['START']
end_index = word2index['END']

bigram_probs = get_bigram_prob(indexed_sentences, V, start_index, end_index, smooth=0.1)

D = 100
W1 = np.random.randn(V, D) / np.sqrt(D)
W2 = np.random.randn(D, V) / np.sqrt(V)
W_bigram = np.log(bigram_probs)

lr = 0.001
losses = []
bigram_losses = []
epoches = 1
i = 0
for epoch in range(epoches):
	random.shuffle(indexed_sentences)

	for sentence in indexed_sentences:
		sentence = [word2index['START']] + sentence + [word2index['END']]
		n = len(sentence)
		prev_sentence = np.zeros(shape=(n-1, V))
		next_sentence = np.zeros(shape=(n-1, V))
		prev_sentence[np.arange(n-1), sentence[:n-1]] = 1
		next_sentence[np.arange(n-1), sentence[1:]] = 1

		# 前向传播
		hidden = np.tanh(prev_sentence.dot(W1))
		predictions = softmax(hidden.dot(W2))
		# 更新权重
		W2 = W2  - lr * hidden.T.dot(predictions - next_sentence)
		dhidden = (predictions - next_sentence).dot(W2.T) * (1 - hidden * hidden)
		W1 = W1 - lr * prev_sentence.T.dot(hidden)

		# 交叉熵, 句子的样本均值（对比时，去除句子不定长影响）
		loss = -np.sum(next_sentence * np.log(predictions)) / n - 1
		losses.append(loss)

		if epoch == 0:
			bigram_predictions = softmax(prev_sentence.dot(W_bigram))
			bigram_loss = -np.sum(next_sentence * np.log(bigram_predictions)) / (n - 1)
			bigram_losses.append(bigram_loss)

		if i % 100 == 0:
			print("epoch:", epoch, "sentence: %s/%s" % (i, len(sentence)), "loss:", loss)
		i += 1

# W_bigram = np.log(bigram_probs)
avg_bigram_loss = np.mean(bigram_losses)
print("avg_bigram_loss:", avg_bigram_loss)
plt.axhline(y=avg_bigram_loss, color='r', linestyle='-')
plt.plot(smoothed_loss(losses))
plt.plot(losses)
plt.show()

# softmax is the opposite of the log()
plt.subplot(1, 2, 1)
plt.title("Neural Network Model")
plt.imshow(np.tanh(W1).dot(W2))
plt.subplot(1, 2, 2)
plt.title("Bigram Probs")
plt.imshow(W_bigram)
plt.show()

