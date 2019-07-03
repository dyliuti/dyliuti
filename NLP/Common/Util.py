import numpy as np

def get_bigram_prob(sentences, vocab_size, start_index, end_index, smooth=1):
	# 极大似然估计->贝叶斯估计
	bigram = np.ones((vocab_size, vocab_size)) * smooth
	for sentence in sentences:
		for i in range(len(sentence)):
			if i == 0:
				bigram[start_index, sentence[i]] += 1
			elif i == len(sentence) - 1:
				bigram[sentence[i], end_index] += 1
			else:
				bigram[sentence[i-1], sentence[i]] += 1
	# P(Y|X) = #(X, Y) / #(X)
	bigram_prob = bigram / np.sum(bigram, axis=1, keepdims=True)
	return bigram_prob

def get_score(bigram_probs, sentence, start_index, end_index):
	score = 0
	for i in range(len(sentence)):
		if i == 0:
			score += np.log(bigram_probs[start_index, sentence[i]])
		elif i == len(bigram_probs[0] - 1):
			score += np.log(bigram_probs[sentence[i], end_index])
		else:
			score += np.log(bigram_probs[sentence[i - 1], sentence[i]])

	# 标准化score
	return score / (len(sentence) + 1)








