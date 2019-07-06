from Data.DataExtract import load_brown_with_limit_vocab
from NLP.Common.Util import get_bigram_prob, get_score
import numpy as np

# 获取Brown词库10000个词
# indexed_sentences: 被词向量化的句子集合, 每个句子不包括'START'，'END'
# word2index：字符词映射到索引，包含词'START'，'END'
indexed_sentences, word2index = load_brown_with_limit_vocab(n_vocab=10000)
V = len(word2index)
start_index = word2index['START']
end_index = word2index['END']
# 每个句子包含首尾'START'，'END'的贝叶斯估计的bigram
bigram_probs = get_bigram_prob(indexed_sentences, V, start_index, end_index, smooth=1)


# 获得字符串句子
index2word = dict([(v, k) for k, v in word2index.items()])
def get_words(indexed_sentence):
	return ''.join([index2word[i] for i in indexed_sentence])

# 随机选择一个句子
index = np.random.choice(len(indexed_sentences))
real_sentence = indexed_sentences[index]
maked_sentence = "I like a dog"
fake_sentence = "italy i dog go"
maked_sentence = [word2index[word] for word in maked_sentence.lower().split()]
fake_sentence = [word2index[word] for word in fake_sentence.lower().split()]

print("句子:", get_words(real_sentence), "真实度评分：", get_score(bigram_probs, real_sentence, start_index, end_index))
print("造真句子:", get_words(maked_sentence), "真实度评分：", get_score(bigram_probs, maked_sentence, start_index, end_index))
print("造假句子:", get_words(fake_sentence), "真实度评分：", get_score(bigram_probs, fake_sentence, start_index, end_index))
