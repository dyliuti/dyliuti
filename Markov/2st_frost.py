import numpy as np
import string
import sys

# 一个句子的开头词分布
initial = {}
# 一个句子中的第二个词的分布，只有一个前面的词
second_word = {}
transitions = {}

# unfortunately these work different ways
# If there are two arguments, they must be strings of equal length, and
#         in the resulting dictionary, each character in x will be mapped to the
#         character at the same position in y. If there is a third argument, it
#         must be a string, whose characters will be mapped to None in the result.
# maketrans两个参数时，参数相同
# 3个参数时，第三个参数中的字符会成为空
def remove_punctuation(s):
    return s.translate(str.maketrans('','',string.punctuation))


def add2dict(key_map_next_word, key, next_word):
    if key not in key_map_next_word:
        key_map_next_word[key] = []
        key_map_next_word[key].append(next_word)

for line in open('robert_frost.txt', encoding='utf-8'):
    # 按空格分离后成为list
    tokens = remove_punctuation(line.rstrip().lower()).split()
    # 句子长度
    T = len(tokens)
    for i in range(T):
        word_t = tokens[i]
        if i == 0:
            # 计算每个句子第一个词的分布
            initial[word_t] = initial.get(word_t, 0.) + 1
        else:
            word_t_1 = tokens[i-1]
            if i == T - 1:
                # 最后一个词时的分布
                add2dict(transitions, (word_t_1, word_t), 'END')
            if i == 1:
                # 句子开头第一个词->第二个词的分布
                add2dict(second_word, word_t_1, word_t)
            else:
                # 第三个词开始
                word_t_2 = tokens[i-2]
                add2dict(transitions, (word_t_2, word_t_1), word_t)


# 标准化第一个词的分布 第一个词：频率 的字典
initial_total = sum(initial.values())
for t, c in initial.items():
    initial[t] = c / initial_total

def list2pdict(ts):
    # 对key后面的可能词列表进行频率统计，返回 可能词：频率
    d = {}
    n = len(ts)
    for t in ts:
        d[t] = d.get(t, 0.) + 1
    for t, c in d.items():
        d[t] = c / n
    return d

for t_1, ts in second_word.items():
    # 将第一个词后面的可能词（list） -> 第一个词后面的可能词：频率（dict）
    second_word[t_1] = list2pdict(ts)

for k, ts in transitions.items():
    # 前两个词后面的可能词（list） -> 前两个词后面的可能词：频率（dict）
    transitions[k] = list2pdict(ts)


# 1.产生1个随机概率值
# 2.随机选择1个映射值。选择方式：遍历可能的值，叠加概率，概率>随机概率值时，此时遍历的值就是随机产生的下一个词。

# 输入为  下一个可能的词：频率   的字典
# 输入的其实就是转台转移矩阵中的行了，但没有概率0的词
def sample_word(map_pair):
    p0 = np.random.random()
    print("p0:", p0)
    cumulative = 0
    for next_word, p in map_pair.items():
        cumulative += p
        if p0 < cumulative:
            return next_word
    assert(False) # should never get here

# initial： 句子开始词：频率
# second_word  句子开始词  映射  第二个词：频率
# sample_word 前两个词 映射 第三个词：频率
def generate():
    for i in range(4):
        sentence =[]
        # 初始化句子中的第一个词
        w0 = sample_word(initial)
        sentence.append(w0)
        # 对句子中的第二个词进行采样
        w1 = sample_word(second_word[w0])
        sentence.append(w1)

        # 2阶马尔可夫
        while True:
            w2 = sample_word(transitions[(w0, w1)])
            if w2 == 'END':
                break
            sentence.append(w2)
            w0 = w1
            w1 = w2
        print(' '.join(sentence))

generate()


