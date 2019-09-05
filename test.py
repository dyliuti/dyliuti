import mxnet.ndarray as nd
import jieba
file_path = "Data/NLP2/党章/中国共产党党章.txt"
en_path = "Data/NLP2/党章/中国共产党党章En.txt"
ch_path = "Data/NLP2/党章/中国共产党党章Ch.txt"
ch_path_fenci = "Data/NLP2/党章/中国共产党党章Ch_FenCi.txt"
n = 0
en_lines = []
ch_lines = []
ch_lines_fenci = []
with open(file_path, encoding='utf-8') as f:
    data = f.readlines()
    for n, line in enumerate(data):
        if n % 2 == 1:
            en_lines.append(line)
        else:
            ch_lines.append(line)
            ch_lines_fenci.append(" ".join(jieba.lcut(line)))

with open(en_path, mode='w', encoding='utf-8') as en_f:
    en_f.writelines(en_lines)
with open(ch_path, mode="w", encoding='utf-8') as ch_f:
    ch_f.writelines(ch_lines)
with open(ch_path_fenci, mode="w", encoding='utf-8') as ch_f:
    ch_f.writelines(ch_lines_fenci)
print(ch_lines_fenci[0].replace("\n", "").split())
print(len(ch_lines_fenci[0].replace("\n", "")))

import jieba
content = ch_lines[5].replace("\n", "")
segs_1_list = jieba.lcut(content, cut_all=False)
print(segs_1_list)


vocab_size = 5
res = nd.one_hot(nd.array([0, 2]), vocab_size)

def to_onehot(X, size):  # 本函数已保存在d2lzh包中方便以后使用
    return [nd.one_hot(x, size) for x in X.T]

X = nd.arange(10).reshape((2, 5))
inputs = to_onehot(X, vocab_size)
len(inputs), inputs[0].shape

a = X.T.reshape((-1,))	# 5 x 2	-> 10
a_ = to_onehot(a, 8)	# 10 -> 10,1 -> 10 8  class_num, V

import numpy as np
# b = np.arange(10).reshape(2, 5)
b = [nd.array([0, 1, 2, 3, 4]), nd.array([5, 6, 7, 8, 9])]
outputs = nd.concat(*b, dim=0)

import re

phone = "2004,959!559 # 这是一个国外电话号码"

phone = phone.translate(str.maketrans({",":" ,", "!":" !", "?": " ?", ".": " ."}))
test = "add,dsad\t你好！我是。"
a, b = test.split('\t')
a = re.sub(",!?.", " ",  a)
b = re.sub("，！？。", "",  b)
a = a.translate(str.maketrans(",!?.", ""))
b = b.translate(str.maketrans("，！？。", "    "))
# 删除字符串中的 Python注释
num = re.sub(r'#.*$', "", phone)
print("电话号码是: ", num)

# 删除非数字(-)的字符串
num = re.sub(r',!', "", phone)
print("电话号码是 : ", num)
