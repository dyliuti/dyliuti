import numpy as np


symbol_map = ['H', 'T']
pi = np.array([0.5, 0.5])
A = np.array([[0.1, 0.9], [0.8, 0.2]])
B = np.array([[0.6, 0.4], [0.3, 0.7]])
M, V = B.shape

# np.random.choice() 中的 p列表或ndarray长度要与 a 的长度相同。 不给出p就是均匀分布
def generate_sequence(N):
    s = np.random.choice(range(M), p=pi) 	# 初始状态
    x = np.random.choice(range(V), p=B[s]) 	# 初始观测
    sequence = [x]
    for n in range(N-1):
        s = np.random.choice(range(M), p=A[s]) # 下一个状态
        x = np.random.choice(range(V), p=B[s]) # 下一次观测
        sequence.append(x)
    return sequence


with open('Data/Markov/coin_data.txt', 'w') as f:
    for n in range(50):
        sequence_ = generate_sequence(30)
        sequence = ''.join(symbol_map[s] for s in sequence_)
        print(sequence)
        f.write("%s\n" % sequence)