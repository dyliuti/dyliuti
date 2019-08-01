# 数据集就两列，前一列表示上次网页的id，下一列表示下次网页的id
# 总共10个网页 id从0-9
# 初始网页id为-1
# 每一页会有B（bounce点击）和C（close）

# 常规马尔可夫，状态及观测结果，状态序列即观测序列
transitions = {}
state_prev_num = {}

# count(s -> e) / count(s)    state_prev_num = count(s)  count(s -> e) = transitions[(s, e)]
for line in open('Markov/site_data.csv'):
    start, end = line.rstrip().split(',')
    transitions[(start, end)] = transitions.get((start, end), 0.) + 1
    state_prev_num[start] = state_prev_num.get(start, 0.) + 1

# 频数->频率化  P(Zt+1 | Zt)
for pair, num in transitions.items():
    start, end = pair
    transitions[pair] = num / state_prev_num[start]

# 初始状态分布
print("初始状态(观测)分布")
# -1 -> e 频率
for pair, pro in transitions.items():
    start, end = pair
    if start == '-1':
        print(end, pro)

# 查看哪个网页有最高的点击率
for pair, pro in transitions.items():
    start, end = pair
    if end == 'B':
        print("bounce rate for %s: %s" % (start, pro))