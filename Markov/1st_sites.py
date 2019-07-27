# 数据集就两列，前一列表示上次网页的id，下一列表示下次网页的id
# 总共10个网页 id从0-9
# 初始网页id为-1
# 每一页会有B（bounce点击）和C（close）

transitions = {}
row_sums = {}

# s -> ei i=10的和为s
# count(s -> e) / count(s)    row_sums = count(s)  count(s -> e) = transitions[(s, e)]
# collect counts
for line in open('site_data.csv'):
    s, e = line.rstrip().split(',')
    transitions[(s, e)] = transitions.get((s, e), 0.) + 1
    row_sums[s] = row_sums.get(s, 0.) + 1

# 标准化
for pair, num in transitions.items():
    s, e = pair
    transitions[pair] = num / row_sums[s]

# 初始状态分布
print("初始状态分布")
# -1 -> e 频率
for pair, num in transitions.items():
    s, e = pair
    if s == '-1':
        print(e, num)

# 查看哪个网页有最高的点击率
for pair, num in transitions.items():
    s, e = pair
    if e == 'B':
        print("bounce rate for %s: %s" % (s, num))