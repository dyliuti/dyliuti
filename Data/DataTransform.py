import numpy as np

# 将data进行one_hot编码
def y2one_hot(data, class_num=1):
	sample_num = len(data)
	res = np.zeros(shape=(sample_num, class_num))
	data = data.astype(np.int32)
	for i in range(sample_num):
		res[i, data[i]] = 1
	return res.astype('float32')
