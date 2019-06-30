import numpy as np

def forward(X, W, b):
	z = X.dot(W) + b
	a = z - np.max(z)	# 排除上溢（a最大为0，expz不会上溢）、下溢(分母必有一项1，不会因z都很小分母小)
	expz = np.exp(a)
	p_y = expz / expz.sum(axis=1, keepdims=True) # 交叉熵计算需要log p_y p_y接近0也不太好
	return p_y

# L2 loss
def gradW(p_y, y_true, X):
	# z = wx+b -> w = x-1z x-1看成转置
	return X.T.dot(p_y - y_true)

# b 是对于每个类别的偏差
def gradb(p_y, y_true):
	return (p_y - y_true).sum(axis=0)

# 交叉熵
def cost(p_y, y_true):
	loss = - np.sum(y_true * np.log(p_y))
	return loss

def predict(p_y):
	return np.argmax(p_y, axis=1)

# y非one-hot编码 y_true是one-hot编码
def error_rate(p_y, y):
	prediction = predict(p_y)	# 返回最大概率所在的索引
	return np.mean(prediction != y)

def forward(X, W1, b1, W2, b2):
	# sigmoid
    # Z = 1 / (1 + np.exp(-( X.dot(W1) + b1 )))
    # relu
	Z = X.dot(W1) + b1
	Z[Z < 0] = 0

	A = Z.dot(W2) + b2
	expA = np.exp(A)
	Y = expA / expA.sum(axis=1, keepdims=True)
	return Z, Y

def derivative_w2(Z, Y, T):
	return Z.T.dot(Y - T)

def derivative_b2(Y, T):
	return (Y - T).sum(axis=0)

# relu(X.W1).W2 = Y     W1 = X-1.Y.W2-1  再加个非线性转换relu(Z > 0)
def derivative_w1(X, Z, Y, T, W2):
	# return X.T.dot( ( ( Y-T ).dot(W2.T) * ( Z*(1 - Z) ) ) ) # for sigmoid
	return X.T.dot( ( ( Y-T ).dot(W2.T) * (Z > 0) ) ) # for relu

# relu(X.W1+b1).W2 = Y  b1 = Y.W2-1  再加个非线性转换relu(Z > 0)
def derivative_b1(Z, Y, T, W2):
	# return (( Y-T ).dot(W2.T) * ( Z*(1 - Z) )).sum(axis=0) # for sigmoid
	return (( Y-T ).dot(W2.T) * (Z > 0)).sum(axis=0) # for relu

