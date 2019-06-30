import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_minist_csv(pca=True):
	train_file_path = 'Data/Minist/train.csv'
	if not os.path.exists(train_file_path):
		print('%s not exist.' % train_file_path)

	train_file = pd.read_csv(train_file_path)
	train_np = train_file.values.astype(np.float32)		# float 不是 int 是有原因的
	np.random.shuffle(train_np)

	Y = train_file['label'].values.astype(np.int32)		# seris -> np
	X_pd = train_file.drop('label', axis=1)
	X = X_pd.values.astype(np.float32) / 255.0		# Min-Max Scaling -> normalization
	# X = train_file.values[:, 1:]  # 同上X, 但没标准化

	# 训练集中分训练、验证集
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
	# PCA降维，丢弃杂质
	if pca is True:
		pca = PCA()
		X_train = pca.fit_transform(X_train)
		X_test = pca.transform(X_test)
		plot_cumulative_variance(pca)
	# 各自Normalize服从正态分布   梯度图变圆有利于梯度下降
	X_train = (X_train - np.mean(X_train)) / np.std(X_train)
	X_test = (X_test - np.mean(X_test)) / np.std(X_test)
	return X_train, X_test, Y_train, Y_test

def plot_cumulative_variance(pca):
	P = []
	# 用奇异值S来解释方差
	for p in pca.explained_variance_ratio_:
		if len(P) == 0:
			P.append(p)
		else:
			P.append(p + P[-1])
	plt.plot(P)
	plt.show
	return P

def load_facial_expression_data(balance_ones=True):
	# images are 48x48 = 2304 size vectors
	train_file_path = 'Data/FacialExpression/fer2013.csv'
	if not os.path.exists(train_file_path):
		print('%s not exist.' % train_file_path)

	Y = []
	X = []
	first = True
	for line in open(train_file_path):
		if first:
			first = False
		else:
			row = line.split(',')
			Y.append(int(row[0]))
			X.append([int(p) for p in row[1].split()])

	X, Y = np.array(X) / 255.0, np.array(Y)

	if balance_ones:
		# balance the 1 class
		X0, Y0 = X[Y!=1, :], Y[Y!=1]
		X1 = X[Y==1, :]
		X1 = np.repeat(X1, 9, axis=0)
		X = np.vstack([X0, X1])
		Y = np.concatenate((Y0, [1]*len(X1)))

	return X, Y



