from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from datetime import datetime


def VGG16_AvgPool(shape):
	# 不包括顶层的全连接层 fc6、fc7
	vgg = VGG16(input_shape=shape, weights='imagenet', include_top = False)
	new_model = Sequential()
	for layer in vgg.layers:
		if layer.__class__ == MaxPooling2D:
			new_model.add(AveragePooling2D())
			# new_model.add(MaxPooling2D())
		else:
			new_model.add(layer)
	return new_model

def gram_matrix(img):
	# img的shape: (H, W, C) (C = # feature maps)
	# 先转换为(C, H*W)
	# Turn a nD tensor into a 2D tensor with same 0th dimension.
	X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))

	# 计算gram矩阵：gram = XX^T / N  常量N并不重要，也只是个权重而已
	G = K.dot(X, K.transpose(X)) / img.get_shape().num_elements()
	print(img.get_shape().num_elements())
	return G

def style_loss(y, t):
	return K.mean(K.square(gram_matrix(y) - gram_matrix(t)))

def unpreprocess(img):
	img[..., 0] += 103.939
	img[..., 1] += 116.779
	img[..., 2] += 126.68
	img = img[..., ::-1]
	return img

def minimize(fn, epochs, batch_shape):
	t0 = datetime.now()
	losses = []
	x = np.random.randn(np.prod(batch_shape))
	for i in range(epochs):
		x, l, _ = fmin_l_bfgs_b(
		  func=fn,
		  x0=x,
		  maxfun=20
		)
		x = np.clip(x, -127, 127)
		print("iter=%s, loss=%s" % (i, l))
		losses.append(l)

	print("duration:", datetime.now() - t0)
	plt.plot(losses)
	plt.show()

	newimg = x.reshape(*batch_shape)
	final_img = unpreprocess(newimg)
	return final_img[0]

