from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import tensorflow as tf
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from CNN.Common.Model import VGG16_AvgPool, style_loss, minimize

# 需要把vgg16模型参数 vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 放在 C:\Users\用户名\.keras\models 文件夹中
def load_img(path, shape=None):
	img = image.load_img(path, target_size=shape)
	# 将图片转换为array，用于vgg输入
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	# caffe: will convert the images from RGB to BGR, 默认为caffe
	# then will zero-center each color channel with respect to the ImageNet dataset, without scaling.
	# tf: will scale pixels between -1 and 1
	x = preprocess_input(x)
	return x

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

img_shape = (225, 300)
content_img = load_img('Data/CNN/content/sydney.jpg', shape=img_shape)
style_img = load_img('Data/CNN/styles/starrynight.jpg', shape=img_shape)
batch_shape = content_img.shape
shape = batch_shape[1:]		# 三维 hwc

##### 建立模型 #####
vgg = VGG16_AvgPool(shape=shape)
# vgg.summary()

# 1,2,4,5,7-9,11-13,15-17 只需要一个输出
print(dir(vgg.layers[12]))
content_model = Model(vgg.input, vgg.layers[12].get_output_at(0))
content_target = K.variable(content_model.predict(content_img))

# 多个卷积层作为输出
sym_style_outputs = [layer.get_output_at(1) for layer in vgg.layers if layer.name.endswith('conv1')]
style_model = Model(vgg.input, sym_style_outputs)
style_outputs = style_model.predict(style_img)
var_style_outputs = [K.variable(x) for x in style_outputs]	# 转换为符号

# 设置content loss权重为1， style loss权重如下: len同作为style输出的conv层数
style_weights = [0.2, 0.4, 0.3, 0.5, 0.2]

loss = K.mean(K.square((content_model.output - content_target)))
for w, sym, actual in zip(style_weights, sym_style_outputs, var_style_outputs):
	loss += w * style_loss(sym[0], actual[0])
	print(sym[0].get_shape())

grads = K.gradients(loss, vgg.input)

loss_grads = K.function(
	inputs=[vgg.input],
	outputs=[loss] + grads
)

def loss_grads_wrapper(x_vec):
	loss_, grad = loss_grads([x_vec.reshape(*batch_shape)])
	return loss_.astype(np.float64), grad.flatten().astype(np.float64)

def scale_img(x):
	x = x - x.min()
	x = x / x.max()
	return x

final_img = minimize(loss_grads_wrapper, 10, batch_shape)
plt.imshow(scale_img(final_img))
plt.show()
