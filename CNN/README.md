**“纸上得来终觉浅，绝知此事要躬行。”  —— 陆游**

**"Practice，practice，practice makes perfect" —— dyliuti**

**1.卷积神经网络的特点**

除1x1xc的卷积核外，卷积神经网络是特征提取器。从构成结构来看，除去BN、非线性转换等优化梯度下降的操作外，骨干结构是卷积+池化。卷积用于提取特征；池化降参数、保持位置关系，也是随着网络加深，感受野越来越大的原因。因为池化的特性，导致了池化在图像处理中特别好用，但在自然语言处理中被抛弃。池化也是解释卷积神经网络浅层检测到目标的纹理，深层检测到目标的整体轮廓的关键。换个应用领域，从目标检测上来看池化，浅层用低比例anchor可以检测小目标，深层用高比例anchor可以检测大目标。（卷积stride大于1的时候hw也缩小了）。

1x1xc的卷积核较特殊，可以看做是对通道维的全连接层，或者看做是作用在像素，共享参数的全连接层。

**2.不同尺寸大小的图像输入，如何获得相同尺寸的输出**

输入网络前对图像进行缩放

在卷积最后一层feature map（此时hw可能不同）使用GlobalAveragePooling或GlobalMaxPooling（keras），tensorflow是tf.reduce_mean, tf.reduce_max，就如resnet那样，经过池化，得到NxC的feature map

**3.ssd目标检测解决的几个关键点**

说明之前，需要知道的是，ssd的迭代目标是以anchor为基础的，就是预测框。如L1损失的anchor4个偏移量，anchor检测到图像目标的交叉熵（背景类不计入）。

**效率问题：**

**目标大小问题：**

**目标形状问题：**



**程序运行说明：**

**前提：**所有程序都是在pycharm以dyliuti文件夹为根目录运行的。直接跳到如CNN文件夹命令行运行ssd_image.py等，会存在文件不存在等问题，这需要注意下。

**style_transfer.py:** 需要下载vgg16模型参数 vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 （不包括顶层的两个全连接层）放在 C:\Users\用户名\.keras\models 文件夹中。

效果：一张图，与油画结合，生成具有油画风格的图。

**ssd_image.py:** 需要下载tensorflow的model，以及 ssd_mobilenet_v1_coco_2018_01_28 参数文件。我已经将该文件放在model文件夹中，所以只需下载model就行了。下载完后，将model放在Data\CNN目录下。图像素材下载放到Data\CNN\ssd文件夹下。

效果：检测出图片中的目标，以框标注出来。

**ssd_video.py:** 除了ssd_image.py需要的配置，还需要下载 ffmpeg-win32-v3.2.4.exe ， 放在C:\Users\用户名\AppData\Local\imageio\ffmpeg 文件夹中。视频素材下载放到Data\CNN\ssd文件夹下。

效果：检测出视频中的目标，以框标注出来，成果物也是视频。

**resnet_*.py:** 需要下载模型参数 resnet50_weights_tf_dim_ordering_tf_kernels.h5 放在 C:\Users\用户名\.keras\models文件夹中。

效果：通过手动搭建resnet50，包括几个关键的模块，如conv_block, identity_block等，熟悉resnet50的各个环节。对原生的tensorflow神经网络再次封装，每个类保持相同的接口。参数复用， 通过对比keras中resnet50的输出结果与手动搭建的resnet50输出结果，查看效果。



**备注**：提供的vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5   与 resnet50_weights_tf_dim_ordering_tf_ kernels.h5 都是V2版本的，具体看你安装的keras.appliacation.vgg16类中用的是v几版本。不对的话，点开提示下载的路径，进行下载就好了。