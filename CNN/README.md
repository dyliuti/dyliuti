# “经验总结”总是在“感性认识”之后。

## **1.不同尺寸大小的图像输入，如何获得相同尺寸的输出**

输入网络前对图像进行缩放

在卷积最后一层feature map（此时hw可能不同）使用GlobalAveragePooling或GlobalMaxPooling，就如resnet那样，经过池化，得到1x1xc的feature map

2.ssd目标检测解决的几个关键点

效率问题：

目标大小问题：

目标形状问题：



**程序运行说明：**

前提：所有程序都是在pycharm以dyliuti文件夹为根目录运行的。直接跳到如CNN文件夹命令行运行ssd_image.py等，会存在文件不存在等问题，这需要注意下。

style_transfer.py: 需要下载vgg16模型参数 vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 （不包括顶层的两个全连接层）放在 C:\Users\用户名\.keras\models 文件夹中。

ssd_image.py: 需要下载tensorflow的model，以及 ssd_mobilenet_v1_coco_2018_01_28 参数文件。我已经将该文件放在model文件夹中，所以只需下载model就行了。下载完后，将model放在Data\CNN目录下。图像素材下载放到Data\CNN\ssd文件夹下。

ssd_video.py: 除了ssd_image.py需要的配置，还需要下载 ffmpeg-win32-v3.2.4.exe ， 放在C:\Users\用户名\AppData\Local\imageio\ffmpeg 文件夹中。视频素材下载放到Data\CNN\ssd文件夹下。