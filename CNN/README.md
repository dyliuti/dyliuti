几大问题：

**1.不同尺寸大小的图像输入，如何获得相同尺寸的输出**

输入网络前对图像进行缩放

在卷积最后一层feature map（此时hw可能不同）使用GlobalAveragePooling或GlobalMaxPooling，就如resnet那样，经过池化，得到1x1xc的feature map