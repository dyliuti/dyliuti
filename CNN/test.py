from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet')
model.summary()


from keras.applications.resnet50 import ResNet50
model = ResNet50()
model.summary()

from keras.applications.resnet50 import ResNet50
model = ResNet50(include_top=False)
model.summary()