import sys, os
from datetime import datetime
import numpy as np
import tensorflow as tf
import imageio


input_video_name = 'Data\CNN\ssd\mobile'
# input_video_name = 'jiedao'
output_video_fps = 10


research_path = 'Data/CNN/models/research'
models_path = 'Data/CNN/models/research/object_detection'
# research_path = 'D:\Git\dyliuti\Data\CNN\models\research'
# models_path = 'D:\Git\dyliuti\Data\CNN\models\research\object_detection'
sys.path.append(research_path)
sys.path.append(models_path)

from utils import label_map_util
from utils import visualization_utils as vis_util

model_name = 'ssd_mobilenet_v1_coco_2018_01_28'
ckpt_path = '%s/%s/frozen_inference_graph.pb' % (models_path, model_name)
labels_path = '%s/data/mscoco_label_map.pbtxt' % models_path
num_classes = 90

image_dir = ''
image_paths = {os.path.join(image_dir, 'Data/CNN/ssd/image{}.jpg'.format(i)) for i in range(1, 3)}
image_size = (12, 8)
# 加载模型到缓存
detection_graph = tf.Graph()
with detection_graph.as_default():
	graph_def = tf.GraphDef()
	with tf.gfile.GFile(ckpt_path, 'rb') as fid:
		serialized_graph = fid.read()
		graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(graph_def, name='')

# 加载 label map
label_map = label_map_util.load_labelmap(labels_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                            use_display_name=True)
# 1：person 3: car
category_index = label_map_util.create_category_index(categories)
print("categories:")
print(categories)

# image -> numpy array
def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		# 定义输入输出 Tensors （从 detection_graph 中取出）
		image_tensor = detection_graph.get_tensor_by_name('image_tensor: 0')
		# detection_boxes： anchor检测到了类的
		detection_boxes = detection_graph.get_tensor_by_name('detection_boxes: 0')
		# detection_scores： anchor检测到的目标的置信度
		# detection_classes： anchor检测到的目标类别
		detection_scores = detection_graph.get_tensor_by_name('detection_scores: 0')
		detection_classes = detection_graph.get_tensor_by_name('detection_classes: 0')
		num_detections = detection_graph.get_tensor_by_name('num_detections: 0')

		video_reader = imageio.get_reader('%s.mp4' % input_video_name)
		video_writer = imageio.get_writer('%s_ssd.mp4' % input_video_name, fps=output_video_fps)

		# 处理每一帧
		t0 = datetime.now()
		n_frames = 0
		for frame in video_reader:
			# rename for convenience
			image_np = frame
			n_frames += 1

			# 扩展维度，因为模型要求输入图像shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)

			# 实际求得的anchor等
			(boxes, scores, classes, num) = sess.run(
				[detection_boxes, detection_scores, detection_classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})

			# 画出预测框
			vis_util.visualize_boxes_and_labels_on_image_array(
				image_np,
				np.squeeze(boxes),
				np.squeeze(classes).astype(np.int32),
				np.squeeze(scores),
				category_index,
				use_normalized_coordinates=True,
				line_thickness=8)

			# 将帧写入video中，而不是显示帧了
			video_writer.append_data(image_np)

		fps = n_frames / (datetime.now() - t0).total_seconds()
		print("已处理的帧: %s, Speed: %s fps" % (n_frames, fps))

		# 释放资源
		video_writer.close()