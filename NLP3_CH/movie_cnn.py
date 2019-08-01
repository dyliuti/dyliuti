import tensorflow as tf

# 设置迭代次数
num_epochs = 5
# 设置BatchSize大小
batch_size = 256
#设置dropout保留比例
dropout_keep = 0.5
# 设置学习率
learning_rate = 0.0001
# 设置每轮显示的batches大小
show_every_n_batches = 20


def recommend_same_type_movie(movie_id_val, top_k=20):
	loaded_graph = tf.Graph()  #
	with tf.Session(graph=loaded_graph) as sess:  #
		# Load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keep_dims=True))
		normalized_movie_matrics = movie_matrics / norm_movie_matrics

		# 推荐同类型的电影
		probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
		probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
		sim = (probs_similarity.eval())
		print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))
		print("以下是给您的推荐：")
		p = np.squeeze(sim)
		p[np.argsort(p)[:-top_k]] = 0
		p = p / np.sum(p)
		results = set()
		while len(results) != 5:
			c = np.random.choice(3883, 1, p=p)[0]
			results.add(c)
		for val in (results):
			print(val)
			print(movies_orig[val])
		return result


def recommend_your_favorite_movie(user_id_val, top_k=10):
	loaded_graph = tf.Graph()  #
	with tf.Session(graph=loaded_graph) as sess:  #
		# Load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		# 推荐您喜欢的电影
		probs_embeddings = (users_matrics[user_id_val - 1]).reshape([1, 200])
		probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))
		sim = (probs_similarity.eval())

		print("以下是给您的推荐：")
		p = np.squeeze(sim)
		p[np.argsort(p)[:-top_k]] = 0
		p = p / np.sum(p)
		results = set()
		while len(results) != 5:
			c = np.random.choice(3883, 1, p=p)[0]
			results.add(c)
		for val in (results):
			print(val)
			print(movies_orig[val])

		return results