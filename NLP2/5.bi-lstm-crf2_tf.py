# encoding=utf8
import os, itertools
import numpy as np

from NLP2.Common.loader import load_sentences, update_tag_scheme
from NLP2.Common.loader import char_mapping, tag_mapping
from NLP2.Common.loader import augment_with_pretrained, prepare_dataset

from NLP2.Common.utils import test_ner
from NLP2.Common.data_utils import load_word2vec, BatchManager, iobes_iob
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

from NLP2.Common import rnncell as rnn

# 极大似然法，损失函数为负的极大似然函数。传入参数有trans，logits，targets  <- crf层的限制吧
# 迭代后用vitebi求出最可能路径

# accuracy:  98.03%; precision:  87.58%; recall:  84.41%; FB1:  85.97
#               LOC: precision:  88.34%; recall:  86.61%; FB1:  87.46  170600
#               ORG: precision:  82.56%; recall:  77.56%; FB1:  79.98  90000
#               PER: precision:  91.47%; recall:  87.57%; FB1:  89.48  83200

# clip
# accuracy:  98.02%; precision:  86.14%; recall:  85.20%; FB1:  85.67
#               LOC: precision:  86.39%; recall:  87.53%; FB1:  86.95  176300
#               ORG: precision:  85.05%; recall:  77.77%; FB1:  81.24  87600
#               PER: precision:  86.73%; recall:  88.72%; FB1:  87.71  88900

# 模型参数
seg_dim = 20  #
word_dim = 100
lstm_dim = 100
tag_schema = "iobes"    # 标签类型 iobes 或 iob     # 4x3 + 0 = 13

# 参数
pre_emb = True
clip = 5         # "Gradient clip"
p_dropout = 0.5
batch_size = 20
lr = 0.001
zeros = False       # 空格转为数字0
lower = True        # 大写转小写
max_epoch = 10

script = "NLP2/conlleval"
result_path= "Data/NLP2/NER"
emb_file = "Data/NLP2/NER/wiki_100.utf8"
train_file = os.path.join("Data/NLP2/NER", "example.train")
dev_file = os.path.join("Data/NLP2/NER", "example.dev")
test_file = os.path.join("Data/NLP2/NER", "example.test")

# train_sentences: [['海', 'O'], ['钓', 'O'], ['比', 'O'], ['赛', 'O'], ['地', 'O'], ['点', 'O'], ['在', 'O'], ['厦', 'B-LOC'], ['门', 'E-LOC'], ['与', 'O'], ['金', 'B-LOC'], ['门', 'E-LOC'], ['之', 'O'], ['间', 'O'], ['的', 'O'], ['海', 'O'], ['域', 'O'], ['。', 'O']]
# dev_sentences  : [['在', 'O'], ['这', 'O'], ['里', 'O'], ['恕', 'O'], ['弟', 'O'], ['不', 'O'], ['恭', 'O'], ['之', 'O'], ['罪', 'O'], ['，', 'O'], ['敢', 'O'], ['在', 'O'], ['尊', 'O'], ['前', 'O'], ['一', 'O'], ['诤', 'O'], ['：', 'O'], ['前', 'O'], ['人', 'O'], ['论', 'O'], ['书', 'O'], ['，', 'O'], ['每', 'O'], ['曰', 'O'], ['“', 'O'], ['字', 'O'], ['字', 'O'], ['有', 'O'], ['来', 'O'], ['历', 'O'], ['，', 'O'], ['笔', 'O'], ['笔', 'O'], ['有', 'O'], ['出', 'O'], ['处', 'O'], ['”', 'O'], ['，', 'O'], ['细', 'O'], ['读', 'O'], ['公', 'O'], ['字', 'O'], ['，', 'O'], ['何', 'O'], ['尝', 'O'], ['跳', 'O'], ['出', 'O'], ['前', 'O'], ['人', 'O'], ['藩', 'O'], ['篱', 'O'], ['，', 'O'], ['自', 'O'], ['隶', 'O'], ['变', 'O'], ['而', 'O'], ['后', 'O'], ['，', 'O'], ['直', 'O'], ['至', 'O'], ['明', 'O'], ['季', 'O'], ['，', 'O'], ['兄', 'O'], ['有', 'O'], ['何', 'O'], ['新', 'O'], ['出', 'O'], ['？', 'O']]
# test_sentences : [['我', 'O'], ['们', 'O'], ['变', 'O'], ['而', 'O'], ['以', 'O'], ['书', 'O'], ['会', 'O'], ['友', 'O'], ['，', 'O'], ['以', 'O'], ['书', 'O'], ['结', 'O'], ['缘', 'O'], ['，', 'O'], ['把', 'O'], ['欧', 'S-LOC'], ['美', 'S-LOC'], ['、', 'O'], ['港', 'S-LOC'], ['台', 'S-LOC'], ['流', 'O'], ['行', 'O'], ['的', 'O'], ['食', 'O'], ['品', 'O'], ['类', 'O'], ['图', 'O'], ['谱', 'O'], ['、', 'O'], ['画', 'O'], ['册', 'O'], ['、', 'O'], ['工', 'O'], ['具', 'O'], ['书', 'O'], ['汇', 'O'], ['集', 'O'], ['一', 'O'], ['堂', 'O'], ['。', 'O']]
train_sentences = load_sentences(train_file, lower, zeros)
dev_sentences = load_sentences(dev_file, lower, zeros)
test_sentences = load_sentences(test_file, lower, zeros)

# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, tag_schema)
update_tag_scheme(test_sentences, tag_schema)

dico_chars_train = char_mapping(train_sentences, lower)[0]
dico_chars, word2index, index2word = augment_with_pretrained(
    dico_chars_train.copy(),
    emb_file,
    list(itertools.chain.from_iterable([[w[0] for w in s] for s in test_sentences]))
)

_t, tag2index, index2tag = tag_mapping(train_sentences)

# 准备数据，将word、tag等转换为索引
# [['海', 'O'], ['钓', 'O'], ['比', 'O'], ['赛', 'O'], ['地', 'O'], ['点', 'O'], ['在', 'O'], ['厦', 'B-LOC'], ['门', 'E-LOC'], ['与', 'O'], ['金', 'B-LOC'], ['门', 'E-LOC'], ['之', 'O'], ['间', 'O'], ['的', 'O'], ['海', 'O'], ['域', 'O'], ['。', 'O']]
# [['海', '钓', '比', '赛', '地', '点', '在', '厦', '门', '与', '金', '门', '之', '间', '的', '海', '域', '。'], [235, 1574, 153, 152, 30, 236, 8, 1500, 238, 89, 182, 238, 112, 198, 3, 235, 658, 4], [1, 3, 1, 3, 1, 3, 0, 1, 3, 0, 1, 3, 1, 3, 0, 1, 3, 0], [0, 0, 0, 0, 0, 0, 0, 2, 6, 0, 2, 6, 0, 0, 0, 0, 0, 0]]
train_data = prepare_dataset(train_sentences, word2index, tag2index, lower)
# [['在', '这', '里', '恕', '弟', '不', '恭', '之', '罪', '，', '敢', '在', '尊', '前', '一', '诤', '：', '前', '人', '论', '书', '，', '每', '曰', '“', '字', '字', '有', '来', '历', '，', '笔', '笔', '有', '出', '处', '”', '，', '细', '读', '公', '字', '，', '何', '尝', '跳', '出', '前', '人', '藩', '篱', '，', '自', '隶', '变', '而', '后', '，', '直', '至', '明', '季', '，', '兄', '有', '何', '新', '出', '？'],
#  [8, 22, 144, 2842, 1327, 17, 2210, 112, 980, 2, 1379, 8, 1179, 92, 6, 3384, 149, 92, 12, 191, 229, 2, 375, 3941, 24, 619, 619, 14, 43, 350, 2, 856, 856, 14, 32, 310, 23, 2, 917, 669, 71, 619, 2, 550, 1811, 1311, 32, 92, 12, 1, 3340, 2, 85, 2807, 353, 109, 83, 2, 479, 309, 168, 912, 2, 1569, 14, 550, 61, 32, 660],
#  [0, 1, 3, 1, 3, 1, 2, 2, 3, 0, 0, 0, 1, 3, 1, 3, 0, 1, 3, 1, 3, 0, 0, 0, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 0, 1, 3, 1, 3, 0, 1, 3, 1, 3, 1, 3, 1, 3, 0, 1, 2, 3, 1, 3, 0, 1, 3, 1, 3, 0, 1, 3, 1, 2, 3, 0],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
#  -word         - word indexes       - word char indexes         - tag indexes
dev_data = prepare_dataset(dev_sentences, word2index, tag2index, lower)
test_data = prepare_dataset(test_sentences, word2index, tag2index, lower)
print("train / dev / test的句子数分别为：%i / %i / %i ." % (len(train_data), 0, len(test_data)))

# 后向填充了<pad>对齐，然后整体随机后yield单个返回成迭代器
# 以batch_size进行sequence_max_len计算，每个bantch的最大序列长度极有可能不同
train_manager = BatchManager(train_data, batch_size)
dev_manager = BatchManager(dev_data, 100)
test_manager = BatchManager(test_data, 100)


##################################### 模型参数 #####################################
num_words = len(word2index)
num_tags = len(tag2index)
num_segs = 4

initializer = initializers.xavier_initializer()

# 模型参数
sentences_inputs = tf.placeholder(dtype=tf.int32,shape=[None, None],name="SentencesInputs")
seg_inputs = tf.placeholder(dtype=tf.int32,shape=[None, None],name="SegInputs")
targets = tf.placeholder(dtype=tf.int32,shape=[None, None],name="Targets")
dropout = tf.placeholder(dtype=tf.float32, name="Dropout")

used = tf.sign(tf.abs(sentences_inputs))
length = tf.reduce_sum(used, reduction_indices=1)   # 句子的长度
lengths = tf.cast(length, tf.int32)
batch_size = tf.shape(sentences_inputs)[0]   # N
num_steps = tf.shape(sentences_inputs)[-1]   # T

# seg特征是利用字在词(句子分词后)中所处的位置，对字的特征进行一个补充，
# 比如一个字可能有：这个字就是一个词0，这个字是某个词语的开始1，这个字是某个词语的中间2，这个字是某个词语的结束3四种状态。
# 每个状态会对应一个向量，其维度是seg_dim。concat就是将一个字向量和其对应的seg向量进行拼接，
# 比如100维的字向量和100维的seg向量，最后表示每个字的向量就是200维的。
embed = []
with tf.variable_scope("char_embedding"), tf.device('/cpu:0'):
    # NxD
    word_embedding = tf.get_variable(name="char_embedding", shape=[num_words, word_dim], initializer=initializer)
    # NxT -> NxTxD
    embed.append(tf.nn.embedding_lookup(word_embedding, sentences_inputs))
    if seg_dim != 0:
        with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
            # 4x20
            seg_embedding = tf.get_variable(name="seg_embedding", shape=[num_segs, seg_dim], initializer=initializer)
            # NxT -> NxTx20
            embed.append(tf.nn.embedding_lookup(seg_embedding, seg_inputs))
    embedding = tf.concat(embed, axis=-1)

# 将数据喂给lstm layer前先进行dropout
lstm_inputs = tf.nn.dropout(embedding, dropout)

# bi-lstm layer
# lstm_inputs: NxTxH -> NxTx2H
with tf.variable_scope("char_BiLSTM"):
    lstm_cell = {}
    for direction in ["forward", "backward"]:
        with tf.variable_scope(direction):
            lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                lstm_dim,
                use_peepholes=True,
                initializer=initializer,
                state_is_tuple=True)
    outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
        lstm_cell["forward"],
        lstm_cell["backward"],
        lstm_inputs,
        dtype=tf.float32,
        sequence_length=lengths)
    lstm_outputs = tf.concat(outputs, axis=2)

# logits for tags
# 在lstm layer和logits中的隐藏层
# lstm_outputs: NxTx2H  -> logits: NxTxV
with tf.variable_scope("project"):
    with tf.variable_scope("hidden"):
        W = tf.get_variable("W", shape=[lstm_dim * 2, lstm_dim],
                            dtype=tf.float32, initializer=initializer)

        b = tf.get_variable("b", shape=[lstm_dim], dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        # NxTx2H -> N*Tx2H
        output = tf.reshape(lstm_outputs, shape=[-1, lstm_dim * 2])
        # N*Tx2H 2HxH -> N*TxH
        hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

    # project to score of tags
    with tf.variable_scope("logits"):
        W = tf.get_variable("W", shape=[lstm_dim, num_tags],dtype=tf.float32, initializer=initializer)
        b = tf.get_variable("b", shape=[num_tags], dtype=tf.float32, initializer=tf.zeros_initializer())
        # N*TxH HxV -> N*TxV   V为tags
        pred = tf.nn.xw_plus_b(hidden, W, b)
    # N*TxV -> NxTxV
    logits = tf.reshape(pred, [-1, num_steps, num_tags])

# loss of the model
# calculate crf loss
# :param logits: [1, num_steps, num_tags]
# :return: scalar loss
# 每个输出后面多个很小的值，V->V+1
# 每个序列前面多个很小的值，T->1+T
# logits NxTxV -> NxTxV+1 -> Nx1+TxV+1   一个词对应 V+1 从上往下看
with tf.variable_scope("crf_loss"):
    small = -1000.0
    # pad logits for crf loss
    # NxTx1
    pad_logits = tf.cast(small * tf.ones([batch_size, num_steps, 1]), tf.float32)
    # NxTxV + NxTx1 -> NxTx(V+1)    加入begin词对应的logit
    logits_ = tf.concat([logits, pad_logits], axis=-1)
    # Nx1xV + Nx1x1 -> Nx1x(V+1)    加了个0 多了个tag 序列长度也+1了
    start_logits = tf.concat([small * tf.ones(shape=[batch_size, 1, num_tags]), tf.zeros(shape=[batch_size, 1, 1])], axis=-1)
    # Nx1x(V+1) + NxTx(V+1) -> Nx(1+T)x(V+1)    加入序列的begin词
    logits_all = tf.concat([start_logits, logits_], axis=1)
    # Nx1 + NxT -> Nx(1+T) 加入begin词对应的target
    targets_all = tf.concat([tf.cast(num_tags * tf.ones([batch_size, 1]), tf.int32), targets], axis=1)
    # (V+1)x(V+1)
    trans = tf.get_variable(
        "transitions",
        shape=[num_tags + 1, num_tags + 1],
        initializer=initializer)
    # logits: Nx(1+T)x(V+1)  tag_indices: Nx(1+V)  transition_params: (V+1)x(V+1)
    log_likelihood, trans = crf_log_likelihood(
        inputs=logits_all,
        tag_indices=targets_all,
        transition_params=trans,
        sequence_lengths=lengths + 1)

    # VxV
    # trans = tf.get_variable(
    #     "transitions",
    #     shape=[num_tags, num_tags],
    #     initializer=initializer)
    # # logits: NxTxV  tag_indices: NxV  transition_params: VxV
    # log_likelihood, trans = crf_log_likelihood(
    #     inputs=logits,
    #     tag_indices=targets,
    #     transition_params=trans,
    #     sequence_lengths=lengths)
    # 负的极大似然
    loss = tf.reduce_mean(-log_likelihood)

with tf.variable_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(lr)
    grads_vars = optimizer.compute_gradients(loss)  # (gradient, variable)
    # -clip <= grad <= clip 小于的改为-clip，大于的改为clip
    capped_grads_vars = [[tf.clip_by_value(g, - clip, clip), v] for g, v in grads_vars]
train_op = optimizer.apply_gradients(capped_grads_vars)
# optimizer = tf.train.AdamOptimizer(lr)
# train_op = optimizer.minimize(loss)

####################################### Train #######################################
tf_config = tf.ConfigProto()
# 动态申请显存
tf_config.gpu_options.allow_growth = True
# 1044
steps_per_epoch = train_manager.len_data
sess = tf.Session(config=tf_config)
# with tf.Session(config=tf_config) as sess:
print("用新的参数创建模型.")
sess.run(tf.global_variables_initializer())
if pre_emb:
    emb_weights = sess.run(word_embedding.read_value())
    # 4412 x 100
    emb_weights = load_word2vec(emb_file,index2word, word_dim, emb_weights)
    sess.run(word_embedding.assign(emb_weights))
    print("加载预训练Embedding.")


print("开始训练")
cost = []
best_dev_f1 = 0.0
best_test_f1 = 0.0
for i in range(max_epoch):
    batch_num = 0
    for batch in train_manager.iter_batch(shuffle=True):
        _, words, segs, tags = batch
        feed_dict = {sentences_inputs: np.asarray(words), seg_inputs: np.asarray(segs), targets: np.asarray(tags), dropout: p_dropout}
        # lengths_test, logit_test, targets_test = sess.run(fetches=[lengths, logits_all, targets_all], feed_dict=feed_dict)
        batch_loss, _ = sess.run(fetches=[loss, train_op], feed_dict=feed_dict)

        cost.append(batch_loss)
        if batch_num % 10 == 0:
            print("epoch:%d step:%d/%d, NER loss: %9.6f" % (i, batch_num, steps_per_epoch, batch_loss))
        batch_num = batch_num + 1
    # 评价dev
    results = []
    trans_ = trans.eval(session=sess)
    for batch_ in dev_manager.iter_batch():
        sentences, chars_, segs_, tags_ = batch_
        feed_dict_dev = {sentences_inputs: np.asarray(chars_), seg_inputs: np.asarray(segs_), targets: np.asarray(tags_), dropout: p_dropout}
        # 100, 最长的是17，最短的是7， scores: 100x17x13
        lengths_dev, scores = sess.run([lengths, logits_all], feed_dict_dev)

        batch_paths = []
        for score, length in zip(scores, lengths_dev):
            score = score[:length+1]  # 左闭右开，  length+1
            path, _ = viterbi_decode(score, trans_)
            # tag和logits前向填充后
            batch_paths.append(path[1:])

        result = []
        for i in range(len(sentences)):
            sentence = sentences[i][:lengths_dev[i]]
            gold = iobes_iob([index2tag[int(x)] for x in tags_[i][:lengths_dev[i]]])
            pred = iobes_iob([index2tag[int(x)] for x in batch_paths[i][:lengths_dev[i]]])
            for word, gold_, pred_ in zip(sentence, gold, pred):
                result.append(" ".join([word, gold_, pred_]))
            results.append(result)

    ner_results = results
    eval_lines = test_ner(ner_results, result_path)
    for line in eval_lines:
        print(line)
    f1 = float(eval_lines[1].strip().split()[-1])
    if f1 > best_dev_f1:
        best_dev_f1 = f1
    dev_best = best_dev_f1
