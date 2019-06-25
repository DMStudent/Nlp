#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import numpy as np
class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 100      # 词向量维度
    seq_length = 30        # 序列长度
    vocab_size = 450000       # 词汇表达小
    num_classes = 5

    num_layers= 2           # 隐藏层层数
    hidden_dim = 100        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 5000         # 每批训练大小
    num_epochs = 100          # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 1000      # 每多少轮存入tensorboard
    retraining = False

    vocab_dir = "embedding/word2id.txt"
    embedding_dir = None
    # embedding_dir = "embedding/word_embedding.txt"
    base_dir = 'data'
    train_dir = 'data/train-part'
    test_dir = 'data/test-part'
    train_dir_tf = 'data/train-part-tf'
    test_dir_tf = 'data/test-part-tf'
    test_dir_output = 'data/test-output'
    save_dir = 'checkpoints'
    save_path = os.path.join(save_dir, 'title_importance')  # 最佳验证结果保存路径
    modelPath = "./checkpoints/title_importance_008"
    # save_path_old = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

    tensorboard_dir = 'tensorboard'


class TextRNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config, input_x_len, input_x, input_y, frame_weight):
        self.config = config

        # 三个待输入的数据
        # 三个待输入的数据
        self.input_x_len = input_x_len
        self.input_x = input_x
        self.input_y = input_y
        self.frame_weight = frame_weight
        self.keep_prob = config.dropout_keep_prob
        if config.embedding_dir != None:
            self.embedding_matrix = self.getEmbedding(config.embedding_dir)
        else:
            self.embedding_matrix = None

        self.rnn()

    def getEmbedding(self, infile_path):
        print "Reading word embedding..."
        with open(infile_path, "r") as infile:
            row_index = 0
            emb_matrix = np.zeros((self.config.vocab_size, self.config.embedding_dim))
            for row in infile:
                items = row.strip().decode('utf-8').split()
                if len(items) == 100:
                    emb_vec = [float(val) for val in items]
                    emb_matrix[row_index] = emb_vec
                    row_index = row_index + 1

        return emb_matrix
    def rnn(self):
        """rnn模型"""

        def lstm_cell():   # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout(): # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 词向量映射
        with tf.device('/cpu:0'):
            # if isinstance(self.embedding_matrix, np.ndarray) or self.embedding_matrix != None:
            #     embedding = tf.Variable(self.embedding_matrix, trainable=True, name="emb", dtype=tf.float32)
            # else:
            #     embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])

            self.embedding = tf.get_variable("embedding", [self.config.vocab_size, self.config.embedding_dim],
                                                dtype=tf.float32, trainable=True)
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope("rnn"):
            # 多层rnn网络
            self.title_length = tf.to_int32(tf.squeeze(self.input_x_len))
            fw_cell = [dropout() for _ in range(self.config.num_layers)]
            fw_cell = tf.contrib.rnn.MultiRNNCell(fw_cell, state_is_tuple=True)

            bw_cell = [dropout() for _ in range(self.config.num_layers)]
            bw_cell = tf.contrib.rnn.MultiRNNCell(bw_cell, state_is_tuple=True)

            states, tuple_final = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                                  inputs=embedding_inputs,
                                                                  dtype=tf.float32,
                                                                  sequence_length=self.title_length,
                                                                  parallel_iterations=256, scope='BILSTM')
            fw_cell_final = states[0]
            bw_cell_final = states[1]

            self.last = tf.concat([fw_cell_final, bw_cell_final], 2)
            self.last = tf.reshape(self.last, [-1, self.config.embedding_dim * 2])


            # _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            # last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            self.fc = tf.layers.dense(self.last, self.config.hidden_dim, name='fc1')
            self.fc = tf.contrib.layers.dropout(self.fc, self.keep_prob)
            self.fc = tf.nn.relu(self.fc)

            # 预测
            self.logits = tf.squeeze(tf.layers.dense(self.fc, 1, name='fc2'))
            # self.y_pred = tf.squeeze(self.logits)

            self.y_pred_re = tf.nn.sigmoid(self.logits)

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            tv = tf.trainable_variables()  # 得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
            regularization_cost = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])  # 0.001是lambda超参数

            self.input_y_re = tf.reshape(self.input_y, [-1])
            self.frame_weight_re = tf.reshape(self.frame_weight, [-1])
            self.y_pred = tf.reshape(self.y_pred_re, [-1, self.config.seq_length])
            self.c = tf.square(self.y_pred_re - self.input_y_re) * self.frame_weight_re

            self.mse = tf.reduce_sum(self.c)/2
            self.loss = self.mse + regularization_cost
            # 优化器
            self.optim = tf.contrib.opt.LazyAdamOptimizer(self.config.learning_rate).minimize(self.loss)
            # self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        # with tf.name_scope("accuracy"):
        #     # 准确率
        #     correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
        #     self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
