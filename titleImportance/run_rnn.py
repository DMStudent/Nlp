# -*- coding: utf-8 -*-
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time:
#!/bin/bash

from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from rnn_model import TRNNConfig, TextRNN
from preprocess_mp import read_vocab, process_file, read_example, read_example_test
from tensorflow.core.protobuf import saver_pb2



def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def train():
    print('Configuring RNN model...')
    config = TRNNConfig()
    config.dropout_keep_prob = 1.0
    start_time = time.time()
    # config.batch_size = 10
    total_batch = 0  # 总批次
    best_mse_val = 99999999  # 最佳验证集准确率
    best_loss_val = 99999999  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 5000  # 如果超过1000轮未提升，提前结束训练
    count = 0
    tensorboard_dir = config.tensorboard_dir

    # 配置GPU内存分配方式
    tfconfig = tf.ConfigProto(log_device_placement=True)
    tfconfig.gpu_options.allow_growth = True
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.6


    with tf.Graph().as_default(), tf.Session(config=tfconfig) as sess:
        train_dir_list = os.listdir(config.train_dir_tf)
        train_dir_list = [os.path.join(config.train_dir_tf, i) for i in train_dir_list]
        queueTrain = tf.train.string_input_producer(train_dir_list, num_epochs=config.num_epochs)
        title_len, title, label, frame_weight = read_example(queueTrain)

        title_len_batch, title_batch, label_batch, frame_weight_batch = tf.train.batch(
            [title_len, title, label, frame_weight], batch_size=config.batch_size,
            capacity=100000,
            num_threads=1)

        with tf.variable_scope("model", initializer=tf.random_uniform_initializer(-1 * 1, 1)):
            model = TextRNN(config=config, input_x_len=title_len_batch, input_x=title_batch, input_y=label_batch,
                            frame_weight=frame_weight_batch)

        tf.summary.scalar("loss", model.loss)
        tf.summary.scalar("mse", model.mse)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

        fetches = [model.loss, model.mse]
        feed_dict = {}
        # init
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        # 配置 Saver
        saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)
        if not config.retraining:
            saver.restore(sess=sess, save_path=config.modelPath)  # 读取保存的模型
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)

        try:
            while not coord.should_stop():
                # Run training steps or whatever
                # titles, labels = sess.run([title_batch, label_batch])
                if total_batch % config.save_per_batch == 0:
                    # 每多少轮次将训练结果写入tensorboard scalar
                    s = sess.run(merged_summary, feed_dict)
                    writer.add_summary(s, total_batch)

                if total_batch % config.print_per_batch == 0:
                    # 每多少轮次输出在训练集和验证集上的性能
                    loss_val, mse_val = sess.run(fetches, feed_dict)

                    if mse_val < best_mse_val or loss_val < best_loss_val:
                        # 保存最好结果
                        best_mse_val = mse_val
                        best_loss_val = loss_val
                        last_improved = total_batch
                        improved_str = '*'
                        # saver.save(sess=sess, save_path=config.save_path)
                        if total_batch % config.save_per_batch == 0:
                            saver.save(sess, config.save_path + '_%03d' % (total_batch/config.save_per_batch), write_meta_graph=False)
                    else:
                        improved_str = ''

                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Val Loss: {1:>6.5}, Mse: {2:>6.5}, Time: {3} {4}'
                    print(msg.format(total_batch, loss_val, mse_val, time_dif, improved_str))
                    # print(embedding_inputs)

                sess.run(model.optim, feed_dict)
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    coord.should_stop()
                    break  # 跳出循环

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        coord.request_stop()
        coord.join(threads)

def test():
    print('Configuring RNN model...')
    config = TRNNConfig()
    config.dropout_keep_prob = 1.0
    config.num_epochs = 1
    start_time = time.time()
    config.batch_size = 10
    count = 0

    # 配置GPU内存分配方式
    tfconfig = tf.ConfigProto(log_device_placement=True)
    tfconfig.gpu_options.allow_growth = True
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.6

    fw = file(config.test_dir_output, "w")
    with tf.Graph().as_default(), tf.Session(config=tfconfig) as sess:
        test_dir_list = os.listdir(config.test_dir_tf)
        test_dir_list = [os.path.join(config.test_dir_tf, i) for i in test_dir_list]
        queueTest = tf.train.string_input_producer(test_dir_list, num_epochs=config.num_epochs)
        text, title_len, title, label, frame_weight = read_example_test(queueTest)

        text_batch, title_len_batch, title_batch, label_batch, frame_weight_batch = tf.train.batch([text, title_len, title, label, frame_weight], batch_size=config.batch_size, capacity=50000,
                                                              num_threads=1)
        with tf.variable_scope("model", initializer=tf.random_uniform_initializer(-1 * 1, 1)):
            model = TextRNN(config=config, input_x_len=title_len_batch, input_x=title_batch, input_y=label_batch, frame_weight=frame_weight_batch)

        fetches = [text_batch, model.input_x_len, model.y_pred, model.input_y]
        feed_dict = {}
        # init
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        # 配置 Saver
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=config.modelPath)  # 读取保存的模型
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)

        try:
            while not coord.should_stop():
                texts, x_len, y_pred, y_test = sess.run(fetches, feed_dict=feed_dict)
                texts = "".join(texts.values).split("\n")
                for i in range(len(texts) - 1):
                    score = [str(int(j*100)) for j in y_test[i]][:x_len[i][0]]
                    y_test_i = " ".join(score)
                    score = [str(int(j * 100)) for j in y_pred[i]][:x_len[i][0]]
                    y_pred_i = " ".join(score)
                    fw.write(
                        texts[i] + "\ttarget:\t" + y_test_i + "\tpredict\t" + y_pred_i + "\n")
                count = count + 1
                if count % 10000 == 0:
                    print(count)


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        coord.request_stop()
        coord.join(threads)
    fw.close()





if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_rnn.py [train / test]""")

    if sys.argv[1] == 'train':
        train()
    else:
        test()
