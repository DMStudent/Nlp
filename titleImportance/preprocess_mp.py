# -*- coding: utf-8 -*-
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time: 
#!/bin/bash
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from rnn_model import TRNNConfig
import multiprocessing as mp
from multiprocessing import Process
import time

import numpy as np



# float
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
# int64
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
# byte
def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def convert_to_TFRecords_withText(dataset, name):
    """Convert mnist dataset to TFRecords"""
    texts, titles, labels = dataset
    n_examples = len(titles)

    filename = name + ".tfrecords"
    print"Writing:"+filename

    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(n_examples):
            if index%100000 == 0:
                print index
            text = "".join(texts[index]).encode("utf-8")+"\n"
            title = titles[index].tolist()
            label = labels[index]
            label = [int(x) for x in label]
            example = tf.train.Example(features=tf.train.Features(
                feature={"text": _byte_feature(text),
                         "title": _int64_feature(title),
                         "label": _int64_feature(label)}))
            writer.write(example.SerializeToString())
    print "done."


def convert_to_TFRecords(dataset, name):
    """Convert mnist dataset to TFRecords"""
    x_len, titles, labels, frame_weights = dataset
    n_examples = len(titles)

    filename = name + ".tfrecords"
    print"Writing:"+filename

    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(n_examples):
            if index%100000 == 0:
                print index
            l = x_len[index]
            title = titles[index]
            label = labels[index]
            frame_weight = frame_weights[index]
            example = tf.train.Example(features=tf.train.Features(
                feature={"title_len": _int64_feature(l),
                         "title": _int64_feature(title),
                         "label": _float_feature(label),
                         "frame_weight": _float_feature(frame_weight)
                         }))
            writer.write(example.SerializeToString())
    print "done."


def convert_to_TFRecords_test(dataset, name):
    """Convert mnist dataset to TFRecords"""
    texts, x_len, titles, labels, frame_weights = dataset
    n_examples = len(titles)

    filename = name + ".tfrecords"
    print"Writing:"+filename

    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(n_examples):
            if index%10000 == 0:
                print index
            l = x_len[index]
            text = texts[index].encode("utf-8")+"\n"
            title = titles[index]
            label = labels[index]
            frame_weight = frame_weights[index]
            example = tf.train.Example(features=tf.train.Features(
                feature={"text": _byte_feature(text),
                         "title_len": _int64_feature(l),
                         "title": _int64_feature(title),
                         "label": _float_feature(label),
                         "frame_weight": _float_feature(frame_weight)
                         }))
            writer.write(example.SerializeToString())
    print "done."


def read_example_test(filename_queue):
    config = TRNNConfig()
    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)
    features = tf.parse_single_example(value, features={"text": tf.VarLenFeature(tf.string),
                                                        "title_len": tf.FixedLenFeature([1], tf.int64),
                                                        "title": tf.FixedLenFeature([config.seq_length], tf.int64),
                                                        "label": tf.FixedLenFeature([config.seq_length], tf.float32),
                                                        "frame_weight": tf.FixedLenFeature([config.seq_length], tf.float32),
                                                        })

    text = features["text"]
    title_len = tf.cast(features["title_len"], tf.int32)
    title = tf.cast(features["title"], tf.int32)
    label = tf.cast(features["label"], tf.float32)
    frame_weight = tf.cast(features["frame_weight"], tf.float32)
    return text, title_len, title, label, frame_weight

def read_example(filename_queue):
    """Read one example from filename_queue"""
    config = TRNNConfig()
    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)
    features = tf.parse_single_example(value, features={"title_len": tf.FixedLenFeature([1], tf.int64),
                                                        "title": tf.FixedLenFeature([config.seq_length], tf.int64),
                                                        "label": tf.FixedLenFeature([config.seq_length], tf.float32),
                                                        "frame_weight": tf.FixedLenFeature([config.seq_length], tf.float32),
                                                        })
    title_len = tf.cast(features["title_len"], tf.int32)
    title = tf.cast(features["title"], tf.int32)
    label = tf.cast(features["label"], tf.float32)
    frame_weight = tf.cast(features["frame_weight"], tf.float32)
    return title_len, title, label, frame_weight

def native_content(content, is_py3=False):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content

def open_file(filename, mode='r', is_py3=False):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)

def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                _, _, x, y, _ = line.strip().split('\t')
                if x and y:
                    contents.append(native_content(x))
                    labels.append(native_content(y))
            except:
                pass
    return contents, labels

def padding(sample, seq_max_len):
    lens = []
    for i in range(len(sample)):
        lens.append([min(len(sample[i]),seq_max_len)])
        if len(sample[i]) < seq_max_len:
            sample[i] += [0 for _ in range(seq_max_len - len(sample[i]))]
        else:
            sample[i] = sample[i][0:seq_max_len]
    return lens, sample

def process_file_test(filename, word_to_id, seq_length, is_padding=True):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id, frame_weight = [], [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] if x in word_to_id else 0 for x in contents[i].split()])
        label = [float(x) / 100 for x in labels[i].split()]
        label_id.append(label)
        frame_weight.append([2.0 if (x > 0.8 or x < 0.2) else 1.0 for x in label])
    if is_padding:
        lenX, X = np.array(padding(data_id, seq_length))
        _, Y = np.array(padding(label_id, seq_length))
        _, W = np.array(padding(frame_weight, seq_length))
    else:
        X = np.array(data_id)
        Y = np.array(label_id)

    return contents, lenX, X, Y, W

def process_file(filename, word_to_id, seq_length, is_padding=True):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id, frame_weight = [], [], []

    for i in range(len(contents)):
        data_id.append([word_to_id[x] if x in word_to_id else 0 for x in contents[i].split()])
        label = [float(x) / 100 for x in labels[i].split()]
        label_id.append(label)
        frame_weight.append([2.0 if(x > 0.9 or x < 0.2) else 1.0 for x in label])

    if is_padding:
        lenX, X = np.array(padding(data_id, seq_length))
        _, Y = np.array(padding(label_id, seq_length))
        _, W = np.array(padding(frame_weight, seq_length))
    else:
        X = np.array(data_id)
        Y = np.array(label_id)

    return lenX, X, Y, W

def process_file_onecore(train_dir, train_dir_tf, word_to_id, seq_length):
    x_len, x_train, y_train, frame_weight = process_file(train_dir, word_to_id, seq_length)
    convert_to_TFRecords([x_len, x_train, y_train, frame_weight], train_dir_tf)

def process_file_onecore_test(train_dir, train_dir_tf, word_to_id, seq_length):
    text, x_len, x_train, y_train, frame_weight = process_file_test(train_dir, word_to_id, seq_length)
    convert_to_TFRecords_test([text, x_len, x_train, y_train, frame_weight], train_dir_tf)

def process_file_multicore(dir, dir_out, word_to_id, seq_length, isTrain=True):
    start = time.time()
    files = os.listdir(dir)

    for file in files:
        if isTrain:
            p = Process(target=process_file_onecore, args=(dir+"/"+file, dir_out + "/" + file, word_to_id, seq_length))
        else:
            p = Process(target=process_file_onecore_test, args=(dir + "/" + file, dir_out + "/" + file, word_to_id, seq_length))
        p.start()

    print("The number of CPU is:" + str(mp.cpu_count()))
    for p in mp.active_children():
        p.join()
        print("child p.name: " + p.name + "\tp.id: " + str(p.pid))


    end = time.time()
    print('processes take %s seconds' % (end - start))



def splitEmbedding(infile_path="/search/odin/data/wangyuan/pycharmProjects/MatchZoo/data/wenti/vector.synonyms.dec.utf8"):
    word2id = "embedding/word2id.txt"
    word_embedding = "embedding/word_embedding.txt"

    fw1 = file(word2id, 'w')
    fw2 = file(word_embedding, 'w')
    row_index = 0

    with open(infile_path, "r") as infile:
        for row in infile:
            row = row.strip()
            items = row.split()
            word = items[0]
            if len(word)>0 and len(items) == 101:
                fw1.write(word+"\t"+str(row_index)+"\n")
                fw2.write(" ".join(items[1:])+"\n")
            row_index = row_index + 1
            if row_index%10000 == 0:
                print "processing:"+str(row_index)
    fw1.close()
    fw2.close()

def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    word_to_id = {}
    id_to_word = {}
    with open(vocab_dir, 'r') as fp:
        for line in fp.readlines():
            [word, wid] = line.decode('utf-8').split("\t")
            wid = int(wid)
            word_to_id[word] = wid
            id_to_word[wid] = word

    return word_to_id, id_to_word

if __name__ == "__main__":
    # train
    # 数据格式转换
    config = TRNNConfig()
    if not os.path.exists(config.embedding_dir) or not os.path.exists(config.vocab_dir):  # 如果不存在词汇表，重建
        splitEmbedding()
    word_to_id, id_to_word = read_vocab(config.vocab_dir)
    process_file_multicore(config.train_dir, config.train_dir_tf, word_to_id, config.seq_length)



    # 读取测试
    train_dir_list = os.listdir(config.train_dir_tf)
    train_dir_list = [os.path.join(config.train_dir_tf, i) for i in train_dir_list]
    queueTrain = tf.train.string_input_producer(train_dir_list, num_epochs=config.num_epochs)
    title_len, title, label, frame_weight = read_example(queueTrain)

    title_len_batch, text_batch, title_batch, frame_weight_batch = tf.train.batch([title_len, title, label, frame_weight], batch_size=20, capacity=500,
                                                    num_threads=1)
    count = 0
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                # Run training steps or whatever
                title_len, titles, labels, frame_weights = sess.run([title_len_batch, text_batch, title_batch, frame_weight_batch])
                print "--------------------------"
                print count
                count = count + 1
                # print titles
                # print labels
                print(titles.shape, labels.shape)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        coord.request_stop()
        coord.join(threads)


    # test
    # 数据格式转换
    config = TRNNConfig()
    # if not os.path.exists(config.embedding_dir) or not os.path.exists(config.vocab_dir):  # 如果不存在词汇表，重建
    #     splitEmbedding()
    word_to_id, id_to_word = read_vocab(config.vocab_dir)
    process_file_multicore(config.test_dir, config.test_dir_tf, word_to_id, config.seq_length, isTrain=False)

    test_dir_list = os.listdir(config.test_dir_tf)
    test_dir_list = [os.path.join(config.test_dir_tf, i) for i in test_dir_list]
    queueTest = tf.train.string_input_producer(test_dir_list, num_epochs=config.num_epochs)
    text, title_len, title, label, frame_weight = read_example_test(queueTest)

    text_batch, title_len_batch, title_batch, label_batch, frame_weight_batch = tf.train.batch([text, title_len, title, label, frame_weight], batch_size=20, capacity=500,
                                                    num_threads=1)
    count = 0
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                # Run training steps or whatever
                texts, text_lens, titles, labels, frame_weights = sess.run([text_batch, title_len_batch, title_batch, label_batch, frame_weight_batch])
                print "--------------------------"
                print count
                count = count + 1
                print text_lens
                # print titles
                # print labels
                texts = "".join(texts.values)
                print texts
                print(titles.shape, labels.shape)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        coord.request_stop()
        coord.join(threads)




