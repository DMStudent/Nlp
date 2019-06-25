# -*- coding: utf-8 -*-
# File : model_quantize.py
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time: 2019/4/16
#!/bin/bash


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["LD_LIBRARY_PATH"] = "/search/anaconda/envs/py36/lib"
import tensorflow as tf
from tensorflow.python.framework import graph_util



def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state("./output") #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "final_layer/probabilities"

    saver = tf.train.import_meta_graph(meta_graph_or_file=input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节



if __name__ == '__main__':
    modelpath = "./output/model.ckpt-0"
    model_dir_src = "./quantize/frozen.pb"
    freeze_graph(modelpath, model_dir_src)

