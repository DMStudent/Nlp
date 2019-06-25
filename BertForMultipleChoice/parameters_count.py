# -*- coding: utf-8 -*-
# File : parameters_count.py
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time: 2019/6/3
#!/bin/bash


from tensorflow.python import pywrap_tensorflow
import os
import numpy as np

def count(checkpoint_path):

    # model_dir = "models_pretrained/"
    # checkpoint_path = os.path.join(model_dir, "model.ckpt-82798")
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    total_parameters = 0
    for key in var_to_shape_map:  # list the keys of the model
        # print(key)
        # print(reader.get_tensor(key))
        shape = np.shape(reader.get_tensor(key))  # get the shape of the tensor in the model
        shape = list(shape)
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim
        print(key+"\t"+str(variable_parameters))
        total_parameters += variable_parameters

    print(total_parameters)

if __name__ == '__main__':
    checkpoint_path = "/search/wangyuan/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
    count(checkpoint_path)