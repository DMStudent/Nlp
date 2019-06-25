import tensorflow as tf
import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# sess = tf.Session(config=config)
# logits = [11,13,1.2,2.3,3,5]
# reshaped_logits = tf.reshape(logits, [-1, 2])
# labels = [[1,0],[0,1],[1,0]]
# mean_labels = tf.reduce_mean(labels, axis=1)
# print(mean_labels)
# sess.run(tf.Print(mean_labels,[mean_labels],summarize=10))
print(np.argmax([1,2,3,1,2,10,2]))