# -*- coding: utf-8 -*-
# File : predict_lite.py
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time: 2019/4/17
#!/bin/bash

import tensorflow as tf
# import numpy as np
import sys
import tokenization

flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("vocab_file", "/search/wangyuan/bert/chinese_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string(
    "frozen_model_path", "/search/wangyuan/BertForMultipleChoice/result/saved_model/1555567255/transform_graph.pb",
    "frozen model path.")
flags.DEFINE_string(
    "model_path", "/search/wangyuan/BertForMultipleChoice/result/saved_model/1555567255",
    "original save_model path.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_integer(
    "max_seq_length", 30,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_integer("batch_size", 512, "Total batch size for training.")



class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example

class SentenceExample(object):
    """A single training/test example for the Sentence dataset."""
    def __init__(self,
                 swag_id,
                 context_sentence,
                 ending_0,
                 ending_1,
                 label = None):
        self.swag_id = swag_id
        self.context_sentence = context_sentence
        self.endings = [
            ending_0,
            ending_1,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "swag_id: {}".format(self.swag_id),
            "context_sentence: {}".format(self.context_sentence),
            "ending_0: {}".format(self.endings[0]),
            "ending_1: {}".format(self.endings[1]),
        ]

        if self.label is not None:
            l.append("label: {}".format(self.label))

        return ", ".join(l)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  context_tokens = tokenizer.tokenize(example.context_sentence)

  examples = []
  for ending_index, ending in enumerate(example.endings):
      # We create a copy of the context tokens in order to be
      # able to shrink it according to ending_tokens
      context_tokens_choice = context_tokens[:]
      ending_tokens = tokenizer.tokenize(ending)
      _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

      tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
      segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      padding = [0] * (max_seq_length - len(input_ids))
      input_ids += padding
      input_mask += padding
      segment_ids += padding

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      def create_int_feature(values):
          f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
          return f

      # feature = InputFeatures(
      #     input_ids=input_ids,
      #     input_mask=input_mask,
      #     segment_ids=segment_ids,
      #     label_id=example.label,
      #     is_real_example=True)
      features = {}
      features["input_ids"] = create_int_feature(input_ids)
      features["input_mask"] = create_int_feature(input_mask)
      features["segment_ids"] = create_int_feature(segment_ids)
      features["label_ids"] = create_int_feature([example.label])
      tf_example = tf.train.Example(features=tf.train.Features(feature=features))
      examples.append(tf_example.SerializeToString())


  return examples

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Reading example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.extend(feature)
  return features


def input_fn_builder(features, batch_size, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  num_examples = len(features)
  d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
  })

  d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
  return d



def predict():
    pb_file_path = FLAGS.model_path
    session_config = tf.ConfigProto(device_count={"CPU": 1}, inter_op_parallelism_threads=1,
                                    intra_op_parallelism_threads=1)
    predict_fn = tf.contrib.predictor.from_saved_model(pb_file_path, config=session_config)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    idx = 0
    label_list = ["0", "1"]
    lines = []
    urls = []
    anchor1 = ""
    anchor2 = ""
    flag = 0
    line_lst = []
    num = 0
    for line in open("/search/wangyuan/BertForMultipleChoice/data/sentence/part").readlines():
    # for line in sys.stdin:
        num += 1
        lines_lst = line.strip().split("\t")
        if len(lines_lst) < 3:
            continue
        lines_anchors = lines_lst[-1].split("###")[:-1]
        for anchor in lines_anchors:
            line_lst = [lines_lst[0], lines_lst[1], anchor]
            flag += 1
            if flag % 2 == 1:
                url1 = line_lst[0]
                title1 = line_lst[1]
                anchor1 = line_lst[2]
                continue
            if flag % 2 == 0:
                idx += 1
                url2 = line_lst[0]
                title2 = line_lst[1]
                anchor2 = line_lst[2]
                if url1 == url2:
                    lines.append([title1, anchor1, anchor2])
                    urls.append(url1)
                    flag = 0
                else:
                    lines.append([title1, anchor1, "anchor"])
                    urls.append(url1)
                    url1 = url2
                    title1 = title2
                    anchor1 = anchor2
                    flag = 1

            if idx % FLAGS.batch_size == 0:
                predict_examples = [
                    SentenceExample(
                        swag_id=i,
                        context_sentence=line[0].strip(),
                        ending_0=line[1].strip(),
                        ending_1=line[2].strip(),
                        label=0
                    ) for (i, line) in enumerate(lines)  # we skip the line with the column names
                ]

                examples = convert_examples_to_features(predict_examples, label_list, FLAGS.max_seq_length,
                                                        tokenizer)
                out_probabilities = predict_fn({'examples': examples})
                result = out_probabilities['probabilities']
                for (i, prediction) in enumerate(result):
                    if lines[i][2] == "anchor":
                        output_line = urls[i] + "\t" \
                                      + predict_examples[i].context_sentence + "\t" \
                                      + predict_examples[i].endings[0] + "\t" \
                                      + str(prediction[0])
                        print(output_line)
                        continue
                    for j in range(2):
                        output_line = urls[i] + "\t" \
                                      + predict_examples[i].context_sentence + "\t" \
                                      + predict_examples[i].endings[j] + "\t" \
                                      + str(prediction[j])
                        print(output_line)

                lines = []
                urls = []

    if flag % 2 == 1 and line_lst:
        url1 = line_lst[0]
        title1 = line_lst[1]
        anchor1 = line_lst[2]
        urls.append(url1)
        lines.append([title1, anchor1, "anchor"])
    if len(lines) < 1:
        return

    predict_examples = [
        SentenceExample(
            swag_id=i,
            context_sentence=line[0].strip(),
            ending_0=line[1].strip(),
            ending_1=line[2].strip(),
            label=0
        ) for (i, line) in enumerate(lines)  # we skip the line with the column names
    ]

    examples = convert_examples_to_features(predict_examples, label_list, FLAGS.max_seq_length,
                                            tokenizer)
    out_probabilities = predict_fn({'examples': examples})
    result = out_probabilities['probabilities']
    for (i, prediction) in enumerate(result):
        if lines[i][2] == "anchor":
            output_line = urls[i] + "\t" \
                          + predict_examples[i].context_sentence + "\t" \
                          + predict_examples[i].endings[0] + "\t" \
                          + str(prediction[0])
            print(output_line)
            continue
        for j in range(2):
            output_line = urls[i] + "\t" \
                          + predict_examples[i].context_sentence + "\t" \
                          + predict_examples[i].endings[j] + "\t" \
                          + str(prediction[j])
            print(output_line)




def predict_freeze():
    pb_file_path=FLAGS.frozen_model_path
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    idx = 0
    label_list = ["0", "1"]
    lines = []
    urls = []
    anchor1 = ""
    anchor2 = ""
    flag = 0
    line_lst = []

    session_config = tf.ConfigProto(device_count={"CPU": 4}, use_per_session_threads=1, inter_op_parallelism_threads=1,
                                    intra_op_parallelism_threads=2)

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
        with tf.Session(config=session_config) as sess:
            gnet_output = sess.graph.get_tensor_by_name("final_layer/probabilities:0")
            for line in open("/search/wangyuan/BertForMultipleChoice/data/sentence/part").readlines():
            # for line in sys.stdin:
                lines_lst = line.strip().split("\t")
                if len(lines_lst) < 3:
                    continue
                lines_anchors = lines_lst[-1].split("###")[:-1]
                for anchor in lines_anchors:
                    line_lst = [lines_lst[0], lines_lst[1], anchor]
                    flag += 1
                    if flag % 2 == 1:
                        url1 = line_lst[0]
                        title1 = line_lst[1]
                        anchor1 = line_lst[2]
                        continue
                    if flag % 2 == 0:
                        idx += 1
                        url2 = line_lst[0]
                        title2 = line_lst[1]
                        anchor2 = line_lst[2]
                        if url1 == url2:
                            lines.append([title1, anchor1, anchor2])
                            urls.append(url1)
                            flag = 0
                        else:
                            lines.append([title1, anchor1, "anchor"])
                            urls.append(url1)
                            url1 = url2
                            title1 = title2
                            anchor1 = anchor2
                            flag = 1
                    if idx % FLAGS.batch_size == 0:
                        predict_examples = [
                            SentenceExample(
                                swag_id=i,
                                context_sentence=line[0].strip(),
                                ending_0=line[1].strip(),
                                ending_1=line[2].strip(),
                                label=0
                            ) for (i, line) in enumerate(lines)  # we skip the line with the column names
                        ]
                        examples = convert_examples_to_features(predict_examples, label_list, FLAGS.max_seq_length,
                                                                tokenizer)
                        result = sess.run(gnet_output, feed_dict={"input_example_tensor:0": examples})
                        # # out_probabilities = predict_fn({'examples': examples})
                        # print("label:", out_probabilities)
                        #
                        # result = out_probabilities['probabilities']
                        for (i, prediction) in enumerate(result):
                            if lines[i][2] == "anchor":
                                output_line = urls[i] + "\t" \
                                              + predict_examples[i].context_sentence + "\t" \
                                              + predict_examples[i].endings[0] + "\t" \
                                              + str(prediction[0])
                                print(output_line)
                                continue
                            for j in range(2):
                                output_line = urls[i] + "\t" \
                                              + predict_examples[i].context_sentence + "\t" \
                                              + predict_examples[i].endings[j] + "\t" \
                                              + str(prediction[j])
                                print(output_line)
                        lines = []
                        urls = []

            if flag % 2 == 1 and line_lst:
                url1 = line_lst[0]
                title1 = line_lst[1]
                anchor1 = line_lst[2]
                urls.append(url1)
                lines.append([title1, anchor1, "anchor"])
            if len(lines) < 1:
                return

            predict_examples = [
                SentenceExample(
                    swag_id=i,
                    context_sentence=line[0].strip(),
                    ending_0=line[1].strip(),
                    ending_1=line[2].strip(),
                    label=0
                ) for (i, line) in enumerate(lines)  # we skip the line with the column names
            ]
            examples = convert_examples_to_features(predict_examples, label_list, FLAGS.max_seq_length,
                                                    tokenizer)
            # result = sess.run(gnet_output, feed_dict={"input_example_tensor:0": examples})
            # # # out_probabilities = predict_fn({'examples': examples})
            # # print("label:", out_probabilities)
            # #
            # # result = out_probabilities['probabilities']
            # for (i, prediction) in enumerate(result):
            #     if lines[i][2] == "anchor":
            #         output_line = urls[i] + "\t" \
            #                       + predict_examples[i].context_sentence + "\t" \
            #                       + predict_examples[i].endings[0] + "\t" \
            #                       + str(prediction[0])
            #         print(output_line)
            #         continue
            #     for j in range(2):
            #         output_line = urls[i] + "\t" \
            #                       + predict_examples[i].context_sentence + "\t" \
            #                       + predict_examples[i].endings[j] + "\t" \
            #                       + str(prediction[j])
            #         print(output_line)



if __name__ == '__main__':
    # print("original modle:")
    # predict()
    # print("frozen modle:")
    predict_freeze()