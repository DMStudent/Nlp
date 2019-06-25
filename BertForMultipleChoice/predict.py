# -*- coding: utf-8 -*-
# File : predict.py
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time: 2019/1/29
#!/bin/bash

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import csv
import modeling
import optimization
import tokenization
import tensorflow as tf
import numpy as np


flags = tf.flags
FLAGS = flags.FLAGS
## Required parameters
flags.DEFINE_string(
    "bert_config_file", "/search/wangyuan/bert/chinese_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")


flags.DEFINE_string("vocab_file", "/search/wangyuan/bert/chinese_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", "/search/wangyuan/BertForMultipleChoice/output/model.ckpt-115000",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 30,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 2000, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

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



class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


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


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

class MultipleChoiceProcessor(DataProcessor):
    """Processor for the Incest data set (GLUE version)."""

    def __init__(self):
        self.labels = ["0", "1"]

    def get_train_examples(self, data_dir):
        """See base class."""
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.csv")), "test")


    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(
                SentenceExample(
                    swag_id=guid,
                    context_sentence=line[0],
                    ending_0=line[1],
                    ending_1=line[2],
                    label=int(line[3])))


        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  context_tokens = tokenizer.tokenize(example.context_sentence)

  features = []
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

      feature = InputFeatures(
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          label_id=example.label,
          is_real_example=True)
      features.append(feature)

  return features



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


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [1, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [1], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    reshaped_logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1, num_labels])
    mean_labels = tf.reduce_mean(labels, axis=1)
    # probabilities = tf.nn.softmax(reshaped_logits, axis=-1)
    # log_probs = tf.log(probabilities)
    probabilities = tf.math.sigmoid(reshaped_logits)
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=reshaped_logits)
    # loss = tf.reduce_mean(loss)

    log_probs = tf.nn.log_softmax(reshaped_logits, axis=-1)

    one_hot_labels = tf.one_hot(mean_labels, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (per_example_loss, input_ids, one_hot_labels, log_probs, loss, mean_labels, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (per_example_loss, input_ids, one_hot_labels, log_probs, total_loss, labels, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
      logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=10)
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          training_hooks=[logging_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_metrics)
    else:
      output_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"probabilities": probabilities})
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
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

  def input_fn(params):
    """The actual input function."""
    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
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

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
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


def main(_):
  tf.logging.set_verbosity(tf.logging.FATAL)
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  processor = MultipleChoiceProcessor()
  label_list = processor.get_labels()
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  session_config = tf.ConfigProto(device_count={"CPU": 8}, inter_op_parallelism_threads=8,
                                  intra_op_parallelism_threads=8)
  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=None,
      num_warmup_steps=None,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config)

  predict_examples = []
  idx = 0
  texts = []
  for line in sys.stdin:
    line_lst = line.split("\t")
    guid = "%s-%s" % ("test", idx)
    predict_examples.append(SentenceExample(
        swag_id=guid,
        context_sentence=line_lst[0],
        ending_0=line_lst[1],
        ending_1=line_lst[2],
        label=0))
    idx += 1

    if idx%FLAGS.predict_batch_size == 0:
      num_actual_predict_examples = len(predict_examples)
      features = convert_examples_to_features(predict_examples, label_list, FLAGS.max_seq_length, tokenizer)
      predict_input_fn = input_fn_builder(
          features=features,
          batch_size=FLAGS.predict_batch_size,
          seq_length=FLAGS.max_seq_length,
          is_training=True,
          drop_remainder=True)

      result = estimator.predict(input_fn=predict_input_fn)
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        for (j, prediction) in enumerate(probabilities):
            output_line = predict_examples[i].context_sentence + "\t" \
                          + predict_examples[i].endings[j] + "\t" \
                          + str(prediction)
            print(output_line)
      predict_examples = []
      texts = []

  num_actual_predict_examples = len(predict_examples)
  if num_actual_predict_examples<1:
      return
  features = convert_examples_to_features(predict_examples, label_list, FLAGS.max_seq_length, tokenizer)
  predict_input_fn = input_fn_builder(
      features=features,
      batch_size=num_actual_predict_examples*2,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)

  result = estimator.predict(input_fn=predict_input_fn)
  for (i, prediction) in enumerate(result):
      probabilities = prediction["probabilities"]
      for (j, prediction) in enumerate(probabilities):
          output_line = predict_examples[i].context_sentence + "\t" \
                        + predict_examples[i].endings[j] + "\t" \
                        + str(prediction)
          print(output_line)

if __name__ == "__main__":
  tf.app.run()
