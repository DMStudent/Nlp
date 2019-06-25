export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/search/anaconda/envs/py36/lib
export BERT_BASE_DIR=/search/wangyuan/bert/chinese_L-12_H-768_A-12
export SENTENCE_DIR=data/sentence
export CKPT_DIR=output_bkp/model.ckpt-105000
export OUTPUT=output/

CUDA_VISIBLE_DEVICES=1 /search/anaconda/envs/py36/bin/python multiple_choice.py \
  --do_train \
  --do_eval \
  --data_dir $SENTENCE_DIR  \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --init_checkpoint $CKPT_DIR \
  --max_seq_length 30 \
  --train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir $OUTPUT \
  --local_rank 3

#python run_classifier_word.py \
#  --task_name NEWS \
#  --do_train \
#  --do_eval \
#  --data_dir $GLUE_DIR/News/ \
#  --vocab_file $BERT_BASE_DIR/vocab.txt \
#  --bert_config_file $BERT_BASE_DIR/bert_config.json \
#  --init_checkpoint $BERT_PYTORCH_DIR/pytorch_model.bin \
#  --max_seq_length 128 \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 3.0 \
#  --output_dir ./news_output/ \
#  --local_rank 2

#python run_classifier.py \
#  --task_name MRPC \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --data_dir $GLUE_DIR/MRPC/ \
#  --vocab_file $BERT_BASE_DIR/vocab.txt \
#  --bert_config_file $BERT_BASE_DIR/bert_config.json \
#  --init_checkpoint $BERT_PYTORCH_DIR/pytorch_model.bin \
#  --max_seq_length 128 \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 3.0 \
#  --output_dir ./mrpc_output/
