#date
cat data/sentence/part | /search/anaconda/envs/py3/bin/python predict_lite.py 
#1>case.out 
#date
#cat incest/dev.tsv | /search/anaconda/envs/py36/bin/python predict.py > case.out
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
#export BERT_BASE_DIR=/search/wangyuan/bert/chinese_L-12_H-768_A-12
#export NEWS_DIR=/search/wangyuan/bert/data
#export INCEST_DIR=/search/wangyuan/bert/incest
#export CKPT_DIR=/search/wangyuan/bert/incest_output/model.ckpt-1000
#OUTPUT=/search/wangyuan/bert/incest_output
#
#CUDA_VISIBLE_DEVICES=0 /search/anaconda/envs/py36/bin/python my_run_classifier.py \
#  --task_name incest \
#  --do_predict \
#  --data_dir $INCEST_DIR  \
#  --vocab_file $BERT_BASE_DIR/vocab.txt \
#  --bert_config_file $BERT_BASE_DIR/bert_config.json \
#  --init_checkpoint $CKPT_DIR \
#  --max_seq_length 30 \
#  --train_batch_size 128 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 3.0 \
#  --output_dir $OUTPUT \
#  --local_rank 3

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
