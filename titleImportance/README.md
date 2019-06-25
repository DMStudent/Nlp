# 重要度计算
提供有区分度，可理解的查询词重要性

- 分数：0-100
- 含义：停用词，可去词，补充词，重要词，核心词

对一句话分词，计算每个词在该句中的重要度，如:
    lstm 模型 如何 工作
    100 30 10 50

* 中文词向量处理
* Bi-LSTM + CRF 模型
* early stop
* learning rate decay

模型采用了Bi-LSTM + CRF。

运行方式：
1. 训练
	sh run.sh train
2. 预测
	sh run.sh test
3. 数据预处理
    python preprocess_mp.py
    多进程并行处理大量数据，转换为tfrecords格式。


