## 面向部分标签数据的植物病虫害实体识别 

这个仓库面向植物病虫害领域的实体识别任务。利用植物病虫害领域的文献摘要，构建部分标签数据，并通过一种结合经验分布和迁移学习的方法，提高部分标注数据在实体识别任务中的表现。目前我们实现了BiLstm-CRF模型和Bert-CRF模型，将该方法用于以上两种模型，分别在植物病虫害数据集和优酷视频数据集上进行了测试。
### Requirements
* PyTorch >= 1.1
* Python 3
### 训练模型
1.把预训练好的词向量放在BiLstm-partial-ner/data目录下，我们使用的是gigaword的50维词向量，[点击进行下载](https://github.com/allanj/ner_incomplete_annotation/tree/aa20c015b3f373ac4a1893e629ac8f2dd137faab)。

2.将预训练的msra模型放在Bert-partial-ner/saved_model/bert_pretrain目录下，[点击下载预训练模型](https://github.com/allanj/ner_incomplete_annotation/tree/aa20c015b3f373ac4a1893e629ac8f2dd137faab)。

3.把模型文件 {"bert_config.json", "pytorch_model.bin", "vocab.txt"} 放在/Bert-partial-ner/bert-base-chinese-pytorch目录下。

4.运行 "runBertModel.sh"
