**“纸上得来终觉浅，绝知此事要躬行。”  —— 陆游**

**"Practice，practice，practice and summary makes perfect" —— dyliuti**

------

**BERT分类**

使用BERT进行分类。下载数据和模型后，控制台中输入以下指令即可运行。

export BERT_Chinese_DIR=chinese_L-12_H-768_A-12

export Demo_DIR=data

python run_classifier.py \

--task_name=demo \

--do_train=true \

--do_eval=true \

--data_dir=$Demo_DIR \

--vocab_file=$BERT_Chinese_DIR/vocab.txt \

--bert_config_file=$BERT_Chinese_DIR/bert_config.json \
  --init_checkpoint=$BERT_Chinese_DIR/bert_model.ckpt \

--max_seq_length=128 \
  --train_batch_size=8 \

--learning_rate=2e-5 \

--num_train_epochs=3.0 \

 --output_dir=Demo_output/



