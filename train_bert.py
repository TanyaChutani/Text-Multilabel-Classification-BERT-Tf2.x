import os
import re
import bert
import nltk
import tqdm
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from bert import BertModelLayer
import matplotlib.pyplot as plt
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights

nltk.download('stopwords')
nltk.download('punkt')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch',
                        '--bert_epochs',
                        type=int,
                        metavar='',
                        default = 3)

    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        metavar='',
                        default = 16)
    
    parser.add_argument('-max_len',
                        '--max_length',
                        type=int,
                        metavar='',
                        default = 150)
    
    parser.add_argument('-img_dir',
                        '--train_path',
                        type=str,
                        metavar='',
                        default = 'train.csv')
    
    parser.add_argument('-img_dir',
                        '--test_path',
                        type=str,
                        metavar='',
                        default = 'test.csv')

    parser.add_argument('-w',
                        '--bert_main_checkpoint',
                        type=str,
                        metavar='',
                        default = "uncased_L-12_H-768_A-12/bert_model.ckpt'")

    parser.add_argument('-b_vocab',
                        '--bert_vocab',
                        type=str,
                        metavar='',
                        default = "uncased_L-12_H-768_A-12/vocab.txt")
  parser.add_argument('-b_config',
                        '--bert_config',
                        type=str,
                        metavar='',
                        default = "uncased_L-12_H-768_A-12/bert_config.json")
    args = parser.parse_args()
    return args

def train_bert():
  args = parse_args()
  classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

  train_data = pd.read_csv(args.train_path)
  test_data = pd.read_csv(args.test_path)

  print(train_data.head())
  print("Missing Values",((train_data.isnull() | train_data.isna())\
                          .sum() * 100 / train_data.index.size).round(2))
  #EDA
  count_labels = pd.DataFrame(train_data[classes].sum()).reset_index()
  count_labels.columns = ['labels','count']
  bar_graph = plt.bar(count_labels['labels'],count_labels['count'])
  plt.xlabel('labels')
  plt.ylabel('count')
  plt.show()
    
  tokenizer = bert.tokenization.bert_tokenization.FullTokenizer(vocab_file=os.path.join(args.bert_main_checkpoint, args.bert_vocab))
  pipeline = InputPipeline(train_data = train_data,
                         test_data = test_data,
                         tokenizer = tokenizer)
  
  bert_model = BERT(max_len=args.max_length, bert_checkpoint=args.bert_main_checkpoint,bert_config=args.bert_config)
  bert_model.compile(
  optimizer="adam",
  loss="binary_crossentropy",
  metrics=["accuracy"])

  history = bert_model.fit(
  x=pipeline.train_input, 
  y=pipeline.train_labels,
  validation_split=0.05,
  batch_size=args.batch_size,
  shuffle=True,
  epochs=args.bert_epochs)

  results = model.evaluate(pipeline.test_input,pipeline.test_labels)

if __name__ == "__main__":
  train_bert()
