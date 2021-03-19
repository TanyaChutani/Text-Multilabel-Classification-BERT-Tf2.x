import tensorflow as tf
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights

class BERT(tf.keras.models.Model):
  def __init__(self,max_len, bert_checkpoint,bert_config):
    super(BERT, self).__init__()
    self.max_len = max_len
    self.bert_checkpoint = bert_checkpoint
    self.bert_config = bert_config
    self.labels = 6
    self.bert = self.call_model()
    self.lambda_layer = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])
    self.dropout_layer_1 = tf.keras.layers.Dropout(0.5)
    self.dense_layer_1 = tf.keras.layers.Dense(units=768, activation="tanh")
    self.dropout_layer_2 = tf.keras.layers.Dropout(0.5)
    self.dense_layer_2 = tf.keras.layers.Dense(units=self.labels, activation="sigmoid")

  def call_model(self):
    with tf.io.gfile.GFile(self.bert_config, "r") as reader:
      cfg = StockBertConfig.from_json_string(reader.read())
      bert_params = map_stock_config_to_params(cfg)
      bert_params.adapter_size = None
      bert_model = BertModelLayer.from_params(bert_params)  
      return bert_model

  def call(self,input_tensor):
    x = self.bert(input_tensor)
    x = self.lambda_layer(x)
    x = self.dropout_layer_1(x)
    x = self.dense_layer_1(x)
    x = self.dropout_layer_2(x)
    x = self.dense_layer_2(x)
    return x
