import re
import nltk
import tensorflow as tf
nltk.download('stopwords')
nltk.download('punkt')

class InputPipeline:
  def __init__(self, train_data,test_data,tokenizer,max_length):
    self.train_data = train_data
    self.test_data = test_data
    self.tokenizer = tokenizer
    self.max_len = max_length
    self.stopwords = nltk.corpus.stopwords.words('english')
    
    self.train_data['comment_text'] = self.train_data['comment_text'].astype(str)
    self.test_data['comment_text'] = self.test_data['comment_text'].astype(str)
    
    self.train_data['cleaned_comment'] = [self.clean_comment(comment) for comment in self.train_data["comment_text"]]
    self.test_data['cleaned_comment'] = [self.clean_comment(comment) for comment in self.test_data["comment_text"]]  

    self.train_data_tokenized_comments = [self.tokenize_comment(comment) for comment in self.train_data['cleaned_comment']]
    self.train_labels = self.train_data.drop(columns=['comment_text','cleaned_comment','id'],axis=1).values
    self.test_data_tokenized_comments = [self.tokenize_comment(comment) for comment in self.test_data['cleaned_comment']]
    self.test_labels = self.test_data.drop(columns=['comment_text','cleaned_comment','id'],axis=1).values

    self.train_input = tf.keras.preprocessing.sequence.pad_sequences(self.train_data_tokenized_comments, padding='post', maxlen=self.max_len)
    self.test_input = tf.keras.preprocessing.sequence.pad_sequences(self.test_data_tokenized_comments, padding='post', maxlen=self.max_len)

  def clean_comment(self,comment):
    comment = nltk.word_tokenize(comment.lower())
    comment = [i for i in comment if i not in self.stopwords]
    comment = " ".join(comment)
    comment = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"\
                              , " ", comment).split())
    return comment     
  
  def tokenize_comment(self,comment):
    return self.tokenizer.convert_tokens_to_ids(["[CLS]"] + self.tokenizer.tokenize(comment)[:self.max_len] + ["[SEP]"])
