#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.core.example import example_pb2
import struct


# In[2]:


import pandas as pd


# In[3]:


whole_dataset = pd.read_csv('../wh_data/wikihowAll.csv')


# In[4]:


split = 0.7
num_train = int(len(whole_dataset) * split)


# In[5]:


train = whole_dataset.loc[:num_train]
val = whole_dataset.loc[num_train:]


# In[9]:


TRAIN_PREFIX = "../wh_data/train/train_"
VAL_PREFIX = "../wh_data/val/val_"
CHUNK_SIZE = 1000

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

def pre_process_frame(frame):
    frame['headline'] = frame['headline'].str.replace('\n', '')
    frame['headline'] = frame['headline'].str.replace('.', ' . ')
    frame['headline'] = frame['headline'].str.replace(',', ' , ')
    frame['headline'] = frame['headline'].str.replace('?', ' ? ')
    frame['headline'] = frame['headline'].str.replace('!', ' ! ')
    frame['headline'] = SENTENCE_START + frame['headline'].str.lower() + SENTENCE_END
    
    frame['text'] = frame['text'].str.replace('\n', '')
    frame['text'] = frame['text'].str.replace('.', ' . ')
    frame['text'] = frame['text'].str.replace(',', ' , ')
    frame['text'] = frame['text'].str.replace('?', ' ? ')
    frame['text'] = frame['text'].str.replace('!', ' ! ')
    frame['text'] = frame['text'].str.lower()
    return frame

def row_to_ex(text, headline):
    tf_example = example_pb2.Example()
    tf_example.features.feature['article'].bytes_list.value.extend([text.encode()])
    tf_example.features.feature['abstract'].bytes_list.value.extend([headline.encode()])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    return tf_example_str, str_len

def write_chunk_to_file(chunk, file):
    with open(file, 'wb') as writer:
        assert len(chunk) != 0
        for idx, row in chunk.iterrows():
            tf_example_str, str_len = row_to_ex(str(row['text']), str(row['headline']))
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))
    
def write_frame_to_file(frame, prefix):
    frame = pre_process_frame(frame)
    start = 0
    end = CHUNK_SIZE
    count = 1
    assert len(frame != 0)
    while start < len(frame):
        end = min(end, len(frame))
        assert start != end
        chunk = frame[start:end]
        assert len(chunk) != 0
        write_chunk_to_file(chunk, prefix + str(count) + ".bin")
        count += 1
        start = end
        end += CHUNK_SIZE
                   


# In[10]:


write_frame_to_file(val, VAL_PREFIX)
write_frame_to_file(train, TRAIN_PREFIX)


# In[11]:





# In[ ]:




