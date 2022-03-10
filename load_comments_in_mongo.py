#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pymongo
import urllib.parse
import json
import re
import spacy
import en_core_web_sm
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import FreqDist
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


# In[2]:


import pprint
from pyspark.sql import SparkSession
from operator import add
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# New API
spark_session = SparkSession    .builder    .appName("test_notebook")    .master("spark://192.168.2.176:7077")    .config("spark.dynamicAllocation.enabled", True)    .config("spark.dynamicAllocation.shuffleTracking.enabled", True)    .config("spark.shuffle.service.enabled", False)    .config("spark.dynamicAllocation.executorIdleTimeout","100s")    .config("spark.driver.port",9998)    .config("spark.blockManager.port",10005)    .getOrCreate()

# Old API (RDD)
spark_context = spark_session.sparkContext

spark_context.setLogLevel("ERROR")


# In[3]:


print(spark_context.uiWebUrl)


# In[4]:


schema = StructType([
      StructField("subreddit",StringType(),True),
      StructField("body",StringType(),True)]
)


# In[5]:


df = spark_session.read.schema(schema).json("hdfs://host-192-168-2-176-de1:9000/comments/RC_2011-07")


# In[6]:


df = df.limit(100000)


# In[7]:


categories = df.select('subreddit').distinct().collect()


# In[8]:


i = 0

cat_to_id = {}
id_to_cat = {}

for c in categories:
    cat_to_id[c['subreddit']] = i
    id_to_cat[i] = c['subreddit']
    i += 1


# In[9]:


def transform_to_id(cat):
    return cat_to_id[cat]

transform_to_id_udf = udf(transform_to_id)


# In[10]:


df = df.withColumn("cat_id",transform_to_id_udf(df["subreddit"]).cast("int"))


# In[11]:


most_popular_categories = df.groupBy('cat_id').count().sort('count', ascending=False).head(50)


# In[12]:


categories_to_keep = []

for r in most_popular_categories:
    
    categories_to_keep.append(r['cat_id'])


# In[13]:


rdd = df.rdd.map(list)


# In[14]:


filtered_rdd = rdd.filter(lambda x: x[2] in categories_to_keep)


# In[15]:


filtered_rdd.count()


# In[16]:


filtered_rdd1 = filtered_rdd.map(lambda x: (x[2], emoji.get_emoji_regexp().sub(u'', x[1])))


# In[17]:


tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|http\S+')
filtered_rdd2 = filtered_rdd1.map(lambda x: (x[0], tokenizer.tokenize(x[1])))


# In[18]:


filtered_rdd3 = filtered_rdd2.map(lambda x: (x[0], [word.lower() for word in x[1]]))


# In[19]:


nlp = en_core_web_sm.load()

all_stopwords = nlp.Defaults.stop_words

filtered_rdd4 = filtered_rdd3.map(lambda x: (x[0], [word for word in x[1] if not word in all_stopwords]))


# In[20]:


nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

filtered_rdd5 = filtered_rdd4.map(lambda x: (x[0], ([lemmatizer.lemmatize(w) for w in x[1]])))


# In[21]:


nltk.download('vader_lexicon')

sia = SIA()

filtered_rdd6 = filtered_rdd5.map(lambda x: (x[0], [sia.polarity_scores(word)['compound'] for word in x[1]]))


# In[22]:


def find_positive_negative(scores):
    pos = 0
    neg = 0
    for s in scores:
        if s > 0.1 :
            pos += 1
        elif s <-0.1:
            neg += 1
    if pos >= neg:
        return 1
    return 0

find_positive_negative_udf = udf(find_positive_negative)


# In[23]:


filtered_rdd7 = filtered_rdd6.map(lambda x: (x[0], find_positive_negative(x[1])))


# In[24]:


final_rdd = filtered_rdd7.reduceByKey(lambda x, y: x+y)


# In[27]:


reduced_rdd = final_rdd.collect()


# In[28]:


count_dic = {}
for i in most_popular_categories:
    count_dic[i['cat_id']] = i['count']


# In[29]:


final_list = []


# In[30]:


for i in reduced_rdd:
    final_list.append([id_to_cat[i[0]], float(int(i[1]/count_dic[i[0]]*10000))/100])


# In[31]:


final_list


# In[32]:


final_df = pd.DataFrame (final_list, columns = ['category', 'frequency'])


# In[33]:


final_df.to_csv('frequency_table.csv', sep='\t')


# In[34]:


spark_session.stop()

