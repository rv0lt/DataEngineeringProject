#!/usr/bin/env python
# coding: utf-8

import sys

# In[1]:


try:
    no_of_records = str(sys.argv[1])
    cores_per_executor = int(sys.argv[2])
    memory = str(int(sys.argv[3]))
    instances = int(sys.argv[4])
except:
    sys.exit(1)


# In[ ]:





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


# In[3]:


import pprint
from pyspark.sql import SparkSession
from operator import add
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

app_string = "R = " + str(no_of_records) + ", C/E = " + str(cores_per_executor) + ", M/E = "  \
 + str(memory) + ", E = " + str(instances)

# New API
spark_session = SparkSession    .builder    .appName(app_string)    .master("spark://host-192-168-2-176-de1:7077")    .config("spark.dynamicAllocation.enabled", True)    .config("spark.dynamicAllocation.shuffleTracking.enabled", True)    .config("spark.shuffle.service.enabled", False)    .config("spark.dynamicAllocation.executorIdleTimeout","100s")    .config("spark.driver.port",9998)    .config("spark.blockManager.port",10005)    .config("spark.cores.max", 10)    .config("spark.executor.cores",cores_per_executor)    .config("spark.executor.instances", instances)  .config("spark.ui.showConsoleProgress", False)  .config("spark.executor.memory", memory + "g")    .getOrCreate()

# Old API (RDD)
spark_context = spark_session.sparkContext

spark_context.setLogLevel("ERROR")

print()
print('------------------------------------------------------------')
print('No of records = ' + str(no_of_records) + 'k')
print('Cores per executor = ' + str(cores_per_executor))
print('Memory of executor = ' + str(memory))
print('Total executors = ' + str(instances))


# In[5]:


schema = StructType([
      StructField("subreddit",StringType(),True),
      StructField("body",StringType(),True)]
)


# In[7]:


df = spark_session.read.schema(schema)    .json("hdfs://host-192-168-2-176-de1:9000/comments/" + no_of_records + "k_elements.json")


# In[11]:


categories = df.select('subreddit').distinct().collect()


# In[13]:


i = 0

cat_to_id = {}
id_to_cat = {}

for c in categories:
    cat_to_id[c['subreddit']] = i
    id_to_cat[i] = c['subreddit']
    i += 1


# In[14]:


def transform_to_id(cat):
    return cat_to_id[cat]

transform_to_id_udf = udf(transform_to_id)


# In[15]:


df2 = df.withColumn("cat_id",transform_to_id_udf(df["subreddit"]).cast("int"))


# In[16]:


most_popular_categories = df2.groupBy('subreddit').count().sort('count', ascending=False).take(50)


# In[17]:


categories_to_keep = []

for r in most_popular_categories:
    
    categories_to_keep.append(cat_to_id[r['subreddit']])


# In[19]:


rdd = df2.rdd


# In[22]:


filtered_rdd = rdd.filter(lambda x: x[2] in categories_to_keep)


# In[23]:


filtered_rdd1 = filtered_rdd.map(lambda x: (x[0], emoji.get_emoji_regexp().sub(u'', x[1])))


# In[24]:


tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|http\S+')
filtered_rdd2 = filtered_rdd1.map(lambda x: (x[0], tokenizer.tokenize(x[1])))


# In[25]:


filtered_rdd3 = filtered_rdd2.map(lambda x: (x[0], [word.lower() for word in x[1]]))


# In[26]:


nlp = en_core_web_sm.load()

all_stopwords = nlp.Defaults.stop_words

filtered_rdd4 = filtered_rdd3.map(lambda x: (x[0], [word for word in x[1] if not word in all_stopwords]))


# In[27]:


# nltk.download('wordnet')
# nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

filtered_rdd5 = filtered_rdd4.map(lambda x: (x[0], ([lemmatizer.lemmatize(w) for w in x[1]])))


# In[28]:


# nltk.download('vader_lexicon')

sia = SIA()

filtered_rdd6 = filtered_rdd5.map(lambda x: (x[0], [sia.polarity_scores(word)['compound'] for word in x[1]]))


# In[29]:


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


# In[30]:


filtered_rdd7 = filtered_rdd6.map(lambda x: (x[0], find_positive_negative(x[1])))


# In[32]:


final_rdd = filtered_rdd7.reduceByKey(lambda x, y: x+y)


# In[33]:


reduced_rdd = final_rdd.collect()


# In[34]:


count_dic = {}
for i in most_popular_categories:
    count_dic[i['subreddit']] = i['count']


# In[35]:


final_list = []


# In[44]:


for i in reduced_rdd:
    final_list.append([i[0], float(int(i[1]/count_dic[i[0]]*10000))/100])


# In[45]:


final_df = pd.DataFrame (final_list, columns = ['category', 'frequency'])


# In[46]:


final_df.to_csv('frequency_table.csv', sep='\t')


# In[47]:


spark_session.stop()

