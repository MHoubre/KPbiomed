#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datasets


# In[2]:


data = datasets.load_dataset("json",data_files="kpmed.jsonl")
data = data["train"]


# In[3]:
def to_int(dataset):
    if dataset["year"] != '':
        dataset["year"] = int(dataset["year"])
    else:
        dataset["year"] = 0
    return dataset

data = data.map(to_int,num_proc=9)


data = data.filter(lambda example: example['year']>=2011,num_proc=9)
data.to_json("output_sub_10y.jsonl")


# In[6]:


print(data.num_rows)


# In[ ]:




