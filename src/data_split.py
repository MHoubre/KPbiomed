#!/usr/bin/env python
# coding: utf-8

from datasets import load_dataset, Dataset
import random
from sklearn.model_selection import train_test_split
import collections
import json
import spacy

data = load_dataset("json",data_files="data_prmu.jsonl")

data = data["train"]

print(collections.Counter(data["year"]))
print("\n")


t_size = len(data)-40000 # 40 000 because we want the test and validation sets to have 20 000 docs each

x_train,x_test = train_test_split(data, train_size=t_size,stratify=data["year"],random_state=1) #Train test split

train_dataset = Dataset.from_dict(x_train) # We have the large train set. Putting it back in Dataset form
test_val_dataset = Dataset.from_dict(x_test) # Putting the subset back to Dataset form to get val,test datasets after split

test,val = train_test_split(test_val_dataset, train_size=0.5,stratify=test_val_dataset["year"],random_state=1) # Getting val,test sets

del(test_val_dataset)
del(x_test)

# Putting both subsets back to Dataset form to store it in json
test_dataset = Dataset.from_dict(test) 
val_dataset = Dataset.from_dict(val)

with open("/home/houbre/train_large.json","w") as output:
    for line in train_dataset:
        json.dump(line,output)
        output.write("\n")


test_dataset.to_json("/home/houbre/test.jsonl",num_proc=5)
val_dataset.to_json("/home/houbre/val.jsonl",num_proc=5)

del(test_dataset)
del(val_dataset)

# Repeating process to get train medium and train small. Val and test stay unchanged

train_medium, _ = train_test_split(train_dataset, train_size=2000000,stratify=train_dataset["year"],random_state=1)

medium_dataset = Dataset.from_dict(train_medium)

with open("/home/houbre/train_medium.json","w") as output2:
    for line in medium_dataset:
        json.dump(line,output2)
        output2.write("\n")

print(collections.Counter(medium_dataset["year"]))
print("\n")

del(medium_dataset)

train_small, _ = train_test_split(train_dataset,train_size=500000,stratify=train_dataset["year"],random_state=1) 

small_dataset = Dataset.from_dict(train_small)

print(collections.Counter(small_dataset["year"]))
print("\n")

small_dataset.to_json("/home/houbre/train_small.jsonl",num_proc=5)
