#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from transformers import BartConfig
from transformers import BartModel, BartForConditionalGeneration
from transformers import BartTokenizerFast
import datasets
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, ReadInstruction
from tokenizers import Tokenizer
from datasets import load_metric

import torch
import numpy as np
import random
import re
from nltk.stem.snowball import SnowballStemmer as Stemmer


# In[20]:


#dataset = load_from_disk("tokenized_keyphrases")



# In[13]:



# In[8]:

# In[16]:


def generate_keyphrases(batch):
    
    
    inputs = tokenizer(
        batch["text"],padding="max_length",max_length= 512,truncation=True, return_tensors='pt'
    )
    
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

                             
    outputs = model.generate(inputs=input_ids,attention_mask=attention_mask,
                             num_beams=20,
                             num_return_sequences=20
                             )
    # all special tokens including will be removed
    output_strs = tokenizer.batch_decode(outputs,skip_special_tokens=False)
    #output_str = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(outputs[0]))
    
    #decoder_log_prob = torch.nn.functional.log_softmax(decoder_logit, dim=-1).view(batch_size, 1, self.vocab_size)

    batch["pred"] = output_strs

    return batch

def concatenate_dataset(dataset):
    dataset["text"] = dataset["title"] + ". " + dataset["abstract"]
    return dataset

"""
Function that splits the sequence of keyword that the model gives us
"""
def predictions_split(dataset):
    splits = []
    for seq in dataset["pred"]:
        seq = re.sub(r'<unk>|<s>|<\/s>|<pad>|<\/unk>','',seq)
        splits.append(seq.split("<KP>"))
    dataset["splits"] = splits

    return dataset

def tokenize(kp):
    keyphrase_tokens = kp.split()
    keyphrase_stems = [Stemmer('porter').stem(w.lower()) for w in keyphrase_tokens]
    return " ".join(keyphrase_stems)
    
    
def tokenize_keyphrases(dataset):
    keyphrases = []
    for keyphrase in dataset["keyphrases"]:
        keyphrases.append(tokenize(keyphrase))
    dataset["keyphrases"] = keyphrases
    return dataset
    
"""
Function that tokenizes the keyphrases before looking for the topk
""" 
def tokenize_predictions(dataset):
    tok_preds=[]
    for kp_list in dataset["splits"]:
        kp_l = []
        for kp in kp_list:
            kp_l.append(tokenize(kp))

        tok_preds.append(kp_l)
    dataset["splits"] = tok_preds
    return dataset
        
"""
Function that gets the n topk keyphrases from a number of generated sequences
"""
def topk(dataset,n=5):
    topk=[]
    dataset["temp"] = dataset["splits"].copy()
    # Tant que l'on n'a pas assez de mots-clés ou qu'on n'a pas tout vidé
    while len(topk) < n and dataset["temp"] != []: 
        #print(l)
        for i,kp_list in enumerate(dataset["temp"]): # pour chacune des listes
            if len(kp_list) > 0: # s'il y a au moins un mot-clé dedans
                if kp_list[0] not in topk: # s'il n'est pas déjà dans la liste
                    topk.append(kp_list.pop(0)) # on l'ajoute
                    #print(len(topk))
                else:
                    kp_list.pop(0)
            else:
                #print(l)
                del dataset["temp"][i]

                break
    if len(topk) > n:
        dataset["top_{}".format(n)] = topk[:n]
    else:
        dataset["top_{}".format(n)] = topk
    return dataset

"""
Function that splits the sequence of keyword that the model gives us
"""
def predictions_split(dataset):
    splits = []
    for seq in dataset["pred"]:
        seq = re.sub(r'<unk>|<s>|<\/s>|<\/unk>|<pad>|mask','',seq)
        splits.append(seq.split("<KP>"))
    dataset["splits"] = splits

    return dataset

def chowdhury_predictions_split(dataset):
    splits = []
    for seq in dataset["pred"]:
        seq = re.sub(r'<unk>|<s>|<\/s>|<\/unk>|<pad>|mask','',seq)
        splits.append(seq.split(","))
    dataset["splits"] = splits

    return dataset

def get_presents_id(dataset):
    presents = np.where(np.isin(dataset["prmu"], "P"))[0].tolist()
    dataset["presents"]=presents
    return dataset


def get_presents(dataset):
    ids = dataset["presents"]
    presents = [dataset["keyphrases"][i] for i in ids]
    dataset["presents"]=presents
    return dataset


def get_absents_id(dataset):
    presents = np.where(np.isin(dataset["prmu"], "P", invert=True))[0].tolist()
    dataset["absents"]=presents
    return dataset

def get_absents(dataset):
    ids = dataset["absents"]
    presents = [dataset["keyphrases"][i] for i in ids]
    dataset["absents"]=presents
    return dataset

def tokenize(kp):
    keyphrase_tokens = kp.split()
    keyphrase_stems = [Stemmer('porter').stem(w.lower()) for w in keyphrase_tokens]
    return " ".join(keyphrase_stems)
    
    
def tokenize_keyphrases(dataset):
    keyphrases = []
    for keyphrase in dataset["keyphrases"]:
        keyphrases.append(tokenize(keyphrase))
    dataset["keyphrases"] = keyphrases
    return dataset
    
"""
Function that tokenizes the keyphrases before looking for the topk
""" 
def tokenize_predictions(dataset):
    tok_preds=[]
    for kp_list in dataset["splits"]:
        kp_l = []
        for kp in kp_list:
            kp_l.append(tokenize(kp))

        tok_preds.append(kp_l)
    dataset["splits"] = tok_preds
    return dataset
        
"""
Function that gets the n topk keyphrases from a number of generated sequences
"""
def topk(dataset,n):
    topk=[]
    # Tant que l'on n'a pas assez de mots-clés ou qu'on n'a pas tout vidé
    while len(topk) < n and dataset["splits"] != []: 
        #print(l)
        for i,kp_list in enumerate(dataset["splits"]): # pour chacune des listes
            if len(kp_list) > 0: # s'il y a au moins un mot-clé dedans
                if kp_list[0] not in topk and kp_list[0] != '': # s'il n'est pas déjà dans la liste
                    topk.append(kp_list.pop(0)) # on l'ajoute
                    #print(len(topk))
                else:
                    kp_list.pop(0)
            else:
                #print(l)
                del dataset["splits"][i]

                break
    if len(topk) > n:
        dataset["top_{}".format(n)] = topk[:n]
    elif len(topk) < n:
        for j in range(n-len(topk)):
            topk.append("<unk>")
        dataset["top_{}".format(n)] = topk
    else:
        dataset["top_{}".format(n)] = topk
    return dataset

# In[17]:



def macro_f1(dataset):
    preci = np.mean(dataset["precision"])
    rec = np.mean(dataset["recall"])
    f= 2*(preci * rec) / (preci + rec)
    return f

if __name__ == "__main__":
    #dataset = load_dataset("json", data_files={"test":"test.jsonl"})
    dataset = load_from_disk("tokenized_test_keyphrases")
    #dataset = dataset.map(concatenate_dataset, num_proc=9)

    dataset = dataset.map(tokenize_keyphrases,num_proc=10)
    dataset.save_to_disk("tokenized_test_keyphrases")

    for i in [640]:

        tokenizer = BartTokenizerFast.from_pretrained("/home/houbre/Documents/BART_kp20k/fine-tuning-kp20k/checkpoint-{}000".format(i))
        model = BartForConditionalGeneration.from_pretrained("/home/houbre/Documents/BART_kp20k/fine-tuning-kp20k/checkpoint-{}000".format(i))

        model.to("cuda")

        dataset = dataset.map(generate_keyphrases)

        dataset = dataset.map(predictions_split,num_proc=8)
        dataset = dataset.map(tokenize_predictions, num_proc=8)

        dataset = dataset.map(get_presents_id,num_proc=10)
        dataset = dataset.map(get_presents,num_proc=10)
        dataset = dataset.map(get_absents_id,num_proc=10)
        dataset= dataset.map(get_absents,num_proc=10)

        dataset.save_to_disk("fine-tuning-kpmed/kp20k_generated_keyphrases_test_{}".format(i))
                                           

# In[15]:


