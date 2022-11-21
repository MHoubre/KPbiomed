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
from fine_tuning import get_text

import torch
import numpy as np
import random
import re
from nltk.stem.snowball import SnowballStemmer as Stemmer


def generate_keyphrases(batch, key):
    
    
    inputs = tokenizer(
        batch[key],padding="max_length",max_length= 512,truncation=True, return_tensors='pt'
    )
    
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

                             
    outputs = model.generate(inputs=input_ids,attention_mask=attention_mask,
                             num_beams=20,
                             num_return_sequences=20
                             )
    # all special tokens including will be removed
    output_strs = tokenizer.batch_decode(outputs,skip_special_tokens=False)

    batch["pred"] = output_strs

    return batch


def stemm(kp):
    keyphrase_tokens = kp.split()
    keyphrase_stems = [Stemmer('porter').stem(w.lower()) for w in keyphrase_tokens]
    return " ".join(keyphrase_stems)
    
    
def stemm_keyphrases(dataset):
    keyphrases = []
    for keyphrase in dataset["keyphrases"]:
        keyphrases.append(stemm(keyphrase))
    dataset["tokenized_keyphrases"] = keyphrases
    return dataset
    
"""
Function that tokenizes the keyphrases before looking for the topk
""" 
def stemm_predictions(dataset):
    tok_preds=[]
    for kp_list in dataset["splits"]:
        kp_l = []
        for kp in kp_list:
            kp_l.append(stemm(kp))

        tok_preds.append(kp_l)
    dataset["splits"] = tok_preds
    return dataset
        


def macro_f1(dataset):
    preci = np.mean(dataset["precision"])
    rec = np.mean(dataset["recall"])
    f= 2*(preci * rec) / (preci + rec)
    return f

if __name__ == "__main__":

    for i in ["kpmed","kp20k","kptimes"]:

        dataset = load_dataset("json",data_files={"test":"data/test_{}.jsonl".format(i)})


        dataset = dataset.map(get_text, num_proc=9)

        tokenizer = BartTokenizerFast.from_pretrained("training_jean-zay/fine-tuning-biobart-kp20k/final_model_biobart-kp20k")
        model = BartForConditionalGeneration.from_pretrained("training_jean-zay/fine-tuning-biobart-kp20k/final_model_biobart-kp20k")

        model.to("cuda")

        dataset = dataset.map(generate_keyphrases, fn_kwargs={"key":"text"})


        dataset.save_to_disk("training_jean-zay/fine-tuning-biobart-kp20k/generated/generated_{}".format(i))
                                            

# In[15]:


