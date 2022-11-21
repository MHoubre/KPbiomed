#!/usr/bin/env python
# coding: utf-8

# In[5]:


from datasets import load_dataset
from tqdm.notebook import tqdm
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import spacy
import sys
import json

nlp = spacy.load("en_core_web_sm",disable=['tagger','parser','ner','lemmatizer','textcat'])

# https://spacy.io/usage/linguistic-features#native-tokenizer-additions

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

# Modify tokenizer infix patterns
infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        # ✅ Commented out regex that splits on hyphens between letters:
        # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)

infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer

d = load_dataset("json",data_files="{}.jsonl".format(sys.argv[1]))


def get_incorrect_kp(dataset):
    nb=0
    for kp in dataset["keyphrases"]:
        d = nlp(kp) #Tokenize the text using spacy tokenizer
        if "," in d.text: # If we have comas in the keyphrases
            nb+=1
    dataset["incorrect_kps"] = nb
    return dataset
            
d = d.map(get_incorrect_kp,num_proc=10)

d = d.filter(lambda example: example["incorrect_kps"]==0) # We remove any document that has keyphrases with comas in it

d = d.remove_columns("incorrect_kps")

d["train"].to_json("{}_correct_form.json".format(sys.argv[1]))

