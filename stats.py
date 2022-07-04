#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset
import sys

dataset = load_dataset("json",data_files=sys.argv[1])


# In[3]:


from tqdm.notebook import tqdm

P, R, M, U, nb_kps = [], [], [], [], []

for split in ['train']:
    
    for sample in tqdm(dataset[split]):
        nb_kps.append(len(sample["keyphrases"]))
        P.append(sample["prmu"].count("P") / nb_kps[-1])
        R.append(sample["prmu"].count("R") / nb_kps[-1])
        M.append(sample["prmu"].count("M") / nb_kps[-1])
        U.append(sample["prmu"].count("U") / nb_kps[-1])
        
print("# keyphrases: {:.2f}".format(sum(nb_kps)/len(nb_kps)))
print("% P: {:.2f}".format(sum(P)/len(P)*100))
print("% R: {:.2f}".format(sum(R)/len(R)*100))
print("% M: {:.2f}".format(sum(M)/len(M)*100))
print("% U: {:.2f}".format(sum(U)/len(U)*100))


# In[4]:


import spacy

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


# In[ ]:


doc_len = []
for split in ['train']:
    for sample in tqdm(dataset[split]):
        doc_len.append(len(nlp(sample["abstract"])))
print("avg doc len: {:.1f}".format(sum(doc_len)/len(doc_len)))


# In[ ]:




