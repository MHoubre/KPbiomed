#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset, ReadInstruction
from nltk.stem.snowball import SnowballStemmer as Stemmer
import sys
from tqdm.notebook import tqdm
import numpy as np
from prmu import contains

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


def tokenize_keyphrases(dataset):
    keyphrases_stems= []
    for keyphrase in dataset["keyphrases"]:
        keyphrase_spacy = nlp(keyphrase)
        keyphrase_tokens = [token.text for token in keyphrase_spacy]
        keyphrase_stems = [Stemmer('porter').stem(w.lower()) for w in keyphrase_tokens]
        keyphrases_stems.append(keyphrase_stems)
        
    dataset["tokenized_keyphrases"] = keyphrases_stems
    return dataset

def tokenize_text(dataset):
    title_spacy = nlp(dataset['title'])
    abstract_spacy = nlp(dataset['abstract'])

    title_tokens = [token.text for token in title_spacy]
    abstract_tokens = [token.text for token in abstract_spacy]

    title_tokens[-1] = title_tokens[-1]+"."

    title_stems = [Stemmer('porter').stem(w.lower()) for w in title_tokens]
    abstract_stems = [Stemmer('porter').stem(w.lower()) for w in abstract_tokens]

    dataset["title_stems"] = title_stems
    dataset["abstract_stems"] = abstract_stems
    return dataset

def length_stats(dataset):
    doc_len = 0
    keyphrase_length = 0

    title = dataset["title_stems"]
    abstract = dataset["abstract_stems"]

    doc_len = (len(title) + len(abstract))
    kps_in_title = 0
    #print(dataset["keyphrases"])
    for i,kp in enumerate(dataset["tokenized_keyphrases"]):
        
        keyphrase_length += (len(kp))
        if contains(kp,title):
            kps_in_title +=1

    dataset["doc_len"] = doc_len
    dataset["kp_len"] = keyphrase_length / len(dataset["keyphrases"])
    dataset["kps_in_title"] = kps_in_title * 100 / len(dataset["tokenized_keyphrases"])
    #print(dataset["kps_in_title"])
    return dataset

if __name__ == "__main__":


    dataset = load_dataset("json",data_files=sys.argv[1])#,split=ReadInstruction("train",from_ = 0, to = 0.001, unit="%"))

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

    dataset = dataset.map(tokenize_keyphrases)
    dataset = dataset.map(tokenize_text)
    dataset = dataset.map(length_stats)

    with open("length_stats.txt","a") as f:
        f.write("average doc len: {:.1f}".format(np.mean(dataset["doc_len"])))
        f.write("\n")
        f.write("average kp len: {:.1f}".format(np.mean(dataset["kp_len"])))
        f.write("\n")
        f.write("average kps in title: {:.1f}%".format(np.mean(dataset["kps_in_title"])))
        f.write("\n")
