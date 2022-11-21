from this import d
from datasets import load_dataset, load_from_disk, ReadInstruction
import spacy
import re
# from spacy.lang.en import English
from spacy.tokenizer import _get_regex_pattern
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from nltk.stem.snowball import SnowballStemmer as Stemmer
import numpy as np



# In[3]:

nlp = spacy.load("en_core_web_sm",disable=['tagger','parser','ner','lemmatizer','textcat'])

"""
Function that splits the sequence of keyword that the model gives us
"""
def predictions_split(dataset):
    splits = []
    for seq in dataset["pred"]: #for each sequence in predictions
        seq = re.sub(r'<unk>|<s>|<\/s>|<pad>|<\/unk>','',seq) #We get rid of residual special tokens
        seq = [re.sub(r'^ | $','',sp) for sp in seq.split("<KP>")] #We split with our special character
        #print(seq)
        if seq[-1]=='':
            seq = seq[:-1] #the split with <KP> leaves residual whitespaces at the beginning or and ending of keyphrases
        splits.append(seq) 
    dataset["pred"] = splits
    dataset["splits"] = splits

    return dataset

def chowdhury_predictions_split(dataset):
    splits = []
    for seq in dataset["pred"]:
        seq = re.sub(r'\s{2}|<$|<K$|<KP$|\([a-zA-Z]+$','',seq)        
        splits.append(seq.split(","))
    dataset["splits"] = splits

    return dataset

def get_presents(dataset):
    presents = np.where(np.isin(dataset["prmu"], "P"))[0].tolist()
    presents = [dataset["tokenized_keyphrases"][i] for i in presents] #Getting tokenized present keyphrases for evaluation
    dataset["presents"]=presents
    return dataset


def get_absents(dataset):
    absents = np.where(np.isin(dataset["prmu"], "P", invert=True))[0].tolist()
    absents = [dataset["tokenized_keyphrases"][i] for i in absents] #Getting tokenized absent keyphrases for evaluation
    dataset["absents"] = absents
    return dataset

def tokenize(kp):
    keyphrase_tokens = kp.split()
    keyphrase_stems = [Stemmer('porter').stem(w.lower()) for w in keyphrase_tokens]
    return " ".join(keyphrase_stems)
  

def tokenize_keyphrases(dataset):
    keyphrases = []
    for keyphrase in dataset["keyphrases"]:
        keyphrases.append(tokenize(keyphrase))
    dataset["tokenized_keyphrases"] = keyphrases
    return dataset
    
"""
Function that tokenizes the keyphrases before looking for the topk
""" 
def tokenize_predictions(dataset):
    tok_preds=[]
    for kp_list in dataset["pred"]:
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

def unstemmed_topk(dataset,n=5):
    topk=[]
    tokenized =[]

    # Tant que l'on n'a pas assez de mots-clés ou qu'on n'a pas tout vidé
    while len(topk) < n and dataset["splits"] != []: 
        #print(l)
        for i,kp_list in enumerate(dataset["splits"]): # pour chacune des listes
            if len(kp_list) > 0: # s'il y a au moins un mot-clé dedans
                t = tokenize(kp_list[0])
                if t not in tokenized and t != '': # s'il n'est pas déjà dans la liste
                    tokenized.append(t)
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



def topm(dataset):
    splits = dataset["splits"][0]
    #print(splits)
    final_splits = []
    tokenized = []

    for split in splits:
        t = tokenize(split)
        if t not in tokenized:
            final_splits.append(split)
            tokenized.append(t)
            

    dataset["top_m"] = tokenized

    return dataset

def correspondance(dataset, topk,keyphrases = None, keyphrase_category="all"):
    hypotheses = dataset["top_{}".format(topk)]
    if keyphrase_category=="all":
        references = dataset["tokenized_keyphrases"]
    else:
        ids = dataset["id"]
        references = dataset[keyphrase_category]
    correspondance=[]
    
    precision = 0
    recall = 0
    f_measure=0
    
    for hypothese in hypotheses :
        if hypothese in references:
            correspondance.append(hypothese)


    if len(correspondance) == 0:
        precision = 0
        recall = 0
    else:
        
        # Hypotheses qui sont dans les références / nombre d'hypotheses
        precision = len(correspondance) / len(hypotheses)
        # Hypotheses qui sont dans les références / nombre de références
        recall = len(correspondance) / len(references)
        
        # Moyenne harmonique des deux précédents
        f_measure = 2 * (precision * recall) / (precision+recall)
    
    dataset["correspondances"] = correspondance
    dataset["precision"] = precision
    dataset["recall"] = recall
    dataset["f1_measure"] = f_measure
    return dataset


def macro_f1(dataset):

    return np.mean(dataset["test"]["f1_measure"])

def macro_recall(dataset):
    return np.mean(dataset["test"]["recall"])