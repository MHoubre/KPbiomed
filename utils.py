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
"""
Function that splits the sequence of keyword that the model gives us
"""
def predictions_split(dataset):
    splits = []
    for seq in dataset["pred"]:
        seq = re.sub(r'<unk>|<s>|<\/s>|<\/unk>|<pad>|mask','',seq)
        #seq = re.sub(r'<$|<K$|<KP$|,$','',seq)  
        splits.append(seq.split(" <KP> "))
    dataset["splits"] = splits

    return dataset

def chowdhury_predictions_split(dataset):
    splits = []
    for seq in dataset["pred"]:
        seq = re.sub(r'\s{2}|<$|<K$|<KP$|\([a-zA-Z]+$','',seq)        
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

def spacy_tokenize(kp):
    keyphrase_tokens = nlp(kp)
    keyphrase_stems = [Stemmer('porter').stem(w.text.lower()) for w in keyphrase_tokens]
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


def correspondance(dataset, topk,keyphrases = None, keyphrase_category="all"):
    hypotheses = dataset["top_{}".format(topk)]
    if keyphrase_category=="all":
        references = dataset["keyphrases"]
    else:
        ids = dataset["id"]
        references = keyphrases[ids][keyphrase_category]
    correspondance=[]
    
    precision = 0
    recall = 0
    f_measure=0
    
    for hypothese in hypotheses :
        if hypothese in references:
            correspondance.append(hypothese)

    # fn est le nombre de mots-clés restants dans la ref parce qu'ils n'ont pas été reconnus
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
    preci = np.mean(dataset["test"]["precision"])
    rec = np.mean(dataset["test"]["recall"])
    f= 2*(preci * rec) / (preci + rec)
    return f