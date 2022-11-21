from json import load
import re
import os
import json
import spacy
from tqdm.notebook import tqdm
from nltk.stem.snowball import SnowballStemmer as Stemmer
from datasets import load_dataset, ReadInstruction
from spacy.tokenizer import _get_regex_pattern



# Tokenization fix for in-word hyphens (e.g. 'non-linear' would be kept 
# as one token instead of default spacy behavior of 'non', '-', 'linear')
# https://spacy.io/usage/linguistic-features#native-tokenizer-additions

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from pke.unsupervised import *
import numpy as np
import sys
from utils import get_presents




def get_text(dataset):
    #dataset["docs"] = dataset["title"]+". "+dataset["abstract"]
    dataset["docs"] = dataset["abstract"]
    return dataset

def tokenize_keyphrases(dataset):
# populates a docs list with spacy doc objects

    references = []

    for keyphrase in dataset["keyphrases"]:
        # tokenize keyphrase
        tokens = [token for token in keyphrase.split()]
        # normalize tokens using Porter's stemming
        stems = [Stemmer('porter').stem(tok.lower()) for tok in tokens]
        references.append(" ".join(stems))
    
    

    dataset["tokenized_keyphrases"] = references
    return dataset


def evaluate(top_N_keyphrases, references):
    
    P = len(set(top_N_keyphrases) & set(references)) / len(top_N_keyphrases)
    R = len(set(top_N_keyphrases) & set(references)) / len(references)
    F = (2*P*R)/(P+R) if (P+R) > 0 else 0 

    return (P, R, F)


if __name__ == "__main__":

    nlp = spacy.load("en_core_web_sm", disable=["ner","lemmatizer","textcat"])

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

    dataset = load_dataset("json",data_files={"test":sys.argv[1]})
    dataset = dataset.map(get_text)
    dataset = dataset.map(tokenize_keyphrases)
    dataset = dataset.map(get_presents)
    dataset = dataset.filter(lambda ex: len(ex["presents"]) != 0)

    print(dataset)

    outputs = {}
    for model in [MultipartiteRank]:
        outputs[model.__name__] = []
        
        extractor = model()
        for i, doc in enumerate(tqdm(dataset["test"]["docs"])):
            doc = nlp(doc)
            extractor.load_document(input=doc, language='en')
            extractor.grammar_selection(grammar="NP: {<ADJ>*<NOUN|PROPN>+}")
            extractor.candidate_weighting()
            outputs[model.__name__].append([u for u,v in extractor.get_n_best(n=10, stemming=True)])

        #print(outputs)

    # loop through the models

    results = {}
    for model in outputs:
        
        results[model] = {}
        # compute the P, R, F scores for the model
        scores = []
        for i, output in enumerate(tqdm(outputs[model])):
            scores.append(evaluate(output, dataset["test"]["presents"][i]))
        
        # compute the average scores
        avg_scores = np.mean(scores, axis=0)

        results[model]["P@10"] = avg_scores[0]
        results[model]["R@10"] = avg_scores[1]
        results[model]["F@10"] = avg_scores[2]
        
        # print out the performance of the model
        print("Model: {} P@10: {:.3f} R@10: {:.3f} F@10: {:.3f}".format(model, avg_scores[0], avg_scores[1], avg_scores[2]))
    
    with open("results_@10_{}".format(os.path.basename(sys.argv[1])),"a") as f:
        json.dump(results,f)

    