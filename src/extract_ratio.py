from datasets import load_dataset
import sys

from prmu import contains
import numpy as np

d = load_dataset("json",data_files="resultats/resultats-kpmed_{}@m.json".format(sys.argv[1]))

def extract_proportion(dataset):
    extracted = 0
    print(dataset["pred"][0])
    for prediction in dataset["pred"][0]:
        if contains(prediction.lower(),dataset["text"].lower()):
            extracted += 1
    dataset["extract_prop"] = extracted  * 100 / len(dataset["pred"][0])
    return dataset

d = d.map(extract_proportion, num_proc = 5)
print(np.mean(d["train"]["extract_prop"]))