from datasets import load_from_disk
from utils import *
import json
import sys


if __name__ == "__main__":

    stats = {}
    for steps in [640]:#,280,320,360,400,440,480,520,560,600]:
        dataset = load_from_disk("fine-tuning-kpmed/kp20k_generated_keyphrases_test_{}".format(steps))

        #print(dataset["test"]["pred"][0])
        #dataset = dataset.map(chowdhury_predictions_split,num_proc=10)
        print("SPLIT \n")
        dataset = dataset.map(predictions_split,num_proc=10)

        print("TOKENIZE \n")
        dataset = dataset.map(tokenize_predictions, num_proc=10)

        print("TOPK \n")
        dataset = dataset.map(topk,num_proc = 8,fn_kwargs={"n":int(sys.argv[1])})

        print("KEYPHRASES")
        print(dataset["test"]["keyphrases"][1])

        stats[steps] = {}


        print("CORRESSPONDANCE \n")
        dataset = dataset.map(correspondance,num_proc=8, fn_kwargs={"topk":int(sys.argv[1])})

        dataset.save_to_disk("temporary")

        stats[steps]["scores"] = macro_f1(dataset)

    print(stats)

    with open("stats_KP20k_evaluation_top{}.jsonl".format(sys.argv[1]),"a") as output:
        for line in stats:
            json.dump(stats[line],output)
            output.write("\n")   

