from datasets import load_from_disk
from utils import *
import json
import sys

# from datasets.utils import disable_progress_bar
# disable_progress_bar()

if __name__ == "__main__":

    stats = {}
    for mod in ["kptimes"]:
        for data in ["kptimes","kpmed","kp20k"]:
            dataset = load_from_disk("generated_{}/generated_{}".format(mod,data))


            print("SPLIT \n")
            dataset = dataset.map(predictions_split,num_proc=10,load_from_cache_file=False)

            print("TOKENIZE \n")
            dataset = dataset.map(tokenize_predictions,num_proc=10,load_from_cache_file=False)
            dataset = dataset.map(tokenize_keyphrases,num_proc=10,load_from_cache_file=False)
            

            print("GETTING PRESENT AND ABSENTS \n")
            dataset = dataset.map(get_presents,num_proc=4,load_from_cache_file=False)
            dataset = dataset.map(get_absents, num_proc=4,load_from_cache_file=False)

            print("TOPK \n")
            if sys.argv[1] == "m":
                dataset = dataset.map(topm,num_proc=8,load_from_cache_file=False)
            
            else:
                dataset = dataset.map(topk,num_proc = 8,fn_kwargs={"n":int(sys.argv[1])},load_from_cache_file=False)

            d_pres = dataset.filter(lambda example:len(example["presents"]) !=0) # For evaluation on present keyphrases we onlytake articles that have at least one present keyphrase

            print("CORRESSPONDANCE \n")
            if sys.argv[1] == "m": # If we are evaluating using F1@M
                
                d_pres = d_pres.map(correspondance,num_proc=8, fn_kwargs={"topk":"m", "keyphrase_category":"presents"},load_from_cache_file=False)
                stats["{}_{}_presents".format(mod,data)] = macro_f1(d_pres)

            else: 

                dataset = dataset.map(correspondance,num_proc=8, fn_kwargs={"topk":int(sys.argv[1]),"keyphrase_category":"all"},load_from_cache_file=False)
                stats["{}_{}_all".format(mod,data)] = macro_f1(dataset)

                d_pres = d_pres.map(correspondance,num_proc=8, fn_kwargs={"topk":int(sys.argv[1]),"keyphrase_category":"presents"},load_from_cache_file=False)
                stats["{}_{}_presents".format(mod,data)] = macro_f1(d_pres)

                d_pres = d_pres.remove_columns(["splits","tokenized_keyphrases"])
                d_pres["test"].to_json("training_jean-zay/fine-tuning-{}/results/res_pres@{}_{}.jsonl".format(mod,sys.argv[1],data))

                d_abs = dataset.filter(lambda example:len(example["absents"]) !=0, num_proc=4) #To not penalize the model if there is no absent keyphrases to generate

                d_abs = d_abs.map(correspondance,num_proc=8, fn_kwargs={"topk":int(sys.argv[1]),"keyphrase_category":"absents"})

                stats["{}_{}_absents".format(mod,data)] = macro_recall(d_abs)

                d_abs = d_abs.remove_columns(["splits","tokenized_keyphrases"])
                d_abs["test"].to_json("training_jean-zay/fine-tuning-{}/results/res_abs@{}_{}.jsonl".format(mod,sys.argv[1],data))
                

            dataset = dataset.remove_columns(["splits","tokenized_keyphrases"])

            dataset["test"].to_json("training_jean-zay/fine-tuning-{}/results/res_all@{}_{}.jsonl".format(mod,sys.argv[1],data))

            print(stats)

    with open("results_evaluation_top{}.jsonl".format(sys.argv[1]),"a") as output:
        for key in stats.keys():
            output.write(str(key)+" ")
            json.dump(stats[key],output)
            output.write("\n")   

