#!/usr/bin/env python
# coding: utf-8

from gc import callbacks
import json
import sys
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from transformers import BartModel, BartForConditionalGeneration
from transformers import BartTokenizerFast
from transformers import Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq

from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, ReadInstruction
from datasets import load_metric


# Loading training dataset.
dataset = load_dataset("json", data_files={"train":"train_"+ sys.argv[1] +".jsonl", "validation" : "val.jsonl"})

def join_keyphrases(dataset):
    dataset["keyphrases"] = " <KP> ".join(dataset["keyphrases"])
    return dataset

# Making the references sequences
dataset = dataset.map(join_keyphrases,num_proc=8,desc="Putting all keyphrases in a single sequence separated by <KP>")

# Loading the model
tokenizer = BartTokenizerFast.from_pretrained("GanjinZero/biobart-base")


# ----------------- FOR SPECIAL TOKENS -----------------------
special_token = {"additional_special_tokens":['<KP>']}
num_added_toks = tokenizer.add_special_tokens(special_token) ##This line is updated
#-------------------------------------------------------------
print(tokenizer.all_special_tokens)

# Getting the text from the title and the abstract
def get_text(dataset):
    dataset["text"] = dataset["title"] + ". " + dataset["abstract"]
    return dataset

# Function to tokenize the text using Huggingface tokenizer
def preprocess_function(dataset):

    model_inputs = tokenizer(
        dataset["text"],max_length= 512,truncation=True
    )
    
    with tokenizer.as_target_tokenizer():
    
        labels = tokenizer(
            dataset["keyphrases"], max_length= 128,truncation=True)
        

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    


dataset = dataset.map(get_text,num_proc=10, desc="Getting full text (title+abstract)")

tokenized_datasets= dataset.map(preprocess_function, batched=True, num_proc = 10, desc="Running tokenizer on dataset")

tokenized_datasets.set_format("torch")

# Training arguments
#--------------------------------------------------------------------------------------------
batch_size = 12
num_train_epochs = 5
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = "fine-tuning"

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-biobart_medium",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    optim="adamw_torch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=10,
    num_train_epochs=num_train_epochs,
    logging_steps=logging_steps,
    push_to_hub=False,
    #load_best_model_at_end=True
)
#-------------------------------------------------------------------------------------------

model = BartForConditionalGeneration.from_pretrained("GanjinZero/biobart-base")
model.resize_token_embeddings(len(tokenizer)) #because we added the <KP> token so we need to update the vocab size

data_collator = DataCollatorForSeq2Seq(tokenizer, model)


tokenized_datasets = tokenized_datasets.remove_columns(
    dataset["train"].column_names
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset = tokenized_datasets["train"], #mettre le train
    eval_dataset = tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=None,
    #callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    #compute_metrics=compute_metrics,
)


trainer.train()
trainer.save_model("fine-tuning-biobart_" + sys.argv[1] +"/final_model")

