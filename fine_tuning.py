#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gc import callbacks
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BartConfig
from transformers import Seq2SeqTrainingArguments, EarlyStoppingCallback
from transformers import BartModel, BartForConditionalGeneration
from transformers import BartTokenizerFast
import datasets
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, ReadInstruction
from tokenizers import Tokenizer
from datasets import load_metric


import numpy as np
import pandas as pd
import random


# In[2]:


dataset = load_dataset("json", data_files={"train":"train_medium.jsonl", "validation" : "val.jsonl"})


# In[3]:


# In[4]:


def join_keyphrases(dataset):
    dataset["keyphrases"] = " <KP> ".join(dataset["keyphrases"])
    return dataset


# In[5]:



# In[6]:


#dataset = dataset.map(tokenize_keyphrases)
dataset = dataset.map(join_keyphrases,num_proc=8,desc="Putting all keyphrases in a single sequence separated by <KP>")




# In[9]:


tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
print(tokenizer.all_special_tokens)

# ----------------- FOR SPECIAL TOKENS -----------------------
special_token = {"additional_special_tokens":['<KP>']}
num_added_toks = tokenizer.add_special_tokens(special_token) ##This line is updated
#-------------------------------------------------------------

# In[10]:


def get_text(dataset):
    dataset["text"] = dataset["title"] + ". " + dataset["abstract"]
    return dataset


# In[11]:


def preprocess_function(dataset):

    model_inputs = tokenizer(
        dataset["text"],max_length= 512,truncation=True
    )
    
    with tokenizer.as_target_tokenizer():
    
        labels = tokenizer(
            dataset["keyphrases"], max_length= 128,truncation=True)
        

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    


# In[12]:


#tokenized_datasets = DatasetDict()
dataset = dataset.map(get_text,num_proc=10, desc="Getting full text (title+abstract)")


# In[13]:


#dataset
#tokenized_datasets["train"]= dataset["test"].map(preprocess_function,batched=True)
#dataset["train"].keys()

tokenized_datasets= dataset.map(preprocess_function, batched=True, num_proc = 10, desc="Running tokenizer on dataset")

tokenized_datasets.save_to_disk("tokenized_dataset")

# In[15]:


tokenized_datasets.set_format("torch")


# In[16]:


from transformers import Seq2SeqTrainingArguments


# In[17]:


batch_size = 12
num_train_epochs = 10
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = "fine-tuning"#model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-kpmed",
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
    load_best_model_at_end=True
)


# In[18]:

# In[20]:


from transformers.data.data_collator import DataCollator
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, DataCollatorWithPadding

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
model.resize_token_embeddings(len(tokenizer)) #because we added the <KP> token

data_collator = DataCollatorForSeq2Seq(tokenizer, model)


# In[21]:


tokenized_datasets = tokenized_datasets.remove_columns(
    dataset["train"].column_names
)


# In[22]:


from transformers import Seq2SeqTrainer

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


# In[ ]:


trainer.train()
trainer.save_model("final_model")

