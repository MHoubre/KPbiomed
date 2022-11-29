# KP-biomed. A Large-Scale Dataset for Biomedical Keyphrase Generation

This is the repository with the code of the paper "A Large-Scale Dataset for Biomedical Keyphrase Generation" Maël Houbre, Florian Boudin, Béatrice Daille.

## Data preprocessing

First, download the 2021 PubMed/Medline baseline at: https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/

and put it in a directory named "baseline".

Then use the _data\_processing.sh_ script to preprocess the data.

You will get 5 jsonl files named train\_large, train\_medium, train\_small, test and val.

For fine tuning, use the fine\tuning.py script as follows:

```
python fine_tuning.py <size of the training split (small, large, medium)>
```

You may have to modify the output directory for your checkpoints (line 72, 75 and 114) depending on the model you want to train.

For the evaluation, use the evaluation script as follows:

```
python evaluation.py <evaluation mode (m for F1@M or 10 for F1@10)>
```
