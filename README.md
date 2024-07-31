# KP-biomed. A Large-Scale Dataset for Biomedical Keyphrase Generation

This is the repository with the code of the paper Maël Houbre, Florian Boudin, and Beatrice Daille. 2022. A Large-Scale Dataset for Biomedical Keyphrase Generation. In Proceedings of the 13th International Workshop on Health Text Mining and Information Analysis (LOUHI).

## Data preprocessing

First, download the 2021 PubMed/Medline baseline currently available at: https://lhncbc.nlm.nih.gov/ii/information/MBR/Baselines/2021.html

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

## Cite
```
@inproceedings{houbre-etal-2022-large,
    title = "A Large-Scale Dataset for Biomedical Keyphrase Generation",
    author = {Houbre, Ma{\"e}l  and
      Boudin, Florian  and
      Daille, Beatrice},
    booktitle = "Proceedings of the 13th International Workshop on Health Text Mining and Information Analysis (LOUHI)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.louhi-1.6",
    pages = "47--53",
    abstract = "Keyphrase generation is the task consisting in generating a set of words or phrases that highlight the main topics of a document. There are few datasets for keyphrase generation in the biomedical domain and they do not meet the expectations in terms of size for training generative models. In this paper, we introduce kp-biomed, the first large-scale biomedical keyphrase generation dataset collected from PubMed abstracts. We train and release several generative models and conduct a series of experiments showing that using large scale datasets improves significantly the performances for present and absent keyphrase generation. The dataset and models are available online.",
}
```
