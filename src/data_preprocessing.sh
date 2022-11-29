python get_pubmed_data.py baseline kpmed.jsonl

python get_recent_abstracts.py

rm kpmed.jsonl

python get_correct_form_kps.py data

rm kpmed_10y.jsonl

python prmu.py

rm data_correct_form.jsonl

python data_split.py

rm data_prmu.jsonl

