I work with a PhD student, for who I modified his repo to run a bunch of new experiments. 
His repo is: https://github.com/skbwu/efficiently-evaluating-llms

He wants to see "my notebooks", meaning the work I did on (1) ACS dataset (in acs_study folder) and (2) the Without replacement of FAQ on the original LLM evaluation dataset.

I want you to create 2 folders in the folder cleaned_processed, one for each task.

In each folder, I want:
- a quick README
- the main code, cleaned:
    - for ACS, I want the different py files that I would run on the cluster to get the csv results (I think it was some csv results right?) Then I want a very complete jupyter notebook that can be easily run to reproduce all results, where we see which models I tested and used, explains the different step of exploration (setup, getting plots just to understand the data, the intermediary models we use, BLR, then BLR with fewer labels or larger dimension of xgboost...) The goal at the end is that he has something understandable and easily reproducible: Like I give him the results that ran on the cluster (I pulled them here on local) and he can manipulate, play etc...
    - for Without replacement, I want the exact equivalent of the py files created in the original repo, I want the PhD student to be able to see a file and be like "this is the in-place replacement of XX file"(look online at the original repo https://github.com/skbwu/efficiently-evaluating-llms). You can then add a ipynb where we can plot and analyse the data once we have the csv results (I didnt uploaded them from the cluster yet for FAQ WOR but will do), doing a bit the role of wor_cleaning_results.py and wor_plot_figure.py
    - for both, you can use auxiliary py files to define general functions
- the results, both csv and png (you can copy the one we already have in it). In the jupyter notebook, I will run it to have the results in it. 

also he wont have the data so if he needs to download it, make sure to create something to help him.

The goal is to give this at the end to the reviewers of the paper, so I want something smooth and high quality but not too much fluff: just reproduce the results in a clean way.