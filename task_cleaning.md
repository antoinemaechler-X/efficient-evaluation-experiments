Here are general instructions from the phd student about the different datasets we worked with (alphafold, galaxy, deforestation, pew):

"Can you clean up all datasets + experiments into per-dataset folders and send them to me with manual documentation by this Friday? Ideally, a README helping navigate through your files would be very helpful. Having a list of datasets + their statuses in the README would be extremely helpful."

It is important we do this together step by step so that I control what you are doing.
We already started in cleaned_processes/ folder but 1) there are a few missing and 2) I want to control exactly what we are doing / check the existing ones.
Datsets is clear. Experiments here is for example did we used WOR on it, only WR FAQ, which version of active inference? Figure it out mainly yourself what it means.

Here are the steps that I want:

1) assess the situation: scan the whole repo and understand all we need to do. We will validate together what we do (for example I wont deal with acs_study) ie which datasets / experiments.
2) then go dataset by dataset. Reproducing what we did for alphafold study (cleaned_processes/alphafold_study) is I think a good goal (even if itself it is not complete / has updates)
    - have a README for each folder (each folder = each dataset)
    - have files to run our experiments + to download data
    - have a folder "results" where we have existing results, and a jupyter notebook where you can play with those results
    - have an environment.yml 
    - have a file to submit on marlowe cluster
Basically exactly like the alphafold_study folder. Goal is to have perfect reproducibility. You will write a first version of documentation (README) and then at the end we will discuss it together.
Do all of this in a new folder clean_experiments/ We start from scratch.

3) provide an overall README that gives status about all datasets and experiments.

Make the code and just all we do look a bit human and not AI-generated.