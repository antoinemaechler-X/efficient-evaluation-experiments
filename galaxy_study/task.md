See what we did for alphafold in the folder alphafold_study/
We used the paper and github https://github.com/tijana-zrnic/active-inference.git where they were using alphafold as one of the dataset, and reproduced the results here. We then adapted our method (FAQ PAI without replacement) to  finally compare: classical, Active inference and WOR Active.
We couldnt exactly reproduce the FAQ method with a factor model because we had no historical data but still had a model to predict and reach better performance.

I want to reproduce this but with another dataset: Galaxy Zoo 2, in the paper Cross PPI: https://arxiv.org/abs/2309.16598 with the associated Github: https://github.com/tijana-zrnic/cross-ppi

The goal is at the end to have:
- the folder galaxy_study/ built similarly to alphafold_study/ with the data from the paper, reproducing their results etc
- Our method WOR FAQ/PAI adapted to it as closely as possible. Obviously you keep the estimator, the main question is about the factor model. Think and do what you think is best (maybe trying to predict with a simpler ML model?). For alphafold I was using alphafold predictions.
- the necessary files to launch the study (ie having at the end baseline vs cross-ppi results vs our results plot) on the marlowe cluster. Also add the file in the folder sherlock/ so that I can also launch it on the sherlock cluster.

Stay as close as possible (in term of saving the data, the structure, the way to compute the results that kind of things) to what is already existing, which is of excellent quality.