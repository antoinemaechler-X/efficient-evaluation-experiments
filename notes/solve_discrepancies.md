1. ALPHA = 0.05 vs Zrnic's alpha = 0.1 (ALL studies except ACS)

Not a problem

2. AlphaFold: Estimating GROUP MEANS vs Zrnic's ODDS RATIO

I dont understand, why did it suddenly changed when we were supposed to simply reproduce?
I want you to create a second version of alphafold (will keeping the first one) that will do this odds ratio thing.

3. Galaxy: Cross-PPI Without Actual Cross-Fitting

Keep as is for Y_hat, seems good to me. Use the whole dataset though, why dont we use the whole dataset? Please explore this.

4. CI Bound Clipping (bernoulli scripts)

Our approach WOR FAQ is not the same as Zrnic approach. So it is OK to use different method like clipping. Just make sure our reproduction of Zrnic results doesnt use this. If it does, change it to remove clipping for Zrnic results and be as close as what she is doing.

5. Torch var() ddof Mismatch (bernoulli scripts)

Modify this small error

6. Pew Budget Range Start

This doesnt matter, we keep starting at 1% in all our experiments

7. ACS: Budget Range and Number of Budgets

Doesnt matter, dont change.

8. ACS: XGBoost Hyperparameters

Doesnt matter, dont change.

9. Data Split Seeds (Pew)

Be consistent for the seed, always 42 no?

10. AlphaFold: Tau Not Tuned (but matches Zrnic)

Doesnt matter, dont change.

11. Coverage Target Consistency

Already addressed before, no change needed.

12. wor_baselines.py

Please indeed correct this.