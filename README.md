The Code folder contains all scripts for Log-Sum-Exponential Estimator. The code for this study is written in Python and Pytorch library is used to train models.

# Code Structure:

* preprocess_raw_dataset_from_model.py: The code to generate the base pre-processed version of the datasets with raw input values.
* preprocess_feature_dataset_from_model.py: The code to generate the base pre-processed version of the datasets with pre-trained features.
* The data folder consists of any potentially generated bandit dataset (which can be generated by running the scripts in code).
* The Code/code folder contains the scripts and codes written for the experiments:
    * requirements.txt contains the Python libraries required to reproduce our results.
    * readme.md includes the syntax of different commands in the code.
    * accs: A folder containing the result reports of different experiments.
    * saved_logs: Training log for different experiments.
    * data.py code to load data for image datasets.
    * eval.py & eval_rec2.py code to evaluate estimators for image datasets and open bandit dataset.
    * config: Contains different configuration files for different setups.
    * runs: Folder containing different batch running scripts.
    * loss.py: Script of our loss functions including LSE and α-LSE.
    * train_logging_policy.py: Script to train the logging policy
    * create_bandit_dataset.py: Code for the generation of the bandit dataset using the logging policy.
    * main_semi_ot.py: Main training code which implements different methods proposed by our paper.
    * main_semi_rec2.py: Main training code for Open Bandit dataset.
* The prepare_real_data folder contains the scripts and codes written for open bandit dataset:
    * create.ipynb: The notebook for preparing open bandit dataset.
    * data: A folder containing open bandit dataest data.
 
# How to Run:
* First, the user needs to download and store the raw dataset using preprocess_raw_dataset_from_model.py.
  > code
