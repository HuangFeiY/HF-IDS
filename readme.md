## TwoStageHybridIDS

#### introduction of each file

In the OCSVM folder, there are:

* the code to train the best model using grid search is stored: gridSearch.py 
* the validation code: eval_ocsvm_KDD.py

In the utils folder, there are Stored dataset pre-processing code

train_CVAE.py: the training code of our second fine-grained classification stage

hybrid_IDS.py: The complete code that implements our two-stage hierarchical approach

#### Usage

To begin with, you need to download NSL-KDD dataset from https://www.unb.ca/cic/datasets/nsl.html

Then you need use the code in utils folder to do some preprocess steps.

Third, you need to train an OCSVM model using the code in OCSVM folder, this model is used for our first stage, namely anomaly detection module. Since running gridSearch.py is time-consuming, you can just train an OCSVM model using the optimal Parameters presented in our paper.

Next, you need to train an CVAE model using the code train_CVAE.py, this model is used for our seconde stage, namely fine-grained classification module.

Last, you can run hybrid_IDS.py to do fine-grained classification for known/unknown intrusion detection. Remember to update the model_path in hybrid_IDS.py so that you can load the model you trained correctly.



