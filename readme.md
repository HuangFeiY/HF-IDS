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


#### Copyright

Copyright (c) 2023 Feiyang Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
