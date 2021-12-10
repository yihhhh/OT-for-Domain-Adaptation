# Large-Scale Optimal Transport for Domain Adaptation
A pytorch re-implement of [Large-Scale Optimal Transport and Mapping Estimation](https://arxiv.org/abs/1711.02283) and apply this method to Domain Adaptation.
## Code Structure
The structure of the code is as follows:
```
|- configs                  # stores the configuration files for training and inference
    |- pretrain_classifier.yml
    |- train_classifier.yml
    |- train_mapping.yml
|- dataset
    |- mapping              # stores the data that used for mapping
    |- mnist                # stores the mnist data
    |- usps                 # stores the usps data
    |- data_process.py      # script for processing the mnist and usps dataset
|- models                   # stores the model python files used in training and inference
    |- __init__.py    
    |- cnn.py               # a simple cnn model for digit classification task
    |- mapping.py           # model used for train the optimal mapping
    |- ot_model.py          # model used for train the optimal tranport plan
|- dataset.py               # Script for constructing Dataset class
|- infer_classifier.py      # Script for CNN inference on test data
|- infer_mapping.py         # Script for mapping inference
|- numerical_example.py     # srcipt for double circle and double moon numerical examples
|- pretrain_classifier.py   # Using learned mapping to pretrain a digit classifier on mnist
|- train_classifier.py      # Using the pretrained model to finetune a digit classifer on usps
|- train_knn.py             # Using the learned mapping to train a knn on mnist and test on usps
|- train_mapping.py         # Script for train a mapping
|- utils.py                 # Some functions that are commonly and universally used in this code
```