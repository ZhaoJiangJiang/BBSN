# BBSN
BBSN for multi-label text classification with imbalanced data

## Dataset

+ AAPD
+ RCV1

You can obtain the original data from following link:  
[**AAPD**](https://github.com/EMNLP2019LSAN/LSAN)  
[**RCV1**](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm)

We give two examples in the folder **data/××/××_train.csv** file.  
You can run the **××_data_util.py** to generate the training files, include the following 5 files:  
+ ××_bbsn1.json
+ ××_bbsn2.json
+ ××_train.json
+ ××_valid.json 
+ ××_test.json  
Note: ×× means the aapd or rcv1

## Train
You can run the **aapd_bbsn.py**  or **rcv1_bbsn.py** to train the model and prediction the result for testing data, the result source will be output to the file named **aapd-bbsn.txt** and **rcv1_bbsn.txt** respectively.
