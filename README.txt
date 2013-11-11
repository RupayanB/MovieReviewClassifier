Name: Rupayan Basu
UNI: rb3034
Kaggle ID: Rupayan


Language used: Python 2.7.3
Number of files: 3 (NBC.py, train.py, test.py)
External libraries used: nltk and two nltk corpora for preprocessing. To install nltk and the required corpora please execute the following commands before running the programs (supervisor privileges necessary):
wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py -O - | python
python -m nltk.downloader stopwords
python -m nltk.downloader wordnet

Steps to run the programs:
1. To train the model:
         python train.py train.csv model
where train.csv is the training file, model is the file to which the Naive Bayes classifier will be saved as a serialized object (not human readable)
If the incorrect number of arguments are provided following msg will be displayed:
        Usage: python train.py trainingFile modelFile
Also, NaiveBayesProbs.dat is produced, which lists the conditional probabilities of the model in human readable format. 


2. To test the model:
        python test.py model test.csv submission.csv
where model is the same model file that was used to save the classifier model during training.
test.csv is the test data set and submission.csv is the prediction file.
If the incorrect number of arguments are provided, following msg will be displayed:
        Usage: python test.py modelFile testSet predictionFile
If the wrong model file is provided, the following error msg will be displayed:
        Invalid model file!