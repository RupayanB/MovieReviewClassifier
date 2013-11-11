import sys
import pickle
from NBC import NaiveBayesClassifier

if __name__ == '__main__':
	if len(sys.argv) == 3:
		trainingFile = sys.argv[1]
		modelFile = sys.argv[2]

		nbc = NaiveBayesClassifier()
		print "Preprocessing training corpus...\n"
		#bool params = useStem, useLemm, removeStop
		nbc.preprocessTrain(trainingFile,True,False,False)
		print "Performing feature selection...\n"
		nbc.chiSquareFeatureSelection()
		print "Training Naive Baise Classifier...\n"
		nbc.TrainNBClassifier()

		tmp = open(modelFile,'w')
		pickle.dump(nbc,tmp)
		tmp.close()
	else:
		print "Usage: python train.py trainingFile modelFile\n"
		exit(1)
	