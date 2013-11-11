import sys
import pickle
import NBC

if __name__ == '__main__':
	if len(sys.argv) == 4:
		modelFile = sys.argv[1]
		testFile = sys.argv[2]
		predictionFile = sys.argv[3]
		try:
			tmp = open(modelFile,'r')
			nbc = pickle.load(tmp)
			tmp.close()
		except:
			print "Invalid model file!"
			exit(1)

		print "Running tests...\n"
		nbc.testNBClassifier(testFile,predictionFile)
		print "Done.\n"
	else:
		print "Usage: python test.py modelFile testSet predictionFile"
		exit(1)
	