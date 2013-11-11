from __future__ import division
import math
import nltk
from nltk.corpus import stopwords
#Note! Required corpora - wordnet, stopwords

class NaiveBayesClassifier:

	def __init__(self):
		self.trainingFile = ''
		self.testFile = ''
		self.submissionFile = ''
		self.modelFile = ''
		self.trainDict = dict()
		self.labels = set()
		self.vocabulary = set()
		#parameters required for preprocessing:
		self.minLength = 2
		self.language = 'english'
		self.useStem = False
		self.useLemm = False
		self.removeStops = False
		#counts required for chi square:
		self.DocsCount = 0
		self.labelCount = dict()
		self.jointCounts = dict()
		#probs for NBC:
		self.labelPrior = dict()
		self.totalJointCount = dict()
		self.condProb = dict()
		self.sumTokenCounts = dict()

	def preprocess(self,sentence):
		tokens = nltk.word_tokenize(sentence)
		#remove words which are short and not useful
		if self.removeStops == True:
			tokens = [token for token in tokens if
			(token not in stopwords.words(self.language)) and len(token) >= self.minLength]
		i = 0
		porter = nltk.PorterStemmer()
		lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
		for token in tokens:
			#convert to lowercase
			tokens[i] = token.lower()
			#use only legitimate words from a dictionary
			if self.useLemm == True:
				tokens[i] = lemmatizer.lemmatize(token)
			#reduce words to their stem/root
			if self.useStem == True:
				tokens[i] = porter.stem(token)
			i += 1
		return tokens

	def preprocessTrain(self,trainingFile,useStem,useLemm,removeStops):
		self.trainingFile = trainingFile
		self.useStem = useStem
		self.useLemm = useLemm
		self.removeStops = removeStops

		train = open(self.trainingFile,'r')
		train.next()
		#initialize training dictionary
		#key = label, value = list of list of tokens
		for line in train:
			label, sentence = line.split(',',1)
			self.labels.add(label)
		for l in self.labels:
			self.trainDict[l] = list()
		#reset file read position
		train.seek(0)
		train.next()
		#begin sentence preprocessing
		porter = nltk.PorterStemmer()
		lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
			
		for line in train:
			label, sentence = line.split(',',1)
			tokens = nltk.word_tokenize(sentence)
			#remove words which are short and not useful
			if self.removeStops == True:
				tokens = [token for token in tokens if
				(token not in stopwords.words(self.language)) and len(token) >= self.minLength]
			i = 0
			seenTokens = set()
			for token in tokens:
				#convert to lowercase
				tokens[i] = token.lower()
				#use only legitimate words from a dictionary
				if self.useLemm == True:
					tokens[i] = lemmatizer.lemmatize(token)
				#reduce words to their stem/root
				if self.useStem == True:
					tokens[i] = porter.stem(token)
				
				t = tokens[i]
				self.vocabulary.add(t)
				if t not in seenTokens:
					if (t,label) not in self.jointCounts.keys():
						self.jointCounts[(t,label)] = 1
					else:
						self.jointCounts[(t,label)] += 1
					seenTokens.add(t)
				#update term freq of each token for each class
				
				if (t,label) not in self.totalJointCount.keys():
					self.totalJointCount[(t,label)] = 1
				else:
					self.totalJointCount[(t,label)] += 1
				i += 1

			listOfLists = self.trainDict[label]
			listOfLists.append(tokens)
			
			#Update total doc count
			self.DocsCount += 1
			
			#Count doc count for each label
			if label in self.labelCount.keys():
				self.labelCount[label] += 1
			else:
				self.labelCount[label] = 1
		train.close()
#try to pair tokens with pos tags for improved perf

	
	def chiSquareFeatureSelection(self):
		chiSquare = dict()
		label0 = self.labels.pop()
		label1 = self.labels.pop()
		for v in self.vocabulary:
			#calculate actual doc counts
			N11 = 0
			N10 = 0
			if (v,label1) in self.jointCounts.keys():
				N11 = self.jointCounts[(v,label1)]
			if (v,label0) in self.jointCounts.keys():
				N10 = self.jointCounts[(v,label0)]
			N01 = self.labelCount[label1] - N11
			N00 = self.labelCount[label0] - N10

			c1 = N11 + N10 + N01 + N00
			c2 = (N11*N00 - N10*N01)
			c2 = math.pow(c2,2)
			c3 = (N11+N01)*(N11+N10)*(N10+N00)*(N01+N00)
			if c3 != 0:
				chiSquare[v] = (c1 * c2)/c3
			else:
				chiSquare[v] = 0
		self.labels.add(label0)
		self.labels.add(label1)	

		for label in self.labels:
			for tlist in self.trainDict[label]:
				for v in tlist:
					#if chiSquare > 6.63, independence assumption 
					#can be rejected with 99% accuracy
					if chiSquare[v] <= 6.63:
						tlist.remove(v)

	def TrainNBClassifier(self):
		simpleModel = open('./NaiveBayesProbs.dat','w')
		
		for label in self.labels:
			self.sumTokenCounts[label] = len(self.vocabulary)
			self.labelPrior[label] = self.labelCount[label]/self.DocsCount

		for token in self.vocabulary:
			for label in self.labels:
				if (token,label) in self.totalJointCount.keys():
					self.sumTokenCounts[label] += self.totalJointCount[(token,label)] 

		for token in self.vocabulary:
			for label in self.labels:
				if (token,label) in self.totalJointCount.keys():
					self.condProb[(token,label)] = (self.totalJointCount[(token,label)]+1)/self.sumTokenCounts[label]
				else:
					self.condProb[(token,label)] = 1/self.sumTokenCounts[label]
				simpleModel.write('\n'+'p('+token+'|'+label+') = '+str(self.condProb[(token,label)]))
		simpleModel.close()

	def classify(self,tokens):
		maxScore = -999
		maxLabel = ''
		for label in self.labels:
			score = math.log10(self.labelPrior[label])
			for token in tokens:
				if token in self.vocabulary:
					score += math.log10(self.condProb[(token,label)])
				else:
					self.condProb[(token,label)] = 1/self.sumTokenCounts[label]
					score += math.log10(self.condProb[(token,label)])
			if score > maxScore:
				maxScore = score
				maxLabel = label
		return maxLabel

	def testNBClassifier(self,testFile,submissionFile):
		self.testFile = testFile
		self.submissionFile = submissionFile
		test = open(self.testFile,'r')
		test.next()
		sub = open(self.submissionFile,'w')
		sub.write('Id,Category')
		i = 1
		for line in test:
			tokens = self.preprocess(line)
			label = self.classify(tokens)
			sub.write('\n'+str(i)+','+label)
			i = i + 1
			
		sub.close()
		test.close()



		

		





