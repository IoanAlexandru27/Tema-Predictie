from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
import json
import random







def ReadReviews(filename): # Citeste reviewuri
	ret = []
	f = open(filename, encoding="utf8")
	data = json.load(f)
	return data


def fcn(e):
	return e[1]

def Get_KeyWords(data, nr=500): # Determinare cuvinte cheie
	Ocur = dict()
	Tokens = dict()
	tok = 0
	max_len = 0
	for el in data:
		if max_len < len(el['text']):
			max_len = len(el['text'])
		for word in el['text'].split(' '):
			if word not in Ocur:
				Ocur[word] = 1
			else:
				Ocur[word] += 1

	aux = []
	for item in Ocur.items():
		aux.append(item)
	aux.sort(reverse=True,key=fcn)
	for i in aux:
		if len(i[0]) > 2:
			tok += 1
			Tokens[i[0]] = tok
		if tok > nr:
			return Tokens, max_len

def TokenizeSentence(tok_dict, text, padding = 512): # Tokenizare propozitii
	ret = padding * [0]
	i = -1
	for word in text.split(' '):
		i += 1
		if word in tok_dict.keys():
			ret[i] = tok_dict[word]
		if i == padding - 1:
			break
	return ret

class NNModel:
	def __init__(self, training_data):

		#Initializare
		l = len(training_data)

		training = list(training_data[:int(l*80/100)])
		self.training_data_x = list()
		self.training_data_y = list()
		for el in training:
			self.training_data_x.append(el[0])
			self.training_data_y.append(el[1])		

		training = list(training_data[int(l*80/100):])
		self.test_x = list()
		self.test_y = list()
		for el in training:
			self.test_x.append(el[0])
			self.test_y.append(el[1])
		self.model = Sequential()

	def NNConstruction(self):	
		self.model.add(Dense(512, input_shape=(len(self.training_data_x[0]),), activation='relu'))
		#self.model.add(Dropout(0.5))
		self.model.add(Dense(1024, activation='relu'))
		#self.model.add(Dropout(0.2))
		self.model.add(Dense(256, activation='relu'))
		self.model.add(Dense(256, activation='relu'))
		self.model.add(Dense(256, activation='relu'))
		#self.model.add(Dropout(0.3))
		self.model.add(Dense(6, activation='softmax'))
		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

	def fitModel(self): # Antrenare
		hist = self.model.fit(np.array(self.training_data_x), np.array(self.training_data_y), epochs=20, batch_size=100, verbose=1)
		self.model.save('ReviewPredicter.h5', hist)

def GetOutput(x):
	ret = list(6 * [0])
	ret[x] = 1
	return ret


def PredictTrain(filename, netname, filetosaveto):
	data = ReadReviews(filename)
	tokens, max_len = Get_KeyWords(data)
	input_data = list()
	for i in data: 
		input_data.append((TokenizeSentence(tokens, i['text']), GetOutput(i['rating'])))
	model = load_model(netname)

	test_x = list()
	for el in input_data:
		test_x.append(el[0])
	preds = model.predict(test_x)

	for pred, rev in zip(preds, data):
		rev['rating'] = int(np.argmax(pred))

	with open(filetosaveto, 'w') as f:
		json.dump(data, f)


def main():
	data = ReadReviews("train.json")
	tokens, max_len = Get_KeyWords(data)
	input_data = list()
	for i in data: 
		input_data.append((TokenizeSentence(tokens, i['text']), GetOutput(i['rating'])))


	l = len(input_data)
	training = list(input_data[int(l*80/100):])
	test_x = list()
	test_y = list()
	for el in training:
		test_x.append(el[0])
		test_y.append(el[1])

	model = NNModel(input_data)
	model.NNConstruction()
	model.fitModel()

	model = load_model('ReviewPredicter.h5')

	PredictTrain("test_wor.json", "ReviewPredicter.h5", "results.json")


if __name__ == '__main__':
	main()